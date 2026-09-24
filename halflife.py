#!/usr/bin/env python3
"""
halflife.py - Nuclear Half-Life & Lifetime Spectrum Fitting Tool
==============================================================================
Based on halflife.c (Carl Wheldon 2009 / F. Riess 1987).

Fits a prompt Gaussian peak convoluted with an exponential decay, supporting:
  - Right-tail exponential decay (t_1/2 > 0)
  - Left-tail exponential decay (t_1/2 < 0)
  - Prompt-only symmetric Gaussian (t_1/2 = 0)
  - Pure exponential decay (FWHM = 0)
  - Constant baseline background offset (fixed, fitted, or scanned via chi^2)

Supports ASCII .dat files (1, 2, 3, or 4 columns) exported from python-cmat
and cmat3d webviewers, taking into account statistical uncertainties and
propagated background-subtraction errors.

Provides:
  1. Python API (HalfLifeFitter class)
  2. Scriptable CLI (for macros and automated pipelines)
  3. Interactive Terminal CLI (matching and enhancing halflife.c menu modes)
==============================================================================
"""

import sys
import os
import math
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import scipy.optimize
import scipy.special

# Optional matplotlib for PDF/PNG export and GUI preview
try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# 1. MATHEMATICAL PHYSICS CONVOLUTION MODEL
# ==============================================================================

def erfpol_new(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Abramowitz & Stegun approximation 7.1.26 for exp(x^2)*erfc(x) = erfcx(x).
    Retained for reference and compatibility with halflife.c.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + 1.061405429 * t))))
    return poly


def eval_halflife(
    x: np.ndarray,
    t12: float,
    fwhm: float,
    centroid: float,
    scale: float,
    bg: float = 0.0
) -> np.ndarray:
    """
    Analytical convolution of a Gaussian prompt peak with an exponential decay:
        convo(x) = ∫ exp(t/tau) * exp(-(x - centroid - t)^2 / sig^2) dt / (sqrt(pi)*sig)
    
    Parameters:
        x        : Channel or energy coordinate array
        t12      : Half-life (positive = right decay, negative = left decay, 0 = prompt only)
        fwhm     : FWHM of prompt peak Gaussian response (FWHM = 2*sqrt(ln(2))*sigma_g)
        centroid : Prompt peak centroid position x0
        scale    : Scaling factor / total integrated area of convoluted component
        bg       : Constant baseline background offset
    
    Returns:
        Fitted spectrum values evaluated at x.
    """
    x = np.asarray(x, dtype=np.float64)
    x1 = x - centroid
    
    # Degenerate case 1: Prompt-only Gaussian (t12 == 0)
    if abs(t12) < 1e-9:
        sigma_g = max(abs(fwhm) / (2.0 * np.sqrt(np.log(2.0))), 1e-9)
        return scale * np.exp(-(x1 / sigma_g)**2) / (np.sqrt(np.pi) * sigma_g) + bg

    tau = -t12 / np.log(2.0)
    abs_tau = abs(tau)

    # Degenerate case 2: Pure exponential decay (fwhm == 0)
    if abs(fwhm) < 1e-9:
        fit = np.zeros_like(x)
        if tau < 0:  # t12 > 0: right-side decay (x1 >= 0)
            mask = x1 >= 0
            fit[mask] = np.exp(x1[mask] / tau) / abs_tau
        else:        # t12 < 0: left-side decay (x1 <= 0)
            mask = x1 <= 0
            fit[mask] = np.exp(x1[mask] / tau) / abs_tau
        return scale * fit + bg

    # General case: Gaussian convoluted with Exponential decay
    sigma_g = max(abs(fwhm) / (2.0 * np.sqrt(np.log(2.0))), 1e-9)
    ep = (0.5 * sigma_g) / tau
    
    # Dimensionless argument u for erfc
    u_orig = (x1 / sigma_g) + ep
    if tau < 0:
        u_orig = -u_orig

    fit = np.zeros_like(x)
    # Use scipy.special.erfcx(u) = exp(u^2)*erfc(u) to prevent floating-point overflow
    # for positive and moderately negative u
    mask_erfcx = u_orig >= -25.0
    if np.any(mask_erfcx):
        u_sub = u_orig[mask_erfcx]
        x1_sub = x1[mask_erfcx]
        fit[mask_erfcx] = 0.5 / abs_tau * scipy.special.erfcx(u_sub) * np.exp(-(x1_sub / sigma_g)**2)

    # In deep exponential decay tail (u_orig < -25), erfc(u) -> 2.0
    mask_tail = ~mask_erfcx
    if np.any(mask_tail):
        arg_tail = x1[mask_tail] / tau + ep * ep
        arg_tail = np.clip(arg_tail, -500.0, 700.0)
        fit[mask_tail] = (1.0 / abs_tau) * np.exp(arg_tail)

    return scale * fit + bg


# ==============================================================================
# 2. DATA I/O & ERROR ESTIMATION
# ==============================================================================

class SpectrumData:
    """Encapsulates 1D spectrum data, calibrated coordinates, errors, and metadata."""
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: Optional[np.ndarray] = None,
        x_energy: Optional[np.ndarray] = None,
        filename: str = "",
        header_lines: Optional[List[str]] = None,
        is_gated: bool = False,
        bg_scale_factor: float = 0.0,
        x_label: str = "Channel"
    ):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if dy is None:
            # Poisson statistical uncertainty: sqrt(max(y, 1.0))
            self.dy = np.sqrt(np.maximum(self.y, 1.0))
        else:
            self.dy = np.asarray(dy, dtype=np.float64)
            # Ensure no zero/negative uncertainties
            self.dy = np.where(self.dy <= 0.0, 1.0, self.dy)
            
        self.x_energy = np.asarray(x_energy, dtype=np.float64) if x_energy is not None else None
        self.filename = filename
        self.header_lines = header_lines or []
        self.is_gated = is_gated
        self.bg_scale_factor = bg_scale_factor
        self.x_label = x_label

    def __len__(self) -> int:
        return len(self.x)

    def copy(self) -> "SpectrumData":
        return SpectrumData(
            x=self.x.copy(),
            y=self.y.copy(),
            dy=self.dy.copy(),
            x_energy=self.x_energy.copy() if self.x_energy is not None else None,
            filename=self.filename,
            header_lines=list(self.header_lines),
            is_gated=self.is_gated,
            bg_scale_factor=self.bg_scale_factor,
            x_label=self.x_label
        )


def load_ascii_spectrum(
    filepath: Union[str, Path],
    col_x: Optional[int] = None,
    col_y: Optional[int] = None,
    col_dy: Optional[int] = None,
    use_energy: bool = False,
    suppress_zeros: bool = False
) -> SpectrumData:
    """
    Load 1D spectrum from ASCII .dat or .txt file.
    
    Intelligently handles:
      - cmat_webviewer 3-col: '# Channel Energy_keV Counts'
      - cmat_webviewer 4-col: '# Channel Energy_keV Counts Error'
      - halflife.c 3-col: 'x y dy'
      - 2-col: 'x y' or 'Channel Counts'
      - 1-col: 'y'
      - Comment lines starting with '#'
      - Propagated background subtraction errors from '# Background Gates ... (Scale factor: s)'
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Spectrum file not found: {filepath}")

    header_lines = []
    data_rows = []
    is_gated = False
    bg_scale = 0.0
    header_col_names = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(stripped)
                lower_hdr = stripped.lower()
                if "gated coincidence cut" in lower_hdr or "peak gates" in lower_hdr:
                    is_gated = True
                if "scale factor:" in lower_hdr:
                    try:
                        part = stripped.split("Scale factor:")[-1].split(")")[0].strip()
                        bg_scale = float(part)
                    except Exception:
                        pass
                if "channel" in lower_hdr and "counts" in lower_hdr:
                    # Header line defining columns
                    tokens = [t.strip("#").strip() for t in stripped.split()]
                    header_col_names = [t for t in tokens if t]
                continue

            parts = stripped.replace(",", " ").replace("\t", " ").split()
            try:
                nums = [float(p) for p in parts]
                if nums:
                    data_rows.append(nums)
            except ValueError:
                continue

    if not data_rows:
        raise ValueError(f"No numeric data found in file: {filepath}")

    arr = np.array(data_rows, dtype=np.float64)
    ncols = arr.shape[1]

    x_col_idx = 0
    y_col_idx = 1
    dy_col_idx = None
    x_energy_arr = None
    x_label = "Channel"

    # Determine columns automatically if not specified
    if col_x is not None and col_y is not None:
        x_col_idx = col_x
        y_col_idx = col_y
        dy_col_idx = col_dy
    elif ncols == 1:
        # Single column: y values, x is 0, 1, 2...
        x_arr = np.arange(len(arr), dtype=np.float64)
        y_arr = arr[:, 0]
        dy_arr = np.sqrt(np.maximum(y_arr, 1.0))
        return SpectrumData(x_arr, y_arr, dy_arr, filename=filepath.name, header_lines=header_lines)
    elif ncols == 2:
        x_col_idx = 0
        y_col_idx = 1
        dy_col_idx = None
    elif ncols == 3:
        # Check if 3-column cmat export (Channel, Energy, Counts) vs classic (x, y, dy)
        has_energy_hdr = any("energy" in h.lower() for h in header_lines)
        if has_energy_hdr or (len(header_col_names) == 3 and "energy" in header_col_names[1].lower()):
            # Column 0: Channel, Column 1: Energy, Column 2: Counts
            if use_energy:
                x_col_idx = 1
                x_label = "Energy (keV)"
            else:
                x_col_idx = 0
                x_label = "Channel"
            y_col_idx = 2
            x_energy_arr = arr[:, 1]
            dy_col_idx = None
        else:
            # Classic halflife.c 3-column format: x, y, dy
            x_col_idx = 0
            y_col_idx = 1
            dy_col_idx = 2
    elif ncols >= 4:
        # 4-column: Channel, Energy, Counts, Error
        if use_energy:
            x_col_idx = 1
            x_label = "Energy (keV)"
        else:
            x_col_idx = 0
            x_label = "Channel"
        y_col_idx = 2
        dy_col_idx = 3
        x_energy_arr = arr[:, 1]

    x_arr = arr[:, x_col_idx]
    y_arr = arr[:, y_col_idx]

    if dy_col_idx is not None and dy_col_idx < ncols:
        dy_arr = arr[:, dy_col_idx]
    else:
        # Calculate Poisson or propagated background-subtracted error
        if is_gated and bg_scale > 0:
            # Propagated error with background subtraction scale factor
            dy_arr = np.sqrt(np.maximum(np.abs(y_arr), 1.0) * (1.0 + bg_scale))
        else:
            dy_arr = np.sqrt(np.maximum(y_arr, 1.0))

    if suppress_zeros:
        mask = y_arr > 0
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        dy_arr = dy_arr[mask]
        if x_energy_arr is not None:
            x_energy_arr = x_energy_arr[mask]

    return SpectrumData(
        x=x_arr,
        y=y_arr,
        dy=dy_arr,
        x_energy=x_energy_arr,
        filename=filepath.name,
        header_lines=header_lines,
        is_gated=is_gated,
        bg_scale_factor=bg_scale,
        x_label=x_label
    )


def save_fit_file(
    filepath: Union[str, Path],
    inname: str,
    spec: SpectrumData,
    fit_result: Dict[str, Any]
) -> None:
    """
    Write standard .fit ASCII format matching halflife.c output format.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    t12 = fit_result["t12"]
    t12_err = fit_result["t12_err"]
    fwhm = fit_result["fwhm"]
    fwhm_err = fit_result["fwhm_err"]
    cent = fit_result["centroid"]
    cent_err = fit_result["centroid_err"]
    scale = fit_result["scale"]
    scale_err = fit_result["scale_err"]
    bg = fit_result["bg"]
    bg_err = fit_result.get("bg_err", 0.0)
    bg_fixed = fit_result.get("bg_fixed", True)
    chisq = fit_result["chisq_ndf"]
    
    free_flags = fit_result.get("freepars", [True, True, True, True, not bg_fixed])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Input file: {inname}\n")
        f.write(f"# Output file: {filepath.name}\n")
        f.write(f"# Initial guess t_1/2 = {fit_result.get('init_t12', t12):.2f}\n")
        f.write(f"# Initial guess FWHM = {fit_result.get('init_fwhm', fwhm):.2f}\n")
        f.write(f"# Initial guess centroid = {fit_result.get('init_centroid', cent):.2f}\n")
        f.write(f"# Initial scaling factor = {fit_result.get('init_scale', scale):.2f}\n")
        f.write(f"#++++++++++++Results of fit++++++++++++\n")
        
        t12_status = f"+/- {t12_err:.3f}" if free_flags[0] else "FIXED"
        f.write(f"# t_1/2 = {t12:.3f} {t12_status}\n")
        
        fwhm_status = f"+/- {fwhm_err:.3f}" if free_flags[1] else "FIXED"
        f.write(f"# FWHM = {fwhm:.3f} {fwhm_status}\n")
        
        cent_status = f"+/- {cent_err:.2f}" if free_flags[2] else "FIXED"
        f.write(f"# Centroid = {cent:.2f} {cent_status}\n")
        
        scale_status = f"+/- {scale_err:.2f}" if free_flags[3] else "FIXED"
        f.write(f"# Scaling factor = {scale:.2f} {scale_status}\n")
        
        bg_status = f"+/- {bg_err:.2f}" if (len(free_flags) > 4 and free_flags[4]) else "(not fitted / fixed)"
        f.write(f"# Background level = {bg:.2f} {bg_status}\n")
        f.write(f"# Chisqr/D.O.F. = {chisq:.3f}\n")
        f.write(f"#++++++++++++++++++++++++++++++++++++++\n")
        f.write(f"#   Chan \t    fit\n")

        x_vals = fit_result.get("x_eval", spec.x)
        y_fit = fit_result.get("y_fit", eval_halflife(x_vals, t12, fwhm, cent, scale, bg))
        for x_v, y_v in zip(x_vals, y_fit):
            f.write(f"{x_v:8.1f}\t{y_v:8.3f}\n")


# ==============================================================================
# 3. CORE HALFLIFE FITTER ENGINE (SCIPY)
# ==============================================================================

class HalfLifeFitter:
    """
    High-performance nuclear lifetime spectrum fitting engine using SciPy.
    """
    def __init__(self):
        self.spec: Optional[SpectrumData] = None
        self.active_range: Optional[Tuple[float, float]] = None
        
        # Parameters: [t12, fwhm, centroid, scale, bg]
        self.pars: List[float] = [20.0, 15.0, 0.0, 1000.0, 0.0]
        self.errs: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.freepars: List[bool] = [True, True, True, True, True]  # background is free by default
        self.last_fit_result: Optional[Dict[str, Any]] = None
        self.last_scan_result: Optional[Dict[str, Any]] = None

    def load_data(
        self,
        filepath: Union[str, Path],
        col_x: Optional[int] = None,
        col_y: Optional[int] = None,
        col_dy: Optional[int] = None,
        use_energy: bool = False,
        suppress_zeros: bool = False
    ) -> SpectrumData:
        """Load spectrum data from an ASCII .dat/.txt file."""
        self.spec = load_ascii_spectrum(
            filepath,
            col_x=col_x,
            col_y=col_y,
            col_dy=col_dy,
            use_energy=use_energy,
            suppress_zeros=suppress_zeros
        )
        self.active_range = (float(np.min(self.spec.x)), float(np.max(self.spec.x)))
        self.auto_guess()
        return self.spec

    def set_data(self, x: np.ndarray, y: np.ndarray, dy: Optional[np.ndarray] = None, x_label: str = "Channel") -> SpectrumData:
        """Directly set in-memory spectrum arrays."""
        self.spec = SpectrumData(x=x, y=y, dy=dy, x_label=x_label)
        self.active_range = (float(np.min(self.spec.x)), float(np.max(self.spec.x)))
        self.auto_guess()
        return self.spec

    def compress(self, factor: int = 2) -> SpectrumData:
        """Compress spectrum data by an integer binning factor (matching Mode 5)."""
        if self.spec is None or factor < 2:
            return self.spec
        
        n_orig = len(self.spec.x)
        n_new = n_orig // factor
        if n_new < 4:
            raise ValueError(f"Cannot compress {n_orig} points by factor of {factor}")

        x_new = np.zeros(n_new, dtype=np.float64)
        y_new = np.zeros(n_new, dtype=np.float64)
        dy_new = np.zeros(n_new, dtype=np.float64)

        for i in range(n_new):
            idx_start = i * factor
            idx_end = idx_start + factor
            x_new[i] = np.mean(self.spec.x[idx_start:idx_end])
            y_new[i] = np.sum(self.spec.y[idx_start:idx_end])
            dy_new[i] = np.sqrt(np.sum(self.spec.dy[idx_start:idx_end]**2))

        self.spec = SpectrumData(
            x=x_new,
            y=y_new,
            dy=dy_new,
            filename=self.spec.filename + f"_x{factor}",
            header_lines=self.spec.header_lines,
            x_label=self.spec.x_label
        )
        self.active_range = (float(np.min(self.spec.x)), float(np.max(self.spec.x)))
        
        # Adjust initial guesses accordingly
        self.pars[0] /= factor
        self.pars[1] /= factor
        self.pars[2] /= factor
        self.pars[3] *= factor
        self.pars[4] *= factor
        return self.spec

    def auto_guess(self) -> Dict[str, float]:
        """
        Estimate reasonable initial fit parameters from active spectrum data.
        """
        if self.spec is None or len(self.spec.x) == 0:
            return {}

        x = self.spec.x
        y = self.spec.y

        if self.active_range:
            mask = (x >= self.active_range[0]) & (x <= self.active_range[1])
            if np.sum(mask) >= 4:
                x = x[mask]
                y = y[mask]

        # Estimate background from the late-time tail of the active range.
        n_tail = max(3, int(np.ceil(len(x) * 0.15)))
        bg_est = float(np.median(y[-n_tail:]))
        bg_est = max(0.0, bg_est)

        # Estimate prompt centroid (maximum counts above baseline)
        y_sub = np.maximum(y - bg_est, 0.0)
        max_idx = int(np.argmax(y_sub))
        centroid_est = float(x[max_idx])
        peak_height = float(y_sub[max_idx])

        # Estimate FWHM from half-maximum width on the left (prompt) side
        half_max = peak_height * 0.5
        left_half_idx = max_idx
        for i in range(max_idx, -1, -1):
            if y_sub[i] <= half_max:
                left_half_idx = i
                break
        hwhm_left = max(abs(centroid_est - x[left_half_idx]), 1.0)
        fwhm_est = float(2.0 * hwhm_left)

        # Estimate half-life from decay slope on the right
        right_decay_pts = []
        for i in range(max_idx + 1, len(x)):
            if y_sub[i] > 0.05 * peak_height:
                right_decay_pts.append((x[i] - centroid_est, y_sub[i]))
            else:
                break
        
        t12_est = 20.0
        if len(right_decay_pts) >= 4:
            try:
                rx, ry = zip(*right_decay_pts)
                rx = np.array(rx)
                lry = np.log(np.maximum(ry, 1.0))
                # Linear fit of log(counts) vs x
                slope, _ = np.polyfit(rx, lry, 1)
                if slope < 0:
                    t12_est = float(-np.log(2.0) / slope)
            except Exception:
                t12_est = 20.0
        
        # Scale (approximate integrated area above background)
        dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
        scale_est = float(np.sum(y_sub) * dx)
        if scale_est <= 0:
            scale_est = float(peak_height * fwhm_est * 1.5)

        self.pars = [t12_est, fwhm_est, centroid_est, scale_est, bg_est]
        return {
            "t12": t12_est,
            "fwhm": fwhm_est,
            "centroid": centroid_est,
            "scale": scale_est,
            "bg": bg_est
        }

    def fit(
        self,
        t12: Optional[float] = None,
        fwhm: Optional[float] = None,
        centroid: Optional[float] = None,
        scale: Optional[float] = None,
        bg: Optional[float] = None,
        freepars: Optional[List[bool]] = None,
        fit_range: Optional[Tuple[float, float]] = None,
        max_nfev: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute non-linear least squares fit using SciPy.
        """
        if self.spec is None or len(self.spec.x) == 0:
            raise ValueError("No spectrum loaded to fit.")

        # Update initial parameters if provided. Interactive fitting is
        # restricted to right-side (nonnegative) lifetimes.
        if t12 is not None: self.pars[0] = max(0.0, float(t12))
        if fwhm is not None: self.pars[1] = float(fwhm)
        if centroid is not None: self.pars[2] = float(centroid)
        if scale is not None: self.pars[3] = float(scale)
        if bg is not None: self.pars[4] = float(bg)
        if freepars is not None:
            self.freepars = [bool(f) for f in freepars]
            while len(self.freepars) < 5:
                self.freepars.append(False)

        if fit_range is not None:
            self.active_range = fit_range

        x_full = self.spec.x
        y_full = self.spec.y
        dy_full = self.spec.dy

        if self.active_range:
            r0, r1 = sorted(self.active_range)
            mask = (x_full >= r0) & (x_full <= r1)
        else:
            mask = np.ones_like(x_full, dtype=bool)

        x_fit = x_full[mask]
        y_fit = y_full[mask]
        dy_fit = dy_full[mask]

        ndp = len(x_fit)
        if ndp < 4:
            raise ValueError(f"Insufficient data points in fit range: {ndp} points.")

        # Determine free parameter indices
        free_indices = [i for i, is_free in enumerate(self.freepars) if is_free]
        nip = len(free_indices)
        ndf = ndp - nip

        if ndf < 1:
            raise ValueError(f"Degrees of freedom ({ndf}) < 1. Free parameters ({nip}) exceed data points ({ndp}).")
        if nip < 1:
            raise ValueError("All parameters are fixed. Free at least one parameter to perform fit.")

        p_current = np.array(self.pars, dtype=np.float64)

        # Residuals function for scipy.optimize.least_squares
        def residuals(p_free_vals: np.ndarray) -> np.ndarray:
            p_full = p_current.copy()
            for idx, val in zip(free_indices, p_free_vals):
                p_full[idx] = val
            model_y = eval_halflife(
                x_fit,
                t12=p_full[0],
                fwhm=p_full[1],
                centroid=p_full[2],
                scale=p_full[3],
                bg=p_full[4]
            )
            return (y_fit - model_y) / dy_fit

        # Initial vector for free parameters
        p0_free = p_current[free_indices]

        # Define reasonable physical bounds for free parameters
        x_min, x_max = np.min(x_fit), np.max(x_fit)
        span = max(1e-6, x_max - x_min)
        dx = float(np.median(np.diff(x_fit))) if len(x_fit) > 1 else span
        edge_count = max(2, len(y_fit) // 10)
        edge_values = np.concatenate((y_fit[:edge_count], y_fit[-edge_count:]))
        peak = max(0.0, float(np.max(y_fit)))
        noise = max(1.0, 1.4826 * float(np.median(np.abs(edge_values - np.median(edge_values)))))
        initial_scale = max(1e-6, abs(p_current[3]))
        lower_bounds = []
        upper_bounds = []

        for idx in free_indices:
            if idx == 0:  # t12: nonnegative, identifiable within the fitted span
                lower_bounds.append(0.0)
                upper_bounds.append(max(10.0 * span, 1e-6))
            elif idx == 1:  # FWHM: prompt width should remain narrow
                lower_bounds.append(0.0)
                upper_bounds.append(max(0.5 * span, 1e-6))
            elif idx == 2:  # Centroid: fit range plus a small margin
                lower_bounds.append(x_min - 0.1 * span)
                upper_bounds.append(x_max + 0.1 * span)
            elif idx == 3:  # Scale: practical range around the current estimate
                lower_bounds.append(0.0)
                upper_bounds.append(max(100.0 * initial_scale, 1000.0 * peak * max(dx, 1e-6), 1e-6))
            elif idx == 4:  # Background: data-driven practical range
                lower_bounds.append(0.0)
                upper_bounds.append(max(5.0 * abs(p_current[4]), 5.0 * noise, 0.05 * peak, 5.0))

        # Keep automatic/user initial guesses inside the practical bounds.
        p0_free = np.asarray(p0_free, dtype=np.float64)
        p0_free = np.minimum(np.maximum(p0_free, np.asarray(lower_bounds)), np.asarray(upper_bounds))

        res = scipy.optimize.least_squares(
            residuals,
            p0_free,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8
        )

        # Extract fitted parameters
        p_opt = p_current.copy()
        for idx, val in zip(free_indices, res.x):
            p_opt[idx] = val
        self.pars = list(p_opt)

        # Estimate parameter uncertainties from Jacobian covariance:
        # cov = inv(J.T @ J)
        errs = np.zeros(5, dtype=np.float64)
        if res.jac is not None and res.jac.shape[1] > 0:
            try:
                # J is already weighted by 1/dy_fit
                J = res.jac
                jt_j = J.T @ J
                cov = np.linalg.pinv(jt_j)
                diag_cov = np.diag(cov)
                for i, col_idx in enumerate(free_indices):
                    if diag_cov[i] > 0:
                        errs[col_idx] = np.sqrt(diag_cov[i])
            except Exception:
                pass
        self.errs = list(errs)

        # Evaluate model across fitted range and full range
        y_model_fit = eval_halflife(x_fit, *self.pars)
        chisq_total = float(np.sum(((y_fit - y_model_fit) / dy_fit)**2))
        chisq_ndf = float(chisq_total / ndf) if ndf > 0 else 0.0

        y_model_full = eval_halflife(x_full, *self.pars)
        residuals_full = (y_full - y_model_full) / dy_full

        result_dict = {
            "success": bool(res.success),
            "status_message": res.message,
            "nfev": int(res.nfev),
            "t12": float(self.pars[0]),
            "t12_err": float(self.errs[0]),
            "fwhm": float(self.pars[1]),
            "fwhm_err": float(self.errs[1]),
            "centroid": float(self.pars[2]),
            "centroid_err": float(self.errs[2]),
            "scale": float(self.pars[3]),
            "scale_err": float(self.errs[3]),
            "bg": float(self.pars[4]),
            "bg_err": float(self.errs[4]),
            "bg_fixed": bool(not self.freepars[4]),
            "freepars": list(self.freepars),
            "chisq_total": chisq_total,
            "chisq_ndf": chisq_ndf,
            "ndp": int(ndp),
            "nip": int(nip),
            "ndf": int(ndf),
            "fit_range": list(self.active_range) if self.active_range else [float(np.min(x_full)), float(np.max(x_full))],
            "x_eval": x_full.tolist(),
            "y_data": y_full.tolist(),
            "dy_data": dy_full.tolist(),
            "y_fit": y_model_full.tolist(),
            "residuals": residuals_full.tolist()
        }
        self.last_fit_result = result_dict
        return result_dict

    def scan_background(
        self,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        steps: int = 21,
        apply_best: bool = True
    ) -> Dict[str, Any]:
        """
        Explore background value as function of chi^2 (matching Mode 6 of halflife.c).
        
        Scans background offset across a range, fitting free parameters at each step,
        to map out chi^2 vs background profile and identify optimal baseline.
        """
        if self.spec is None or len(self.spec.x) == 0:
            raise ValueError("No spectrum loaded for background scan.")

        current_bg = self.pars[4]
        if b_min is None or b_max is None:
            # Default window: current_bg +/- max(1.0, 0.2 * current_bg)
            delta = max(2.0, 0.25 * current_bg) if current_bg > 0 else 5.0
            b_min = max(0.0, current_bg - delta)
            b_max = current_bg + delta

        b_values = np.linspace(b_min, b_max, steps)
        scan_records = []
        best_chisq = float("inf")
        best_bg = current_bg
        best_pars = list(self.pars)
        best_errs = list(self.errs)

        # Preserve original free parameters, but ensure bg is fixed during scan
        orig_free = list(self.freepars)
        scan_free = list(self.freepars)
        scan_free[4] = False  # Fixed bg during scan steps

        for b_val in b_values:
            try:
                res = self.fit(bg=b_val, freepars=scan_free)
                if res["success"]:
                    record = {
                        "bg": float(b_val),
                        "t12": float(res["t12"]),
                        "t12_err": float(res["t12_err"]),
                        "fwhm": float(res["fwhm"]),
                        "centroid": float(res["centroid"]),
                        "chisq_ndf": float(res["chisq_ndf"])
                    }
                    scan_records.append(record)
                    if res["chisq_ndf"] < best_chisq:
                        best_chisq = res["chisq_ndf"]
                        best_bg = float(b_val)
                        best_pars = list(self.pars)
                        best_errs = list(self.errs)
            except Exception:
                continue

        # Restore original free setting
        self.freepars = orig_free

        if apply_best and scan_records:
            self.pars = list(best_pars)
            self.errs = list(best_errs)
            # Re-evaluate final fit with best background
            self.fit(bg=best_bg, freepars=scan_free)

        scan_result = {
            "b_min": float(b_min),
            "b_max": float(b_max),
            "steps": int(len(scan_records)),
            "best_bg": float(best_bg),
            "best_chisq_ndf": float(best_chisq),
            "records": scan_records
        }
        self.last_scan_result = scan_result
        return scan_result

    def export_plot(
        self,
        out_path: Union[str, Path],
        title: str = "",
        log_scale: bool = False
    ) -> None:
        """Export publication-quality vector PDF or PNG plot of spectrum and fit."""
        if not HAS_MATPLOTLIB:
            raise RuntimeError("matplotlib is required for plotting export.")

        if self.spec is None or self.last_fit_result is None:
            raise ValueError("No fit results available to plot.")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = self.last_fit_result
        x = np.array(res["x_eval"])
        y = np.array(res["y_data"])
        dy = np.array(res["dy_data"])
        y_fit = np.array(res["y_fit"])
        residuals = np.array(res["residuals"])

        r0, r1 = res["fit_range"]
        mask_plot = (x >= r0 - 0.05 * (r1 - r0)) & (x <= r1 + 0.05 * (r1 - r0))
        if not np.any(mask_plot):
            mask_plot = np.ones_like(x, dtype=bool)

        fig, (ax_main, ax_res) = plt.subplots(
            2, 1,
            figsize=(9, 6),
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
            sharex=True
        )

        # Main Spectrum Plot
        ax_main.errorbar(
            x[mask_plot], y[mask_plot], yerr=dy[mask_plot],
            fmt="o", color="#4b5563", ecolor="#9ca3af", elinewidth=0.8,
            capsize=0, markersize=3, alpha=0.85, label="Data"
        )
        
        # Dense fit curve for smooth vector line
        x_dense = np.linspace(r0, r1, 1000)
        y_dense = eval_halflife(x_dense, res["t12"], res["fwhm"], res["centroid"], res["scale"], res["bg"])
        ax_main.plot(x_dense, y_dense, color="#dc2626", linewidth=1.8, label="Convoluted Fit")

        # Background Line
        ax_main.axhline(res["bg"], color="#2563eb", linestyle="--", linewidth=1.2, label=f"Background ({res['bg']:.1f})")

        # Fit parameters info box
        info_text = (
            f"$t_{{1/2}} = {res['t12']:.3f} \\pm {res['t12_err']:.3f}$\n"
            f"FWHM $= {res['fwhm']:.3f} \\pm {res['fwhm_err']:.3f}$\n"
            f"Centroid $= {res['centroid']:.2f} \\pm {res['centroid_err']:.2f}$\n"
            f"Scale $= {res['scale']:.1f} \\pm {res['scale_err']:.1f}$\n"
            f"$\\chi^2/\\mathrm{{NDF}} = {res['chisq_ndf']:.3f}$"
        )
        ax_main.text(
            0.97, 0.95, info_text,
            transform=ax_main.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.92)
        )

        if log_scale:
            ax_main.set_yscale("log")

        ax_main.set_ylabel("Counts", fontsize=10, fontweight="bold")
        plot_title = title or f"Half-Life Fit: {self.spec.filename} ($t_{{1/2}} = {res['t12']:.3f}$)"
        ax_main.set_title(plot_title, fontsize=11, fontweight="bold", pad=10)
        ax_main.grid(True, linestyle=":", alpha=0.5)
        ax_main.legend(loc="upper left", frameon=True, fontsize=8)

        # Residuals Plot
        ax_res.axhline(0, color="#64748b", linestyle="-", linewidth=0.8)
        ax_res.axhline(2, color="#ef4444", linestyle=":", linewidth=0.6)
        ax_res.axhline(-2, color="#ef4444", linestyle=":", linewidth=0.6)
        ax_res.errorbar(
            x[mask_plot], residuals[mask_plot],
            fmt="o", color="#3b82f6", markersize=2.5, alpha=0.8
        )
        ax_res.set_ylabel("Residuals ($\\sigma$)", fontsize=8, fontweight="bold")
        ax_res.set_xlabel(self.spec.x_label, fontsize=10, fontweight="bold")
        ax_res.set_ylim(-4.5, 4.5)
        ax_res.grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ==============================================================================
# 4. INTERACTIVE CLI MENU (HALFLIFE.C ENHANCED REPL)
# ==============================================================================

def run_interactive_cli(initial_file: Optional[str] = None):
    """
    Interactive terminal menu matching halflife.c modes 1-9 with enhanced features.
    """
    fitter = HalfLifeFitter()
    inname = initial_file or ""
    outname = ""

    print("\n" + "═" * 70)
    print("  ⏱️  Welcome to Python Half-Life & Lifetime Fitter (halflife.py)")
    print("  Fits Gaussian convoluted with exponential decay on .dat spectra")
    print("═" * 70)

    if inname:
        try:
            fitter.load_data(inname)
            outname = str(Path(inname).with_suffix(".fit"))
            print(f"[+] Loaded {len(fitter.spec)} channels from {inname}")
            print(f"    Default output fit filename: {outname}")
        except Exception as e:
            print(f"[!] Error loading {inname}: {e}")
            inname = ""

    while True:
        print("\n" + "─" * 50)
        print(" 1) Enter filename and read data")
        print(" 2) Enter initial parameters (or auto-guess)")
        print(" 3) Fix or free parameters")
        print(" 4) Perform fit and write output to file")
        print(" 5) Compress data by factor of 2")
        print(" 6) Explore background value as function of chi^2")
        print(" 7) Print current values of parameters to screen")
        print(" 8) Write output with current values of parameters")
        print(" 9) Export spectrum as 3-column ASCII (x y dy)")
        print(" 10) Export publication vector PDF plot")
        print(" 0) Quit")
        print("─" * 50)

        try:
            choice_str = input("Select mode [0-10]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not choice_str:
            continue
        try:
            mode = int(choice_str)
        except ValueError:
            print("[!] Invalid input. Please enter a number from 0 to 10.")
            continue

        if mode == 0:
            print("Goodbye.")
            break

        elif mode == 1:
            prompt = f"Enter ASCII data filename [{inname}]: " if inname else "Enter ASCII data filename: "
            fn = input(prompt).strip()
            if not fn and inname:
                fn = inname
            if fn:
                try:
                    fitter.load_data(fn)
                    inname = fn
                    outname = str(Path(inname).with_suffix(".fit"))
                    print(f"[+] Loaded {len(fitter.spec)} channels from: {inname}")
                    print(f"[+] Output fit filename: {outname}")
                except Exception as e:
                    print(f"[!] Failed to load file: {e}")

        elif mode == 2:
            if fitter.spec is None:
                print("[!] Please load a spectrum file first (Mode 1).")
                continue

            print("\n--- Initial Parameter Estimation ---")
            print("Press Enter to accept current values / auto-guesses.")

            def get_float_input(param_name, current_val):
                val_str = input(f"Enter initial guess for {param_name} [{current_val:.3f}]: ").strip()
                if not val_str:
                    return current_val
                try:
                    return float(val_str)
                except ValueError:
                    print(f"Invalid number, keeping {current_val:.3f}")
                    return current_val

            t12 = get_float_input("halflife t_1/2 (-ve if left decay)", fitter.pars[0])
            fwhm = get_float_input("FWHM of prompt peak", fitter.pars[1])
            cent = get_float_input("prompt peak centroid", fitter.pars[2])
            scale = get_float_input("scaling factor / area", fitter.pars[3])
            bg = get_float_input("background level", fitter.pars[4])

            fitter.pars = [t12, fwhm, cent, scale, bg]
            print("\n[+] Updated Initial Parameters:")
            print(f"    t_1/2          = {fitter.pars[0]:.3f}")
            print(f"    FWHM           = {fitter.pars[1]:.3f}")
            print(f"    Centroid       = {fitter.pars[2]:.2f}")
            print(f"    Scaling factor = {fitter.pars[3]:.2f}")
            print(f"    Background     = {fitter.pars[4]:.2f}")

        elif mode == 3:
            names = ["t_1/2", "FWHM", "Centroid", "Scaling Factor", "Background"]
            while True:
                print("\nParameter Fix/Free Status:")
                for i, name in enumerate(names):
                    status = "FREE " if fitter.freepars[i] else "*FIX*"
                    print(f"  [{status}]  {i+1}) {name} = {fitter.pars[i]:.3f}")
                print("         0) Finish")
                
                sel = input("Enter number to toggle fix/free [0-5]: ").strip()
                if sel == "0" or not sel:
                    break
                try:
                    p_idx = int(sel) - 1
                    if 0 <= p_idx < 5:
                        fitter.freepars[p_idx] = not fitter.freepars[p_idx]
                        if not fitter.freepars[p_idx]:
                            v_str = input(f"Enter fixed value for {names[p_idx]} [{fitter.pars[p_idx]:.3f}]: ").strip()
                            if v_str:
                                fitter.pars[p_idx] = float(v_str)
                except Exception as e:
                    print(f"[!] Error: {e}")

        elif mode == 4:
            if fitter.spec is None:
                print("[!] Please load a spectrum file first (Mode 1).")
                continue

            print("\n[*] Performing non-linear convolution fit...")
            try:
                res = fitter.fit()
                print("\n" + "═" * 45)
                print(" ++++++++++++ Results of fit ++++++++++++")
                print("═" * 45)
                t12_err_s = f"+/- {res['t12_err']:.3f}" if fitter.freepars[0] else "FIXED"
                fwhm_err_s = f"+/- {res['fwhm_err']:.3f}" if fitter.freepars[1] else "FIXED"
                cent_err_s = f"+/- {res['centroid_err']:.2f}" if fitter.freepars[2] else "FIXED"
                scale_err_s = f"+/- {res['scale_err']:.2f}" if fitter.freepars[3] else "FIXED"
                bg_err_s = f"+/- {res['bg_err']:.2f}" if fitter.freepars[4] else "(not fitted / fixed)"

                print(f" t_1/2           = {res['t12']:8.3f} {t12_err_s}")
                print(f" FWHM            = {res['fwhm']:8.3f} {fwhm_err_s}")
                print(f" Centroid        = {res['centroid']:8.2f} {cent_err_s}")
                print(f" Scaling factor  = {res['scale']:8.2f} {scale_err_s}")
                print(f" Background level= {res['bg']:8.2f} {bg_err_s}")
                print(f" Chisqr/D.O.F.   = {res['chisq_ndf']:8.3f} ({res['ndf']} D.O.F.)")
                print("═" * 45)

                if outname:
                    save_fit_file(outname, inname, fitter.spec, res)
                    print(f"[+] Output written to file: {outname}")
            except Exception as e:
                print(f"[!] Fit failed: {e}")

        elif mode == 5:
            if fitter.spec is None:
                print("[!] Please load a spectrum first.")
                continue
            try:
                fitter.compress(2)
                print(f"[+] Spectrum compressed by factor of 2 ({len(fitter.spec)} channels).")
            except Exception as e:
                print(f"[!] Compression failed: {e}")

        elif mode == 6:
            if fitter.spec is None:
                print("[!] Please load a spectrum first.")
                continue
            print("\n[*] Exploring background profile as function of Chi^2 (Mode 6)...")
            try:
                scan_res = fitter.scan_background(steps=21, apply_best=True)
                print("\n  Bkgnd      t_1/2     Chisq/D.O.F")
                print("  " + "─" * 32)
                for rec in scan_res["records"]:
                    print(f"  {rec['bg']:7.2f}   {rec['t12']:8.2f}   {rec['chisq_ndf']:8.3f}")
                print("  " + "─" * 32)
                print(f"[+] Optimal background: {scan_res['best_bg']:.2f} (Chi^2/NDF = {scan_res['best_chisq_ndf']:.3f})")
                if outname and fitter.last_fit_result:
                    save_fit_file(outname, inname, fitter.spec, fitter.last_fit_result)
                    print(f"[+] Updated output written to: {outname}")
            except Exception as e:
                print(f"[!] Background scan failed: {e}")

        elif mode == 7:
            print("\n--- Current Parameter Values ---")
            print(f" t_1/2           = {fitter.pars[0]:.3f} {'(free)' if fitter.freepars[0] else '(fixed)'}")
            print(f" FWHM            = {fitter.pars[1]:.3f} {'(free)' if fitter.freepars[1] else '(fixed)'}")
            print(f" Centroid        = {fitter.pars[2]:.2f} {'(free)' if fitter.freepars[2] else '(fixed)'}")
            print(f" Scaling factor  = {fitter.pars[3]:.2f} {'(free)' if fitter.freepars[3] else '(fixed)'}")
            print(f" Background      = {fitter.pars[4]:.2f} {'(free)' if fitter.freepars[4] else '(fixed)'}")

        elif mode == 8:
            if fitter.spec is None or not outname:
                print("[!] No active spectrum or output file.")
                continue
            res = fitter.last_fit_result or {
                "t12": fitter.pars[0], "t12_err": 0.0,
                "fwhm": fitter.pars[1], "fwhm_err": 0.0,
                "centroid": fitter.pars[2], "centroid_err": 0.0,
                "scale": fitter.pars[3], "scale_err": 0.0,
                "bg": fitter.pars[4], "bg_err": 0.0,
                "chisq_ndf": 0.0, "freepars": fitter.freepars
            }
            save_fit_file(outname, inname, fitter.spec, res)
            print(f"[+] Output written to: {outname}")

        elif mode == 9:
            if fitter.spec is None:
                print("[!] No active spectrum.")
                continue
            out_ascii = str(Path(inname).with_name(Path(inname).stem + "_3col.txt"))
            with open(out_ascii, "w", encoding="utf-8") as f:
                f.write("# x\ty\tdy\n")
                for x_v, y_v, dy_v in zip(fitter.spec.x, fitter.spec.y, fitter.spec.dy):
                    f.write(f"{x_v:8.1f}\t{y_v:8.3f}\t{dy_v:8.3f}\n")
            print(f"[+] Exported 3-column data to: {out_ascii}")

        elif mode == 10:
            if not HAS_MATPLOTLIB:
                print("[!] Matplotlib is not available for PDF export.")
                continue
            if fitter.last_fit_result is None:
                print("[!] Please perform a fit first (Mode 4).")
                continue
            out_pdf = str(Path(inname).with_suffix(".pdf"))
            fitter.export_plot(out_pdf)
            print(f"[+] Vector PDF plot exported to: {out_pdf}")


# ==============================================================================
# 5. COMMAND-LINE INTERFACE (SCRIPTABLE & BATCH EXECUTION)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nuclear Half-Life & Lifetime Spectrum Fitting Tool (halflife.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Interactive terminal REPL mode:
  python3 halflife.py -i spectrum.dat

  # 2. Automated command-line fit with custom initial guesses:
  python3 halflife.py spectrum.dat --t12 25.0 --fwhm 15.0 --centroid 1509.0 --bg 1000.0 --range 1480 1650 -o fit_result.fit

  # 3. Fit with fixed FWHM and background chi^2 profiling:
  python3 halflife.py spectrum.dat --fwhm 18.2 --fix-fwhm --scan-bg --pdf spectrum_fit.pdf --json
        """
    )
    parser.add_argument("file", nargs="?", default=None, help="Input ASCII spectrum .dat/.txt file")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive menu mode (halflife.c REPL)")
    parser.add_argument("--t12", type=float, default=None, help="Initial guess for half-life (t_1/2)")
    parser.add_argument("--fwhm", type=float, default=None, help="Initial guess for prompt peak FWHM")
    parser.add_argument("--centroid", type=float, default=None, help="Initial guess for prompt peak centroid")
    parser.add_argument("--scale", type=float, default=None, help="Initial guess for scaling factor / area")
    parser.add_argument("--bg", type=float, default=None, help="Initial guess for constant background offset")
    
    parser.add_argument("--fix-t12", action="store_true", help="Fix half-life parameter during fit")
    parser.add_argument("--fix-fwhm", action="store_true", help="Fix FWHM parameter during fit")
    parser.add_argument("--fix-centroid", action="store_true", help="Fix centroid parameter during fit")
    parser.add_argument("--fix-scale", action="store_true", help="Fix scaling factor parameter during fit")
    parser.add_argument("--free-bg", action="store_true", help="Allow background to float freely during fit (default; retained for compatibility)")
    parser.add_argument("--fix-bg", action="store_true", help="Fix background during fit")
    
    parser.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Fit range bounds [x_min x_max]")
    parser.add_argument("--energy", action="store_true", help="Fit on calibrated energy/time axis instead of channels")
    parser.add_argument("--compress", type=int, default=None, help="Compress spectrum by integer binning factor (e.g. 2)")
    parser.add_argument("--scan-bg", action="store_true", help="Explore background chi^2 profile and select optimal baseline")
    
    parser.add_argument("-o", "--out", type=str, default=None, help="Output .fit file path")
    parser.add_argument("--pdf", type=str, default=None, help="Export publication vector PDF plot")
    parser.add_argument("--png", type=str, default=None, help="Export PNG plot")
    parser.add_argument("--json", action="store_true", help="Print fit results as JSON dictionary")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress terminal banner and verbose output")

    args = parser.parse_args()

    # If no file provided or interactive flag set, launch REPL
    if args.interactive or (args.file is None and len(sys.argv) == 1):
        run_interactive_cli(args.file)
        sys.exit(0)

    if not args.file:
        parser.print_help()
        sys.exit(1)

    fitter = HalfLifeFitter()
    try:
        fitter.load_data(args.file, use_energy=args.energy)
    except Exception as e:
        print(f"[!] Error reading '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    if args.compress and args.compress > 1:
        fitter.compress(args.compress)

    freepars = [
        not args.fix_t12,
        not args.fix_fwhm,
        not args.fix_centroid,
        not args.fix_scale,
        not args.fix_bg
    ]

    fit_range = tuple(args.range) if args.range else None

    try:
        if args.scan_bg:
            if not args.quiet:
                print(f"[*] Exploring background chi^2 profile on '{args.file}'...")
            fitter.scan_background(apply_best=True)

        res = fitter.fit(
            t12=args.t12,
            fwhm=args.fwhm,
            centroid=args.centroid,
            scale=args.scale,
            bg=args.bg,
            freepars=freepars,
            fit_range=fit_range
        )

        if args.json:
            print(json.dumps(res, indent=2))
        elif not args.quiet:
            print("\n" + "═" * 50)
            print(f" ⏱️ Half-Life Fit Results for: {args.file}")
            print("═" * 50)
            t12_status = f"+/- {res['t12_err']:.3f}" if freepars[0] else "(FIXED)"
            fwhm_status = f"+/- {res['fwhm_err']:.3f}" if freepars[1] else "(FIXED)"
            cent_status = f"+/- {res['centroid_err']:.2f}" if freepars[2] else "(FIXED)"
            scale_status = f"+/- {res['scale_err']:.2f}" if freepars[3] else "(FIXED)"
            bg_status = f"+/- {res['bg_err']:.2f}" if freepars[4] else "(FIXED / not fitted)"

            print(f"  Half-Life (t_1/2):   {res['t12']:9.3f} {t12_status} {fitter.spec.x_label}")
            print(f"  Prompt FWHM:         {res['fwhm']:9.3f} {fwhm_status} {fitter.spec.x_label}")
            print(f"  Centroid:            {res['centroid']:9.2f} {cent_status}")
            print(f"  Scaling Factor:      {res['scale']:9.2f} {scale_status}")
            print(f"  Background Level:    {res['bg']:9.2f} {bg_status}")
            print(f"  Chi^2 / D.O.F.:      {res['chisq_ndf']:9.3f} ({res['ndf']} degrees of freedom)")
            print("═" * 50)

        # Output .fit file
        out_path = args.out or str(Path(args.file).with_suffix(".fit"))
        save_fit_file(out_path, args.file, fitter.spec, res)
        if not args.quiet:
            print(f"[+] Fit results written to: {out_path}")

        # Export PDF or PNG if requested
        if args.pdf:
            fitter.export_plot(args.pdf)
            if not args.quiet:
                print(f"[+] Vector PDF plot saved to: {args.pdf}")
        if args.png:
            fitter.export_plot(args.png)
            if not args.quiet:
                print(f"[+] PNG plot saved to: {args.png}")

    except Exception as e:
        print(f"[!] Fit execution failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
