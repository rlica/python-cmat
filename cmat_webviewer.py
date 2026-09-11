#!/usr/bin/env python3
"""
cmat_webviewer.py - High-Performance Web-based Interactive 2D Matrix Viewer with Classic Binned 1D Histogram.

Features:
  - Classic Binned Histogram: The 1D spectrum renders as a true stepped staircase histogram with discrete
    channel bins, subtle fill, and crisp outlines.
  - Real-Time 1D Mouse Readout: Hovering over the 1D histogram displays a dynamic hairline tracker,
    highlights the exact channel bin, and outputs real-time Channel, Energy (keV), and Counts
    to both the top HUD and on-canvas badge.
  - Automatic 1D Projection: The 1D spectrum automatically updates to show the X or Y projection
    of the 2D region currently on display after zooming or panning.
  - Projection Axis Selector: Toggle between Det 1 (X Projection summing over visible Y) and
    Det 2 (Y Projection summing over visible X) with buttons or keyboard shortcuts (P, X, Y).
  - Exact 1:1 Pixel-to-Display matching: Fast, peak-preserving 2D max-pooling when zoomed out,
    and 1:1 single-channel binning when zoomed in.
  - Ultra-Fast 32-bit Uint32 Color LUT rendering (<0.5 ms).
  - Full GASPware cmat Navigation:
      * Left-Click Drag: Direct box zoom into rectangle (no Ctrl required)
      * Left / Right Arrow: Set Left (Xmin) and Right (Xmax) limit markers at cursor
      * Down / Up Arrow: Set Down (Ymin) and Up (Ymax) limit markers at cursor
      * E / e: Expand / Zoom into set limit markers
      * F / f: Full matrix view (reset zoom)
      * P / X / Y: Toggle 1D Projection Axis (Det 1 vs Det 2)
      * 1 / 2 / 4 (or L): Switch Linear, Sqrt, Log color scale
      * C / c: Cycle color gradients (Turbo, Viridis, Plasma, Inferno, Hot, Jet, Gray)
      * H / ?: Toggle keyboard shortcuts modal

Usage:
  python3 cmat_webviewer.py GeE-symm.cmat --port 8080
"""

import os
import sys
import time
import math
import json
import socket
import argparse
import webbrowser
import threading
import glob
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Configure Matplotlib cache directory within workspace
cache_dir = Path(__file__).resolve().parent / ".matplotlib_cache"
try:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
except Exception:
    pass

from cmat import CMATReader

CONFIG_FILENAME = "python-cmat-config.txt"

DEFAULT_CONFIG = {
    "cal": "0.0, 1.0, 0.0",
    "cal_0": "0.0, 1.0, 0.0",
    "cal_1": "0.0, 1.0, 0.0",
    "fit_type": "gaussian",
    "fwhm_mult_1d": 4.0,
    "roi_half_width_2d": 16,
    "fit_verbosity": "compact",
    "colormap": "turbo",
    "scale_mode": "log",
    "vmax": 500,
    "vmin": 1,
    "scroll_sensitivity": 4,
    "peak_search_method": "cwt",
    "peak_search_snr": 9.0,
    "proj_range": "synced",
    "proj_scale": "linear",
    "host": "0.0.0.0",
    "port": 8080,
    "open_browser": True,
    "browser": "default",
}


def get_local_ip() -> str:
    """Detect the primary network/outbound IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    return "127.0.0.1"


def is_ssh_session() -> bool:
    """Check if the current script is running within an SSH session."""
    return any(k in os.environ for k in ("SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY"))


def launch_browser(url: str, browser_name: str = "default") -> None:
    """Launch the specified web browser to open the given URL."""
    try:
        b_name = (browser_name or "default").strip().lower()
        if b_name in ("default", "auto", "true", "1", ""):
            webbrowser.open(url)
        else:
            try:
                controller = webbrowser.get(browser_name)
                controller.open(url)
            except Exception:
                # Fallback to default if specific browser controller was not found
                webbrowser.open(url)
    except Exception as e:
        print(f"[!] Warning: Could not automatically launch browser: {e}", file=sys.stderr)


def parse_cal_coefficients(cal_val) -> list:
    """Parse calibration values into a 3-element list of floats [a0, a1, a2]."""
    if isinstance(cal_val, (list, tuple)):
        vals = [float(v) for v in cal_val]
    elif isinstance(cal_val, str):
        parts = cal_val.replace(",", " ").split()
        vals = [float(p) for p in parts] if parts else [0.0, 1.0, 0.0]
    else:
        vals = [0.0, 1.0, 0.0]
    while len(vals) < 3:
        vals.append(0.0)
    return vals[:3]


parse_cal_string = parse_cal_coefficients


def ch_to_energy(ch: float, cal: list) -> float:
    """Convert channel to calibrated energy: E = a0 + a1*ch + a2*ch^2."""
    if cal is None:
        return float(ch)
    a0 = cal[0] if len(cal) > 0 else 0.0
    a1 = cal[1] if len(cal) > 1 else 1.0
    a2 = cal[2] if len(cal) > 2 else 0.0
    return a0 + a1 * float(ch) + a2 * (float(ch) ** 2)


def energy_to_ch(energy: float, cal: list, default_ch: float = None) -> float:
    """Convert calibrated energy (keV) to channel: solve a0 + a1*ch + a2*ch^2 = E."""
    if cal is None:
        return float(energy)
    a0 = cal[0] if len(cal) > 0 else 0.0
    a1 = cal[1] if len(cal) > 1 else 1.0
    a2 = cal[2] if len(cal) > 2 else 0.0
    if abs(a2) < 1e-12:
        if abs(a1) < 1e-12:
            return default_ch if default_ch is not None else float(energy)
        return (float(energy) - a0) / a1
    disc = a1 ** 2 - 4.0 * a2 * (a0 - float(energy))
    if disc < 0:
        return default_ch if default_ch is not None else (float(energy) - a0) / a1
    r1 = (-a1 + math.sqrt(disc)) / (2.0 * a2)
    r2 = (-a1 - math.sqrt(disc)) / (2.0 * a2)
    if r1 >= 0 and r2 < 0:
        return r1
    if r2 >= 0 and r1 < 0:
        return r2
    if default_ch is not None:
        return r1 if abs(r1 - default_ch) < abs(r2 - default_ch) else r2
    return r1 if r1 >= 0 else r2


def de_dch(ch: float, cal: list) -> float:
    """Calculate first derivative dE/dch = |a1 + 2*a2*ch|."""
    if cal is None:
        return 1.0
    a1 = cal[1] if len(cal) > 1 else 1.0
    a2 = cal[2] if len(cal) > 2 else 0.0
    return max(1e-9, abs(a1 + 2.0 * a2 * float(ch)))


def is_calibrated_coeffs(cal: list) -> bool:
    """Check if calibration coefficients represent non-trivial energy calibration."""
    if not cal or len(cal) < 2:
        return False
    a0 = cal[0]
    a1 = cal[1]
    a2 = cal[2] if len(cal) > 2 else 0.0
    return abs(a0) > 1e-12 or abs(a1 - 1.0) > 1e-12 or abs(a2) > 1e-12



def generate_config_content(cfg: dict) -> str:
    """Generate clean, commented INI-style python-cmat-config.txt text."""
    cal_str = cfg.get("cal", "0.0, 1.0, 0.0")
    if isinstance(cal_str, (list, tuple)):
        cal_str = ", ".join(str(v) for v in cal_str)

    cal_0_str = cfg.get("cal_0", cal_str)
    if isinstance(cal_0_str, (list, tuple)):
        cal_0_str = ", ".join(str(v) for v in cal_0_str)

    cal_1_str = cfg.get("cal_1", cal_str)
    if isinstance(cal_1_str, (list, tuple)):
        cal_1_str = ", ".join(str(v) for v in cal_1_str)

    open_br = cfg.get("open_browser", True)
    open_br_str = "true" if open_br in (True, "true", "True", "1", 1) else "false"

    return f"""# ==============================================================================
# python-cmat configuration file
# Automatically generated when no config file is present in the working directory.
# You can edit these values directly or click "Save Config" in the Web Viewer.
# ==============================================================================

# Energy Calibration (Quadratic: E = a0 + a1*ch + a2*ch^2)
# Define calibration per-axis: Det 1 / X (cal_0) and Det 2 / Y (cal_1)
cal_0 = {cal_0_str}
cal_1 = {cal_1_str}
# Shorthand for symmetric matrices (applies to both axes if cal_0/cal_1 are not set):
cal = {cal_str}

# Default Peak Function Model: gaussian, gaussian_tail (RadWare), hypermet
fit_type = {cfg.get('fit_type', 'gaussian')}

# 1D Peak Fitting Region multiplier (times estimated FWHM, e.g. 1.0 to 10.0)
fwhm_mult_1d = {cfg.get('fwhm_mult_1d', 4.0)}

# 2D Coincidence ROI half-width in channels (e.g. 6 to 36)
roi_half_width_2d = {cfg.get('roi_half_width_2d', 16)}

# Fit results verbosity: compact, detailed
fit_verbosity = {cfg.get('fit_verbosity', 'compact')}

# Default 2D Colormap: turbo, viridis, plasma, inferno, hot, jet, gray
colormap = {cfg.get('colormap', 'turbo')}

# Default 2D Scale Mode: log, sqrt, linear
scale_mode = {cfg.get('scale_mode', 'log')}

# Default Max Contrast (vmax, 0-1000) and Min Threshold (vmin)
vmax = {cfg.get('vmax', 500)}
vmin = {cfg.get('vmin', 1)}

# Scroll Zoom Sensitivity percentage (1 to 15)
scroll_sensitivity = {cfg.get('scroll_sensitivity', 4)}

# 1D Automatic Peak Search Method: cwt, prominence, mariscotti
peak_search_method = {cfg.get('peak_search_method', 'cwt')}

# 1D Peak Search Sensitivity / Min SNR (e.g. 1.0 to 15.0)
peak_search_snr = {cfg.get('peak_search_snr', 9.0)}

# 1D Projection Display Range: synced, full
proj_range = {cfg.get('proj_range', 'synced')}

# 1D Projection Y-Scale: linear, log
proj_scale = {cfg.get('proj_scale', 'linear')}

# Web Server Host / Bind Interface (0.0.0.0 binds to all network interfaces)
host = {cfg.get('host', '0.0.0.0')}

# Web Server Port
port = {cfg.get('port', 8080)}

# Automatically open web browser on launch: true, false
open_browser = {open_br_str}

# Preferred Browser to launch (default, firefox, google-chrome, chromium, safari, none)
browser = {cfg.get('browser', 'default')}
"""


def load_or_create_config(config_path: Path) -> dict:
    """Load configuration from file, or create default file if it doesn't exist."""
    cfg = DEFAULT_CONFIG.copy()
    if not config_path.exists():
        try:
            config_path.write_text(generate_config_content(cfg), encoding="utf-8")
            print(f"[*] No config file found. Created default config: {config_path.name}")
        except Exception as e:
            print(f"[!] Warning: Could not create default config {config_path.name}: {e}", file=sys.stderr)
        return cfg

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip().lower()
                    val = val.strip()
                    if key in ("cal_x", "cal_0"):
                        cfg["cal_0"] = val
                    elif key in ("cal_y", "cal_1"):
                        cfg["cal_1"] = val
                    elif key in cfg:
                        if key in ("fwhm_mult_1d", "peak_search_snr"):
                            try:
                                cfg[key] = float(val)
                            except ValueError:
                                pass
                        elif key in ("vmax", "vmin"):
                            try:
                                cfg[key] = float(val) if "." in val else int(val)
                            except ValueError:
                                pass
                        elif key in ("roi_half_width_2d", "scroll_sensitivity", "port"):
                            try:
                                cfg[key] = int(val)
                            except ValueError:
                                pass
                        elif key == "open_browser":
                            cfg[key] = (val.lower() in ("true", "1", "yes", "on"))
                        elif key in ("host", "browser"):
                            cfg[key] = val
                        else:
                            cfg[key] = val
        print(f"[*] Loaded configuration from {config_path.name}")
    except Exception as e:
        print(f"[!] Warning: Could not read config {config_path.name}: {e}. Using defaults.", file=sys.stderr)
    return cfg


def save_config_file(config_path: Path, current_settings: dict) -> None:
    """Save current configuration to the config file."""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(current_settings)
    config_path.write_text(generate_config_content(cfg), encoding="utf-8")


def _vec_erfc(arr):
    f = np.vectorize(math.erfc, otypes=[np.float64])
    return f(arr)


def fit_gaussian_peak(x, y, x_center, fit_type="gaussian", fwhm_mult=4.0, cal=[0.0, 1.0, 0.0], roi_half_width=None):
    """
    Fits a single peak with one of three scientific models:
      1. 'gaussian': Standard Symmetric Gaussian + linear background.
      2. 'gaussian_tail': RadWare / SAMPO Gaussian with Left Exponential Tail (Helmer & Lee / Radford).
      3. 'hypermet': Hypermet Model (Phillips & Marlow / Campbell & Maxwell) with analytical
                     Exponentially Modified Gaussian (EMG) convolved tail + erfc Compton step.
    ROI window size is determined by fwhm_mult * FWHM_est (or explicit roi_half_width).
    Uses Poisson counting statistics weights and computes full parameter covariance matrix.
    Assumes histogram channels represent bins [k, k+1) with continuous center coordinate k + 0.5
    following the standard GASPware / xtrackn convention ('al centro del canale').
    """
    y = np.asarray(y, dtype=np.float64)
    if x is None:
        x = np.arange(len(y), dtype=np.float64) + 0.5
    else:
        x = np.asarray(x, dtype=np.float64)
        if len(x) > 0 and np.isclose(x[0] % 1.0, 0.0):
            x = x + 0.5

    i_center = int(np.clip(int(np.floor(x_center)), 0, len(y) - 1))
    center_coord = i_center + 0.5

    if roi_half_width is not None:
        half_w = max(3, int(round(roi_half_width)))
        i_min = max(0, i_center - half_w)
        i_max = min(len(y) - 1, i_center + half_w)
    else:
        search_r = max(8, int(round(fwhm_mult * 4)))
        i_smin = max(0, i_center - search_r)
        i_smax = min(len(y) - 1, i_center + search_r)
        sub_y = y[i_smin:i_smax+1]

        if len(sub_y) >= 3:
            kernel = np.array([0.25, 0.5, 0.25])
            s_y = np.convolve(sub_y, kernel, mode='same')
            apex_local = int(np.argmax(s_y))
            apex_ch = i_smin + apex_local
            apex_val = s_y[apex_local]
            bg_est = 0.5 * (s_y[0] + s_y[-1])
            half_max = bg_est + 0.5 * max(1.0, apex_val - bg_est)

            left_k = apex_local
            while left_k > 0 and s_y[left_k] > half_max:
                left_k -= 1
            right_k = apex_local
            while right_k < len(s_y) - 1 and s_y[right_k] > half_max:
                right_k += 1
            fwhm_est = max(1.5, float(right_k - left_k))
        else:
            apex_ch = i_center
            fwhm_est = 2.5

        half_w = max(3, int(round(0.5 * fwhm_mult * fwhm_est)))
        i_min = max(0, apex_ch - half_w)
        i_max = min(len(y) - 1, apex_ch + half_w)

    if i_max - i_min < 5:
        raise ValueError("ROI window too small for fitting (minimum 5 channels required).")

    roi_x = x[i_min:i_max+1]
    roi_y = y[i_min:i_max+1]
    N = len(roi_x)

    # 1. Initial parameter estimates
    kernel = np.array([0.25, 0.5, 0.25])
    y_smooth = np.convolve(roi_y, kernel, mode='same')
    apex_idx = int(np.argmax(y_smooth))
    mu_init = float(roi_x[apex_idx])

    # Estimate background from endpoints (average 3 channels left and right)
    n_end = max(1, min(3, N // 4))
    bg_left = float(np.mean(roi_y[:n_end]))
    bg_right = float(np.mean(roi_y[-n_end:]))
    b1_init = (bg_right - bg_left) / max(1e-6, roi_x[-1] - roi_x[0])
    b0_init = 0.5 * (bg_left + bg_right)

    # Net height
    bg_at_apex = b0_init + b1_init * (mu_init - center_coord)
    H_init = max(1.0, float(roi_y[apex_idx]) - bg_at_apex)
    sigma_init = max(0.5, (fwhm_est / 2.355) if 'fwhm_est' in locals() else 1.5)

    sigma_y = np.sqrt(np.maximum(1.0, roi_y))
    weights = 1.0 / sigma_y

    is_hypermet = (fit_type == "hypermet")
    is_tail = (fit_type == "gaussian_tail")

    if is_hypermet:
        f_T_init = 0.15 * (H_init * sigma_init * math.sqrt(2.0 * math.pi))
        beta_init = 1.5 * sigma_init
        A_S_init = 0.02 * H_init
        theta = np.array([b0_init, b1_init, H_init, mu_init, sigma_init, f_T_init, beta_init, A_S_init], dtype=np.float64)
        n_params = 8
    elif is_tail:
        alpha_init = 1.5
        theta = np.array([b0_init, b1_init, H_init, mu_init, sigma_init, alpha_init], dtype=np.float64)
        n_params = 6
    else:
        theta = np.array([b0_init, b1_init, H_init, mu_init, sigma_init], dtype=np.float64)
        n_params = 5

    lam = 0.001
    max_iters = 80

    def calc_residuals_and_jacobian(p):
        if is_hypermet:
            b0, b1, H, mu, sig, f_T, beta, A_S = p
            sig = max(0.1, abs(sig))
            beta = max(0.1, abs(beta))
            H = max(0.0, H)
            f_T = max(0.0, f_T)
            A_S = max(0.0, A_S)

            dx = roi_x - mu
            dx_c = roi_x - center_coord
            z = dx / sig

            g = H * np.exp(np.clip(-0.5 * z**2, -50.0, 0.0))
            u = np.clip(dx / beta + 0.5 * (sig / beta)**2, -50.0, 50.0)
            v = np.clip(dx / (math.sqrt(2.0) * sig) + sig / (math.sqrt(2.0) * beta), -20.0, 20.0)
            t = (f_T / (2.0 * beta)) * np.exp(u) * _vec_erfc(v)
            s = 0.5 * A_S * _vec_erfc(np.clip(z / math.sqrt(2.0), -20.0, 20.0))

            model = b0 + b1 * dx_c + g + t + s
            r = (model - roi_y) * weights

            J = np.zeros((N, n_params), dtype=np.float64)
            eps = 1e-6
            for i in range(n_params):
                p_step = p.copy()
                p_step[i] += eps
                b0_s, b1_s, H_s, mu_s, sig_s, fT_s, beta_s, AS_s = p_step
                sig_s, beta_s = max(0.1, abs(sig_s)), max(0.1, abs(beta_s))
                dx_s, dx_cs = roi_x - mu_s, roi_x - center_coord
                z_s = dx_s / sig_s
                g_s = max(0.0, H_s) * np.exp(np.clip(-0.5 * z_s**2, -50.0, 0.0))
                u_s = np.clip(dx_s / beta_s + 0.5 * (sig_s / beta_s)**2, -50.0, 50.0)
                v_s = np.clip(dx_s / (math.sqrt(2.0) * sig_s) + sig_s / (math.sqrt(2.0) * beta_s), -20.0, 20.0)
                t_s = (max(0.0, fT_s) / (2.0 * beta_s)) * np.exp(u_s) * _vec_erfc(v_s)
                s_s = 0.5 * max(0.0, AS_s) * _vec_erfc(np.clip(z_s / math.sqrt(2.0), -20.0, 20.0))
                mod_s = b0_s + b1_s * dx_cs + g_s + t_s + s_s
                J[:, i] = (mod_s - model) / eps * weights
            return r, J, model

        if is_tail:
            b0, b1, H, mu, sig, alpha = p
            alpha = max(0.3, min(5.0, alpha))
        else:
            b0, b1, H, mu, sig = p
            alpha = 100.0

        sig = max(0.1, abs(sig))
        H = max(0.0, H)

        dx = roi_x - mu
        dx_c = roi_x - center_coord
        z = dx / sig

        is_gauss = (z >= -alpha)
        g_gauss = H * np.exp(-0.5 * z**2)
        exponent = np.clip(0.5 * alpha**2 + alpha * z, -50.0, 50.0)
        g_tail = H * np.exp(exponent)
        g = np.where(is_gauss, g_gauss, g_tail)

        model = b0 + b1 * dx_c + g
        r = (model - roi_y) * weights

        J = np.zeros((N, n_params), dtype=np.float64)
        J[:, 0] = weights
        J[:, 1] = weights * dx_c
        J[:, 2] = weights * (g / max(1e-12, H))

        d_mu = np.where(is_gauss, g * z / sig, -g * alpha / sig)
        J[:, 3] = weights * d_mu

        d_sig = np.where(is_gauss, g * (z**2) / sig, -g * (alpha * z) / sig)
        J[:, 4] = weights * d_sig

        if is_tail:
            d_alpha = np.where(is_gauss, 0.0, g * (alpha + z))
            J[:, 5] = weights * d_alpha

        return r, J, model

    r, J, model = calc_residuals_and_jacobian(theta)
    chi2 = np.sum(r**2)

    for _ in range(max_iters):
        JT = J.T
        JTJ = JT @ J
        diag_JTJ = np.diag(np.diag(JTJ))
        Hessian = JTJ + lam * np.maximum(diag_JTJ, 1e-4 * np.eye(n_params))
        gradient = JT @ r

        try:
            d_theta = np.linalg.solve(Hessian, -gradient)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        theta_new = theta + d_theta
        theta_new[2] = max(0.0, theta_new[2])  # H >= 0
        theta_new[4] = max(0.2, min(half_w, abs(theta_new[4])))  # sigma
        theta_new[3] = max(roi_x[0], min(roi_x[-1], theta_new[3]))  # mu within ROI
        if is_hypermet:
            theta_new[5] = max(0.0, theta_new[5])  # f_T >= 0
            theta_new[6] = max(0.2, min(half_w, theta_new[6]))  # beta
            theta_new[7] = max(0.0, theta_new[7])  # A_S >= 0
        elif is_tail:
            theta_new[5] = max(0.3, min(5.0, theta_new[5]))

        r_new, J_new, model_new = calc_residuals_and_jacobian(theta_new)
        chi2_new = np.sum(r_new**2)

        if chi2_new < chi2:
            lam = max(1e-7, lam * 0.3)
            theta = theta_new
            r = r_new
            J = J_new
            model = model_new
            if (chi2 - chi2_new) / (chi2 + 1e-12) < 1e-7:
                chi2 = chi2_new
                break
            chi2 = chi2_new
        else:
            lam = min(1e7, lam * 5.0)

    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J)

    param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))

    if is_hypermet:
        b0, b1, H, mu, sig, f_T, beta, A_S = theta
        sig = abs(sig)
        area = float(math.sqrt(2.0 * math.pi) * H * sig + f_T)
        grad_A = np.zeros(n_params)
        grad_A[2] = math.sqrt(2.0 * math.pi) * sig
        grad_A[4] = math.sqrt(2.0 * math.pi) * H
        grad_A[5] = 1.0
        area_err = float(np.sqrt(np.maximum(0.0, grad_A @ cov @ grad_A)))
        alpha_val, alpha_err_val = None, None
        f_T_val, f_T_err_val = round(float(f_T), 1), round(float(param_errors[5]), 1)
        beta_val, beta_err_val = round(float(beta), 3), round(float(param_errors[6]), 3)
        A_S_val, A_S_err_val = round(float(A_S), 2), round(float(param_errors[7]), 2)
    elif is_tail:
        b0, b1, H, mu, sig, alpha = theta
        sig = abs(sig)
        term1 = math.sqrt(math.pi / 2.0) * (1.0 + math.erf(alpha / math.sqrt(2.0)))
        term2 = math.exp(-0.5 * alpha**2) / alpha
        area = H * sig * (term1 + term2)
        dA_dH = sig * (term1 + term2)
        dA_dsig = H * (term1 + term2)
        dA_dalpha = H * sig * (math.exp(-0.5 * alpha**2) - (1.0 + alpha**2) * math.exp(-0.5 * alpha**2) / (alpha**2))
        grad_A = np.zeros(n_params)
        grad_A[2] = dA_dH
        grad_A[4] = dA_dsig
        grad_A[5] = dA_dalpha
        area_err = float(np.sqrt(np.maximum(0.0, grad_A @ cov @ grad_A)))
        alpha_val = round(float(alpha), 3)
        alpha_err_val = round(float(param_errors[5]), 3)
        f_T_val, f_T_err_val = None, None
        beta_val, beta_err_val = None, None
        A_S_val, A_S_err_val = None, None
    else:
        b0, b1, H, mu, sig = theta
        sig = abs(sig)
        area = float(H * sig * np.sqrt(2.0 * np.pi))
        dA_dH = sig * np.sqrt(2.0 * np.pi)
        dA_dsig = H * np.sqrt(2.0 * np.pi)
        grad_A = np.array([0.0, 0.0, dA_dH, 0.0, dA_dsig])
        area_err = float(np.sqrt(np.maximum(0.0, grad_A @ cov @ grad_A)))
        alpha_val, alpha_err_val = None, None
        f_T_val, f_T_err_val = None, None
        beta_val, beta_err_val = None, None
        A_S_val, A_S_err_val = None, None

    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm = fwhm_factor * sig
    fwhm_err = fwhm_factor * param_errors[4]

    peak_win_min = max(roi_x[0], mu - 1.5 * fwhm)
    peak_win_max = min(roi_x[-1], mu + 1.5 * fwhm)
    win_mask = (roi_x >= peak_win_min) & (roi_x <= peak_win_max)
    bg_roi_sum = np.sum(b0 + b1 * (roi_x[win_mask] - center_coord))
    gross_roi_sum = np.sum(roi_y[win_mask])

    ndf = max(1, N - n_params)
    red_chi2 = chi2 / ndf

    a0 = cal[0] if len(cal) > 0 else 0.0
    a1 = cal[1] if len(cal) > 1 else 1.0
    a2 = cal[2] if len(cal) > 2 else 0.0
    def ch_to_e(c): return a0 + a1 * c + a2 * (c**2)
    def de_dch(c): return abs(a1 + 2.0 * a2 * c)

    energy_mu = ch_to_e(mu)
    energy_mu_err = param_errors[3] * de_dch(mu)
    energy_fwhm = fwhm * de_dch(mu)
    energy_fwhm_err = fwhm_err * de_dch(mu)

    # Dense curve for drawing
    x_dense = np.linspace(roi_x[0], roi_x[-1], 200)
    dx_c_dense = x_dense - center_coord
    bg_dense = b0 + b1 * dx_c_dense
    z_dense = (x_dense - mu) / sig

    if is_hypermet:
        g_dense = H * np.exp(np.clip(-0.5 * z_dense**2, -50.0, 0.0))
        u_dense = np.clip((x_dense - mu) / beta + 0.5 * (sig / beta)**2, -50.0, 50.0)
        v_dense = np.clip((x_dense - mu) / (math.sqrt(2.0) * sig) + sig / (math.sqrt(2.0) * beta), -20.0, 20.0)
        t_dense = (f_T / (2.0 * beta)) * np.exp(u_dense) * _vec_erfc(v_dense)
        s_dense = 0.5 * A_S * _vec_erfc(np.clip(z_dense / math.sqrt(2.0), -20.0, 20.0))
        peak_dense = g_dense + t_dense + s_dense
    elif is_tail:
        is_gauss_dense = (z_dense >= -alpha)
        g_gauss_dense = H * np.exp(-0.5 * z_dense**2)
        exponent_dense = np.clip(0.5 * alpha**2 + alpha * z_dense, -50.0, 50.0)
        g_tail_dense = H * np.exp(exponent_dense)
        peak_dense = np.where(is_gauss_dense, g_gauss_dense, g_tail_dense)
    else:
        peak_dense = H * np.exp(-0.5 * z_dense**2)

    fit_dense = bg_dense + peak_dense

    return {
        "success": True,
        "fit_type": fit_type,
        "centroid_ch": round(float(mu), 3),
        "centroid_ch_err": round(float(param_errors[3]), 3),
        "centroid_e": round(float(energy_mu), 2),
        "centroid_e_err": round(float(energy_mu_err), 2),
        "area": round(float(area), 1),
        "area_err": round(float(area_err), 1),
        "fwhm_ch": round(float(fwhm), 3),
        "fwhm_ch_err": round(float(fwhm_err), 3),
        "fwhm_e": round(float(energy_fwhm), 2),
        "fwhm_e_err": round(float(energy_fwhm_err), 2),
        "amplitude": round(float(H), 1),
        "amplitude_err": round(float(param_errors[2]), 1),
        "alpha": alpha_val,
        "alpha_err": alpha_err_val,
        "tail_area": f_T_val,
        "tail_area_err": f_T_err_val,
        "tail_slope": beta_val,
        "tail_slope_err": beta_err_val,
        "step_height": A_S_val,
        "step_height_err": A_S_err_val,
        "bg_b0": round(float(b0), 2),
        "bg_b1": round(float(b1), 4),
        "gross_counts": round(float(gross_roi_sum), 1),
        "bg_counts": round(float(bg_roi_sum), 1),
        "chi2": round(float(chi2), 2),
        "ndf": int(ndf),
        "red_chi2": round(float(red_chi2), 3),
        "roi_ch_min": int(np.floor(roi_x[0])),
        "roi_ch_max": int(np.floor(roi_x[-1])),
        "curve_x": [round(float(v), 2) for v in x_dense],
        "curve_fit": [round(float(v), 2) for v in fit_dense],
        "curve_bg": [round(float(v), 2) for v in bg_dense],
    }


def print_fit_terminal_report(res, det_name, filename, is_cal, verbosity="compact"):
    ft = res.get("fit_type", "gaussian")
    model_tag = "Hypermet" if ft == "hypermet" else ("RadWare" if ft == "gaussian_tail" else "Gaussian")
    if verbosity == "compact":
        if is_cal:
            print(f"[1D Fit] [{det_name}] ({model_tag}):\tCentroid: {res['centroid_e']:.2f}({res['centroid_e_err']:.2f}) keV\tArea: {res['area']:.1f}({res['area_err']:.1f}) counts\tFWHM: {res['fwhm_e']:.2f}({res['fwhm_e_err']:.2f}) keV", flush=True)
        else:
            print(f"[1D Fit] [{det_name}] ({model_tag}):\tCentroid: {res['centroid_ch']:.3f}({res['centroid_ch_err']:.3f}) ch\tArea: {res['area']:.1f}({res['area_err']:.1f}) counts\tFWHM: {res['fwhm_ch']:.3f}({res['fwhm_ch_err']:.3f}) ch", flush=True)
        return

    bar = "═" * 80
    subbar = "─" * 80
    if ft == "hypermet":
        model_name = "Hypermet Model (Convolved Tail + Erfc Step)"
    elif ft == "gaussian_tail":
        model_name = "Gaussian with Left Tail (RadWare / HPGe)"
    else:
        model_name = "Standard Symmetric Gaussian"

    print(f"\n{bar}")
    print(f"[GASPware 1D Peak Fit] [{det_name}] - {filename} ({model_name})")
    print(subbar)
    if is_cal:
        print(f"  Peak Centroid     : {res['centroid_e']:10.2f} ± {res['centroid_e_err']:<6.2f} keV  (ch: {res['centroid_ch']:.3f} ± {res['centroid_ch_err']:.3f})")
        print(f"  Peak FWHM         : {res['fwhm_e']:10.2f} ± {res['fwhm_e_err']:<6.2f} keV  (ch: {res['fwhm_ch']:.3f} ± {res['fwhm_ch_err']:.3f})")
    else:
        print(f"  Peak Centroid     : {res['centroid_ch']:10.3f} ± {res['centroid_ch_err']:<6.3f} ch")
        print(f"  Peak FWHM         : {res['fwhm_ch']:10.3f} ± {res['fwhm_ch_err']:<6.3f} ch")
    print(f"  Net Peak Area     : {res['area']:10.1f} ± {res['area_err']:<6.1f} counts")
    print(f"  Peak Amplitude (H): {res['amplitude']:10.1f} ± {res['amplitude_err']:<6.1f} counts")
    if res.get("tail_area") is not None:
        print(f"  Hypermet Tail(fT) : {res['tail_area']:10.1f} ± {res['tail_area_err']:<6.1f} counts  (Slope β: {res['tail_slope']:.3f} ± {res['tail_slope_err']:.3f} ch)")
        print(f"  Compton Step (As) : {res['step_height']:10.2f} ± {res['step_height_err']:<6.2f} counts")
    elif res.get("alpha") is not None:
        print(f"  Left Tail Join(α) : {res['alpha']:10.3f} ± {res['alpha_err']:<6.3f} (Join at ch {res['centroid_ch'] - res['alpha']*res['fwhm_ch']/2.355:.2f})")
    print(f"  Gross Counts (ROI): {res['gross_counts']:10.1f} counts  (Background: {res['bg_counts']:.1f} counts)")
    print(f"  Fit ROI Window    : ch {res['roi_ch_min']} to {res['roi_ch_max']} ({res['roi_ch_max'] - res['roi_ch_min'] + 1} channels)")
    print(f"  Reduced Chi2/NDF  : {res['red_chi2']:.3f} (Chi2 = {res['chi2']:.1f}, NDF = {res['ndf']})")
    print(f"{bar}\n", flush=True)


def _fit_2d_gaussian_single_roi(
    matrix, x_center, y_center, fit_type="gaussian", cal=[0.0, 1.0, 0.0], roi_half_width=16,
    proj_x=None, proj_y=None, total_counts=None, cal_x=None, cal_y=None, **kwargs
):
    mat = np.asarray(matrix, dtype=np.float64)
    H_mat, W_mat = mat.shape

    ix = int(np.clip(int(np.floor(x_center)), 0, W_mat - 1))
    iy = int(np.clip(int(np.floor(y_center)), 0, H_mat - 1))
    center_x_coord = ix + 0.5
    center_y_coord = iy + 0.5

    x_min = max(0, ix - roi_half_width)
    x_max = min(W_mat - 1, ix + roi_half_width)
    y_min = max(0, iy - roi_half_width)
    y_max = min(H_mat - 1, iy + roi_half_width)

    if (x_max - x_min < 4) or (y_max - y_min < 4):
        raise ValueError("2D ROI window too small for fitting (minimum 5x5 channels required).")

    roi_raw = mat[y_min:y_max+1, x_min:x_max+1]
    Ny, Nx = roi_raw.shape
    N_pixels = Ny * Nx
    gross_counts = float(np.sum(roi_raw))

    xs = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    ys = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    X_grid, Y_grid = np.meshgrid(xs, ys)

    x_flat = X_grid.ravel()
    y_flat = Y_grid.ravel()
    z_raw_flat = roi_raw.ravel()

    # Estimate bg|bg continuum from 4 outer corners of the ROI
    c_w = max(1, min(3, min(Nx, Ny) // 4))
    corners = [
        roi_raw[:c_w, :c_w],
        roi_raw[:c_w, -c_w:],
        roi_raw[-c_w:, :c_w],
        roi_raw[-c_w:, -c_w:]
    ]
    b0_init = max(0.0, float(np.mean([np.mean(c) for c in corners])))
    bx_init = 0.0
    by_init = 0.0

    # Estimate p|bg and bg|p ridges from border strips (subtracting b0)
    border_y = (roi_raw[:, 0] + roi_raw[:, -1]) / 2.0
    ry_init = max(0.0, float(np.max(border_y) - b0_init))

    border_x = (roi_raw[0, :] + roi_raw[-1, :]) / 2.0
    rx_init = max(0.0, float(np.max(border_x) - b0_init))

    # Initial centroid estimates via smoothed apex
    pad = np.pad(np.maximum(0.0, roi_raw), 1, mode='edge')
    smooth = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
              pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
              pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]) / 9.0
    apex_idx = np.unravel_index(np.argmax(smooth), smooth.shape)
    mu_x_init = float(xs[apex_idx[1]])
    mu_y_init = float(ys[apex_idx[0]])

    sig_x_init = 1.4
    sig_y_init = 1.4
    h_init = max(1.0, float(np.max(roi_raw)) - b0_init - rx_init - ry_init)

    is_hypermet = (fit_type == "hypermet")
    is_tail = (fit_type == "gaussian_tail")

    if is_hypermet:
        n_params = 16
        eta_tx_init = 0.15 * math.sqrt(2.0 * math.pi) * sig_x_init
        betax_init = 1.5 * sig_x_init
        stx_init = 0.02
        eta_ty_init = 0.15 * math.sqrt(2.0 * math.pi) * sig_y_init
        betay_init = 1.5 * sig_y_init
        sty_init = 0.02
        theta = np.array([
            b0_init, bx_init, by_init, rx_init, ry_init, h_init,
            mu_x_init, mu_y_init, sig_x_init, sig_y_init,
            eta_tx_init, betax_init, stx_init,
            eta_ty_init, betay_init, sty_init
        ], dtype=np.float64)
    elif is_tail:
        n_params = 12
        theta = np.array([b0_init, bx_init, by_init, rx_init, ry_init, h_init, mu_x_init, mu_y_init, sig_x_init, sig_y_init, 1.5, 1.5], dtype=np.float64)
    else:
        n_params = 10
        theta = np.array([b0_init, bx_init, by_init, rx_init, ry_init, h_init, mu_x_init, mu_y_init, sig_x_init, sig_y_init], dtype=np.float64)

    sigma_z = np.sqrt(np.maximum(1.0, z_raw_flat))
    weights = 1.0 / sigma_z

    def _calc_1d_hyp_profile(coords, mu, sig, eta_t, beta, step_amp):
        d = coords - mu
        z = d / sig
        g = np.exp(np.clip(-0.5 * z**2, -50.0, 0.0))
        u = np.clip(d / beta + 0.5 * (sig / beta)**2, -50.0, 50.0)
        v = np.clip(d / (math.sqrt(2.0) * sig) + sig / (math.sqrt(2.0) * beta), -20.0, 20.0)
        t = (eta_t / (2.0 * beta)) * np.exp(u) * _vec_erfc(v)
        s = 0.5 * step_amp * _vec_erfc(np.clip(z / math.sqrt(2.0), -20.0, 20.0))
        return g + t + s

    def calc_residuals_and_jacobian(p):
        if is_hypermet:
            b0, bx, by, rx, ry, H, mx, my, sx, sy, eta_tx, betax, stx, eta_ty, betay, sty = p
            sx, sy = max(0.2, abs(sx)), max(0.2, abs(sy))
            betax, betay = max(0.2, abs(betax)), max(0.2, abs(betay))
            eta_tx, eta_ty = max(0.0, eta_tx), max(0.0, eta_ty)
            stx, sty = max(0.0, stx), max(0.0, sty)
            H, rx, ry = max(0.0, H), max(0.0, rx), max(0.0, ry)

            dxc = x_flat - center_x_coord
            dyc = y_flat - center_y_coord
            bg_cont = b0 + bx * dxc + by * dyc

            px = _calc_1d_hyp_profile(x_flat, mx, sx, eta_tx, betax, stx)
            py = _calc_1d_hyp_profile(y_flat, my, sy, eta_ty, betay, sty)
            p2d = px * py

            model = bg_cont + rx * px + ry * py + H * p2d
            r = (model - z_raw_flat) * weights

            J = np.zeros((N_pixels, n_params), dtype=np.float64)
            eps = 1e-6
            for i in range(n_params):
                p_step = p.copy()
                p_step[i] += eps
                b0_s, bx_s, by_s, rx_s, ry_s, H_s, mx_s, my_s, sx_s, sy_s, etx_s, bx_t_s, stx_s, ety_s, by_t_s, sty_s = p_step
                sx_s, sy_s = max(0.2, abs(sx_s)), max(0.2, abs(sy_s))
                bx_t_s, by_t_s = max(0.2, abs(bx_t_s)), max(0.2, abs(by_t_s))
                etx_s, ety_s = max(0.0, etx_s), max(0.0, ety_s)
                stx_s, sty_s = max(0.0, stx_s), max(0.0, sty_s)
                H_s, rx_s, ry_s = max(0.0, H_s), max(0.0, rx_s), max(0.0, ry_s)

                bg_s = b0_s + bx_s * dxc + by_s * dyc
                px_s = _calc_1d_hyp_profile(x_flat, mx_s, sx_s, etx_s, bx_t_s, stx_s)
                py_s = _calc_1d_hyp_profile(y_flat, my_s, sy_s, ety_s, by_t_s, sty_s)
                mod_s = bg_s + rx_s * px_s + ry_s * py_s + H_s * (px_s * py_s)
                J[:, i] = (mod_s - model) / eps * weights
            return r, J, model

        if is_tail:
            b0, bx, by, rx, ry, H, mx, my, sx, sy, ax, ay = p
            ax = max(0.3, min(5.0, ax))
            ay = max(0.3, min(5.0, ay))
        else:
            b0, bx, by, rx, ry, H, mx, my, sx, sy = p
            ax, ay = 100.0, 100.0

        sx = max(0.2, abs(sx))
        sy = max(0.2, abs(sy))
        H = max(0.0, H)
        rx = max(0.0, rx)
        ry = max(0.0, ry)

        dx = x_flat - mx
        dy = y_flat - my
        dxc = x_flat - ix
        dyc = y_flat - iy
        zx = dx / sx
        zy = dy / sy

        is_gx = (zx >= -ax)
        is_gy = (zy >= -ay)

        gx_g = np.exp(np.clip(-0.5 * zx**2, -50.0, 0.0))
        gx_t = np.exp(np.clip(0.5 * ax**2 + ax * zx, -50.0, 50.0))
        gx = np.where(is_gx, gx_g, gx_t)

        gy_g = np.exp(np.clip(-0.5 * zy**2, -50.0, 0.0))
        gy_t = np.exp(np.clip(0.5 * ay**2 + ay * zy, -50.0, 50.0))
        gy = np.where(is_gy, gy_g, gy_t)

        g2d = gx * gy

        bg_cont = b0 + bx * dxc + by * dyc
        model = bg_cont + rx * gx + ry * gy + H * g2d
        r = (model - z_raw_flat) * weights

        J = np.zeros((N_pixels, n_params), dtype=np.float64)
        J[:, 0] = weights
        J[:, 1] = weights * dxc
        J[:, 2] = weights * dyc
        J[:, 3] = weights * gx
        J[:, 4] = weights * gy
        J[:, 5] = weights * g2d

        dgx_dmx = np.where(is_gx, gx * zx / sx, -gx * ax / sx)
        J[:, 6] = weights * (rx * dgx_dmx + H * gy * dgx_dmx)

        dgy_dmy = np.where(is_gy, gy * zy / sy, -gy * ay / sy)
        J[:, 7] = weights * (ry * dgy_dmy + H * gx * dgy_dmy)

        dgx_dsx = np.where(is_gx, gx * (zx**2) / sx, -gx * (ax * zx) / sx)
        J[:, 8] = weights * (rx * dgx_dsx + H * gy * dgx_dsx)

        dgy_dsy = np.where(is_gy, gy * (zy**2) / sy, -gy * (ay * zy) / sy)
        J[:, 9] = weights * (ry * dgy_dsy + H * gx * dgy_dsy)

        if is_tail:
            dgx_dax = np.where(is_gx, 0.0, gx * (ax + zx))
            J[:, 10] = weights * (rx * dgx_dax + H * gy * dgx_dax)

            dgy_day = np.where(is_gy, 0.0, gy * (ay + zy))
            J[:, 11] = weights * (ry * dgy_day + H * gx * dgy_day)

        return r, J, model

    lam = 0.001
    max_iters = 80
    r, J, model = calc_residuals_and_jacobian(theta)
    chi2 = np.sum(r**2)

    for _ in range(max_iters):
        JT = J.T
        JTJ = JT @ J
        diag_JTJ = np.diag(np.diag(JTJ))
        Hessian = JTJ + lam * np.maximum(diag_JTJ, 1e-4 * np.eye(n_params))
        gradient = JT @ r

        try:
            d_theta = np.linalg.solve(Hessian, -gradient)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        theta_new = theta + d_theta
        theta_new[3] = max(0.0, theta_new[3])  # Rx
        theta_new[4] = max(0.0, theta_new[4])  # Ry
        theta_new[5] = max(0.0, theta_new[5])  # H
        theta_new[6] = max(x_min, min(x_max, theta_new[6]))
        theta_new[7] = max(y_min, min(y_max, theta_new[7]))
        theta_new[8] = max(0.2, min(roi_half_width, abs(theta_new[8])))
        theta_new[9] = max(0.2, min(roi_half_width, abs(theta_new[9])))
        if is_hypermet:
            theta_new[10] = max(0.0, min(roi_half_width, theta_new[10]))  # eta_tx
            theta_new[11] = max(0.2, min(roi_half_width, theta_new[11]))  # betax
            theta_new[12] = max(0.0, min(1.0, theta_new[12]))  # stx
            theta_new[13] = max(0.0, min(roi_half_width, theta_new[13]))  # eta_ty
            theta_new[14] = max(0.2, min(roi_half_width, theta_new[14]))  # betay
            theta_new[15] = max(0.0, min(1.0, theta_new[15]))  # sty
        elif is_tail:
            theta_new[10] = max(0.3, min(5.0, theta_new[10]))
            theta_new[11] = max(0.3, min(5.0, theta_new[11]))

        r_new, J_new, model_new = calc_residuals_and_jacobian(theta_new)
        chi2_new = np.sum(r_new**2)

        if chi2_new < chi2:
            lam = max(1e-7, lam * 0.3)
            theta = theta_new
            r = r_new
            J = J_new
            model = model_new
            if (chi2 - chi2_new) / (chi2 + 1e-12) < 1e-7:
                chi2 = chi2_new
                break
            chi2 = chi2_new
        else:
            lam = min(1e7, lam * 5.0)

    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J)

    param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))

    if is_hypermet:
        b0, bx, by, rx, ry, H, mx, my, sx, sy, eta_tx, betax, stx, eta_ty, betay, sty = theta
        sx, sy = abs(sx), abs(sy)
        vol = H * (math.sqrt(2.0 * math.pi) * sx + eta_tx) * (math.sqrt(2.0 * math.pi) * sy + eta_ty)

        grad_vol = np.zeros(n_params)
        grad_vol[5] = (math.sqrt(2.0 * math.pi) * sx + eta_tx) * (math.sqrt(2.0 * math.pi) * sy + eta_ty)
        grad_vol[8] = H * math.sqrt(2.0 * math.pi) * (math.sqrt(2.0 * math.pi) * sy + eta_ty)
        grad_vol[9] = H * math.sqrt(2.0 * math.pi) * (math.sqrt(2.0 * math.pi) * sx + eta_tx)
        grad_vol[10] = H * (math.sqrt(2.0 * math.pi) * sy + eta_ty)
        grad_vol[13] = H * (math.sqrt(2.0 * math.pi) * sx + eta_tx)
        vol_err = float(np.sqrt(np.maximum(0.0, grad_vol @ cov @ grad_vol)))
        ax_val, ay_val = None, None
        ax_err, ay_err = None, None
        hyper_res = {
            "eta_tx": round(float(eta_tx), 3),
            "eta_tx_err": round(float(param_errors[10]), 3),
            "beta_x": round(float(betax), 3),
            "beta_x_err": round(float(param_errors[11]), 3),
            "step_x": round(float(stx), 3),
            "eta_ty": round(float(eta_ty), 3),
            "eta_ty_err": round(float(param_errors[13]), 3),
            "beta_y": round(float(betay), 3),
            "beta_y_err": round(float(param_errors[14]), 3),
            "step_y": round(float(sty), 3),
        }
    elif is_tail:
        b0, bx, by, rx, ry, H, mx, my, sx, sy, ax, ay = theta
        sx, sy = abs(sx), abs(sy)
        k_x = math.sqrt(math.pi / 2.0) * (1.0 + math.erf(ax / math.sqrt(2.0))) + math.exp(-0.5 * ax**2) / ax
        k_y = math.sqrt(math.pi / 2.0) * (1.0 + math.erf(ay / math.sqrt(2.0))) + math.exp(-0.5 * ay**2) / ay
        vol = H * sx * sy * k_x * k_y

        dvol_dH = sx * sy * k_x * k_y
        dvol_dsx = H * sy * k_x * k_y
        dvol_dsy = H * sx * k_x * k_y
        grad_vol = np.zeros(n_params)
        grad_vol[5] = dvol_dH
        grad_vol[8] = dvol_dsx
        grad_vol[9] = dvol_dsy
        vol_err = float(np.sqrt(np.maximum(0.0, grad_vol @ cov @ grad_vol)))
        ax_val, ay_val = round(float(ax), 3), round(float(ay), 3)
        ax_err = round(float(param_errors[10]), 3) if param_errors[10] < 1000.0 else None
        ay_err = round(float(param_errors[11]), 3) if param_errors[11] < 1000.0 else None
        hyper_res = {}
    else:
        b0, bx, by, rx, ry, H, mx, my, sx, sy = theta
        sx, sy = abs(sx), abs(sy)
        vol = 2.0 * np.pi * H * sx * sy
        grad_vol = 2.0 * np.pi * np.array([sx * sy, H * sy, H * sx])
        sub_cov = cov[np.ix_([5, 8, 9], [5, 8, 9])]
        var_vol = float(grad_vol.T @ sub_cov @ grad_vol)
        vol_err = np.sqrt(max(0.0, var_vol))
        ax_val, ay_val = None, None
        ax_err, ay_err = None, None
        hyper_res = {}

    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_x = fwhm_factor * sx
    fwhm_x_err = fwhm_factor * param_errors[8]
    fwhm_y = fwhm_factor * sy
    fwhm_y_err = fwhm_factor * param_errors[9]

    if cal_x is None:
        if isinstance(cal, dict):
            cal_x = cal.get(0, [0.0, 1.0, 0.0])
        elif isinstance(cal, (list, tuple)) and len(cal) > 0 and isinstance(cal[0], (list, tuple)):
            cal_x = cal[0]
        else:
            cal_x = cal if cal is not None else [0.0, 1.0, 0.0]

    if cal_y is None:
        if isinstance(cal, dict):
            cal_y = cal.get(1, [0.0, 1.0, 0.0])
        elif isinstance(cal, (list, tuple)) and len(cal) > 1 and isinstance(cal[1], (list, tuple)):
            cal_y = cal[1]
        else:
            cal_y = cal if cal is not None else [0.0, 1.0, 0.0]

    cal_x = parse_cal_coefficients(cal_x)
    cal_y = parse_cal_coefficients(cal_y)

    e_x = ch_to_energy(mx, cal_x)
    e_x_err = param_errors[6] * de_dch(mx, cal_x)
    e_y = ch_to_energy(my, cal_y)
    e_y_err = param_errors[7] * de_dch(my, cal_y)

    fwhm_e_x = fwhm_x * de_dch(mx, cal_x)
    fwhm_e_x_err = fwhm_x_err * de_dch(mx, cal_x)
    fwhm_e_y = fwhm_y * de_dch(my, cal_y)
    fwhm_e_y_err = fwhm_y_err * de_dch(my, cal_y)

    ndf = max(1, N_pixels - n_params)
    red_chi2 = chi2 / ndf

    # Background decomposition counts in ROI from continuous model
    dx = x_flat - mx
    dy = y_flat - my
    dxc = x_flat - center_x_coord
    dyc = y_flat - center_y_coord
    zx = dx / sx
    zy = dy / sy
    if is_hypermet:
        gx_val = _calc_1d_hyp_profile(x_flat, mx, sx, eta_tx, betax, stx)
        gy_val = _calc_1d_hyp_profile(y_flat, my, sy, eta_ty, betay, sty)
    elif is_tail:
        gx_val = np.where(zx >= -ax, np.exp(-0.5 * zx**2), np.exp(np.clip(0.5 * ax**2 + ax * zx, -50.0, 50.0)))
        gy_val = np.where(zy >= -ay, np.exp(-0.5 * zy**2), np.exp(np.clip(0.5 * ay**2 + ay * zy, -50.0, 50.0)))
    else:
        gx_val = np.exp(-0.5 * zx**2)
        gy_val = np.exp(-0.5 * zy**2)

    cont_counts = float(np.sum(b0 + bx * dxc + by * dyc))
    gx_2d = np.tile(gx_val[:Nx], Ny)
    gy_2d = np.repeat(gy_val[::Nx], Nx)
    ridge_x_counts = float(rx * np.sum(gx_2d))
    ridge_y_counts = float(ry * np.sum(gy_2d))
    total_bg_counts = cont_counts + ridge_x_counts + ridge_y_counts

    # Discrete 4-region Gamba & Morhác decomposition (Eqs. 4 & 14 in Gamba et al., NIM A 928)
    # Define peak region as ±2 sigma around centroid
    w_gx = max(1, int(round(2.0 * sx)))
    w_gy = max(1, int(round(2.0 * sy)))
    ix_c = int(np.floor(mx)) - x_min
    iy_c = int(np.floor(my)) - y_min

    px0 = max(0, ix_c - w_gx)
    px1 = min(Nx - 1, ix_c + w_gx)
    py0 = max(0, iy_c - w_gy)
    py1 = min(Ny - 1, iy_c + w_gy)

    # Masks for 4 regions
    mask_peak_x = np.zeros(Nx, dtype=bool)
    mask_peak_x[px0:px1+1] = True
    mask_bg_x = ~mask_peak_x

    mask_peak_y = np.zeros(Ny, dtype=bool)
    mask_peak_y[py0:py1+1] = True
    mask_bg_y = ~mask_peak_y

    area_pp = int(np.sum(mask_peak_y)) * int(np.sum(mask_peak_x))
    area_pbg = int(np.sum(mask_bg_y)) * int(np.sum(mask_peak_x))
    area_bgp = int(np.sum(mask_peak_y)) * int(np.sum(mask_bg_x))
    area_bgbg = int(np.sum(mask_bg_y)) * int(np.sum(mask_bg_x))

    # Raw counts in each region
    roi_pp = roi_raw[py0:py1+1, px0:px1+1]
    n_pp_m = float(np.sum(roi_pp))

    n_pbg_raw = float(np.sum(roi_raw[mask_bg_y, :][:, mask_peak_x])) if area_pbg > 0 else 0.0
    n_bgp_raw = float(np.sum(roi_raw[mask_peak_y, :][:, mask_bg_x])) if area_bgp > 0 else 0.0
    n_bgbg_raw = float(np.sum(roi_raw[mask_bg_y, :][:, mask_bg_x])) if area_bgbg > 0 else 0.0

    # Normalized counts to peak gate area
    s_pbg = (area_pp / max(1, area_pbg)) if area_pbg > 0 else 0.0
    s_bgp = (area_pp / max(1, area_bgp)) if area_bgp > 0 else 0.0
    s_bgbg = (area_pp / max(1, area_bgbg)) if area_bgbg > 0 else 0.0

    n_pbg_m = n_pbg_raw * s_pbg
    n_bgp_m = n_bgp_raw * s_bgp
    n_bgbg_m = n_bgbg_raw * s_bgbg

    # True net peak counts (Gamba Eq. 4 / Eq. 14): n_pp_t = n_pp_m - n_pbg_m - n_bgp_m + n_bgbg_m
    n_pp_t = n_pp_m - n_pbg_m - n_bgp_m + n_bgbg_m
    var_gamba = n_pp_m + (s_pbg**2 * n_pbg_raw) + (s_bgp**2 * n_bgp_raw) + (s_bgbg**2 * n_bgbg_raw)
    n_pp_t_err = float(np.sqrt(max(0.0, var_gamba)))

    # Peak-to-Total-Background ratio Pi (Gamba Eq. 17)
    pi_ratio = n_pp_t / max(1.0, n_pp_m)

    res_dict = {
        "success": True,
        "is_2d": True,
        "fit_type": fit_type,
        "centroid_x_ch": round(float(mx), 3),
        "centroid_x_ch_err": round(float(param_errors[6]), 3),
        "centroid_y_ch": round(float(my), 3),
        "centroid_y_ch_err": round(float(param_errors[7]), 3),
        "centroid_x_e": round(float(e_x), 2),
        "centroid_x_e_err": round(float(e_x_err), 2),
        "centroid_y_e": round(float(e_y), 2),
        "centroid_y_e_err": round(float(e_y_err), 2),
        "volume": round(float(vol), 1),
        "volume_err": round(float(vol_err), 1),
        "gamba_net": round(float(n_pp_t), 1),
        "gamba_net_err": round(float(n_pp_t_err), 1),
        "pi_ratio": round(float(pi_ratio), 4),
        "pi_ratio_percent": round(float(pi_ratio * 100.0), 2),
        "fwhm_x_ch": round(float(fwhm_x), 3),
        "fwhm_x_ch_err": round(float(fwhm_x_err), 3),
        "fwhm_y_ch": round(float(fwhm_y), 3),
        "fwhm_y_ch_err": round(float(fwhm_y_err), 3),
        "fwhm_x_e": round(float(fwhm_e_x), 2),
        "fwhm_x_e_err": round(float(fwhm_e_x_err), 2),
        "fwhm_y_e": round(float(fwhm_e_y), 2),
        "fwhm_y_e_err": round(float(fwhm_e_y_err), 2),
        "amplitude": round(float(H), 1),
        "amplitude_err": round(float(param_errors[5]), 1),
        "alpha_x": ax_val,
        "alpha_x_err": ax_err,
        "alpha_y": ay_val,
        "alpha_y_err": ay_err,
        "gross_counts": round(gross_counts, 1),
        "total_bg_counts": round(total_bg_counts, 1),
        "cont_counts": round(cont_counts, 1),
        "ridge_x_counts": round(ridge_x_counts, 1),
        "ridge_y_counts": round(ridge_y_counts, 1),
        "n_pp_m": round(float(n_pp_m), 1),
        "n_pbg_m": round(float(n_pbg_m), 1),
        "n_bgp_m": round(float(n_bgp_m), 1),
        "n_bgbg_m": round(float(n_bgbg_m), 1),
        "ridge_x": round(float(rx), 1),
        "ridge_x_err": round(float(param_errors[3]), 1),
        "ridge_y": round(float(ry), 1),
        "ridge_y_err": round(float(param_errors[4]), 1),
        "bg_b0": round(float(b0), 2),
        "bg_bx": round(float(bx), 4),
        "bg_by": round(float(by), 4),
        "chi2": round(float(chi2), 2),
        "ndf": int(ndf),
        "red_chi2": round(float(red_chi2), 3),
        "roi_x_min": int(x_min),
        "roi_x_max": int(x_max),
        "roi_y_min": int(y_min),
        "roi_y_max": int(y_max)
    }
    res_dict.update(hyper_res)
    return res_dict


def fit_2d_gaussian_peak(
    matrix, x_center, y_center, fit_type="gaussian", cal=[0.0, 1.0, 0.0], roi_half_width=16,
    proj_x=None, proj_y=None, total_counts=None, recenter=True, cal_x=None, cal_y=None, **kwargs
):
    """
    Fits a true 2D coincidence peak (Symmetric Gaussian, RadWare Tail, or Hypermet Model)
    on the 2D gamma-gamma coincidence matrix using the self-consistent 4-component background
    decomposition established by Gamba et al. (NIM A 928, 2019, 93-103) & Morhác et al. (NIM A 401, 1997, 113):
      - bg|bg: 2D Compton continuum + accidental random coincidences: b0 + bx*(x - x_c) + by*(y - y_c)
      - p|bg : Det 1 peak with Det 2 Compton/random continuum ridge: R_x * P_X(x)
      - bg|p : Det 2 peak with Det 1 Compton/random continuum ridge: R_y * P_Y(y)
      - p|p^t: True 2D coincidence peak volume: H * P_X(x) * P_Y(y)
    
    Performs a 2-pass iterative recentering: after a preliminary fit at the initial pointer location,
    it readjusts the ROI center to the fitted peak centroid to ensure perfect symmetry and invariance
    to the initial pointer position.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    H_mat, W_mat = mat.shape

    # Pass 1: Preliminary fit around initial user pointer position
    res_prelim = _fit_2d_gaussian_single_roi(
        mat, x_center, y_center, fit_type=fit_type, cal=cal, roi_half_width=roi_half_width,
        proj_x=proj_x, proj_y=proj_y, total_counts=total_counts, cal_x=cal_x, cal_y=cal_y, **kwargs
    )

    if not recenter or not res_prelim.get("success"):
        return res_prelim

    # Check fitted centroid coordinates
    mx = res_prelim.get("centroid_x_ch", x_center)
    my = res_prelim.get("centroid_y_ch", y_center)
    ix_orig = int(np.floor(x_center))
    iy_orig = int(np.floor(y_center))
    ix_new = int(np.floor(mx))
    iy_new = int(np.floor(my))

    # If peak is offset from initial click position, recenter ROI and perform refined Pass 2
    if (ix_new != ix_orig or iy_new != iy_orig) or (abs(mx - (ix_orig + 0.5)) > 0.4 or abs(my - (iy_orig + 0.5)) > 0.4):
        ix_clamped = max(roi_half_width + 1, min(W_mat - 1 - roi_half_width - 1, ix_new))
        iy_clamped = max(roi_half_width + 1, min(H_mat - 1 - roi_half_width - 1, iy_new))
        try:
            res_refined = _fit_2d_gaussian_single_roi(
                mat, ix_clamped + 0.5, iy_clamped + 0.5, fit_type=fit_type, cal=cal, roi_half_width=roi_half_width,
                proj_x=proj_x, proj_y=proj_y, total_counts=total_counts, cal_x=cal_x, cal_y=cal_y, **kwargs
            )
            if res_refined.get("success"):
                return res_refined
        except Exception:
            pass

    return res_prelim


def print_fit_2d_terminal_report(res, filename, is_cal, verbosity="compact"):
    ft = res.get("fit_type", "gaussian")
    model_tag = "Hypermet" if ft == "hypermet" else ("RadWare" if ft == "gaussian_tail" else "Gaussian")

    if isinstance(is_cal, (list, tuple)):
        is_cal_x, is_cal_y = bool(is_cal[0]), bool(is_cal[1])
    else:
        is_cal_x, is_cal_y = bool(is_cal), bool(is_cal)

    unit_x = "keV" if is_cal_x else "ch"
    unit_y = "keV" if is_cal_y else "ch"
    dec_x = 2 if is_cal_x else 3
    dec_y = 2 if is_cal_y else 3

    if verbosity == "compact":
        cx_val = res['centroid_x_e'] if is_cal_x else res['centroid_x_ch']
        cx_err = res['centroid_x_e_err'] if is_cal_x else res['centroid_x_ch_err']
        cy_val = res['centroid_y_e'] if is_cal_y else res['centroid_y_ch']
        cy_err = res['centroid_y_e_err'] if is_cal_y else res['centroid_y_ch_err']

        fx_val = res['fwhm_x_e'] if is_cal_x else res['fwhm_x_ch']
        fx_err = res['fwhm_x_e_err'] if is_cal_x else res['fwhm_x_ch_err']
        fy_val = res['fwhm_y_e'] if is_cal_y else res['fwhm_y_ch']
        fy_err = res['fwhm_y_e_err'] if is_cal_y else res['fwhm_y_ch_err']

        # Align centroid integer part and error width so decimal points line up
        cx_fmt = f"{cx_val:.{dec_x}f}"
        cy_fmt = f"{cy_val:.{dec_y}f}"
        cx_int, cx_dec = cx_fmt.split(".")
        cy_int, cy_dec = cy_fmt.split(".")
        max_c_int = max(len(cx_int), len(cy_int))
        cx_err_s = f"{cx_err:.{dec_x}f}"
        cy_err_s = f"{cy_err:.{dec_y}f}"
        max_c_err = max(len(cx_err_s), len(cy_err_s))
        cx_str = f"{cx_int:>{max_c_int}}.{cx_dec}({cx_err_s:>{max_c_err}})"
        cy_str = f"{cy_int:>{max_c_int}}.{cy_dec}({cy_err_s:>{max_c_err}})"

        # Align FWHM integer part and error width so decimal points line up
        fx_fmt = f"{fx_val:.{dec_x}f}"
        fy_fmt = f"{fy_val:.{dec_y}f}"
        fx_int, fx_dec = fx_fmt.split(".")
        fy_int, fy_dec = fy_fmt.split(".")
        max_f_int = max(len(fx_int), len(fy_int))
        fx_err_s = f"{fx_err:.{dec_x}f}"
        fy_err_s = f"{fy_err:.{dec_y}f}"
        max_f_err = max(len(fx_err_s), len(fy_err_s))
        fx_str = f"{fx_int:>{max_f_int}}.{fx_dec}({fx_err_s:>{max_f_err}})"
        fy_str = f"{fy_int:>{max_f_int}}.{fy_dec}({fy_err_s:>{max_f_err}})"

        vol_str = f"{res['volume']:.1f}({res['volume_err']:.1f})"

        print(f"[2D Fit] [{filename}] ({model_tag} + Gamba BG):", flush=True)
        print(f"  Det 1 (X):\tCentroid: {cx_str} {unit_x}\tArea: {vol_str} counts\tFWHM: {fx_str} {unit_x}", flush=True)
        print(f"  Det 2 (Y):\tCentroid: {cy_str} {unit_y}\tArea: {vol_str} counts\tFWHM: {fy_str} {unit_y}", flush=True)
        print(f"  Gamba Net Area (p|p^t): {res['gamba_net']:.1f} ± {res['gamba_net_err']:.1f} counts\tPeak/Total-BG Ratio (Π): {res['pi_ratio_percent']:.1f}%\n", flush=True)
        return

    bar = "═" * 80
    subbar = "─" * 80
    if ft == "hypermet":
        model_name = "Hypermet Model (Convolved Tail + Erfc Step Profile)"
    elif ft == "gaussian_tail":
        model_name = "Gaussian with Left Tail (RadWare / HPGe)"
    else:
        model_name = "Standard Symmetric Gaussian"

    print(f"\n{bar}")
    print(f"[GASPware 2D Coincidence Peak Fit] - {filename} ({model_name})")
    print(f"   [Self-Consistent 4-Component BG Decomposition: Gamba et al., NIM A 928 (2019) 93]")
    print(subbar)
    if is_cal_x or is_cal_y:
        cx_s = f"{res['centroid_x_e']:.2f} ± {res['centroid_x_e_err']:.2f} keV" if is_cal_x else f"{res['centroid_x_ch']:.3f} ch"
        cy_s = f"{res['centroid_y_e']:.2f} ± {res['centroid_y_e_err']:.2f} keV" if is_cal_y else f"{res['centroid_y_ch']:.3f} ch"
        print(f"  2D Centroid         : Det 1 (X) = {cx_s} | Det 2 (Y) = {cy_s}")
        print(f"  2D Centroid (ch)    : ({res['centroid_x_ch']:.3f} ± {res['centroid_x_ch_err']:.3f}, {res['centroid_y_ch']:.3f} ± {res['centroid_y_ch_err']:.3f}) ch")
        fx_s = f"{res['fwhm_x_e']:.2f} ± {res['fwhm_x_e_err']:.2f} keV" if is_cal_x else f"{res['fwhm_x_ch']:.3f} ch"
        fy_s = f"{res['fwhm_y_e']:.2f} ± {res['fwhm_y_e_err']:.2f} keV" if is_cal_y else f"{res['fwhm_y_ch']:.3f} ch"
        print(f"  FWHM                : Det 1 (X) = {fx_s} | Det 2 (Y) = {fy_s}")
        print(f"  FWHM (ch)           : Det 1 (X) = {res['fwhm_x_ch']:.3f} ± {res['fwhm_x_ch_err']:.3f} ch | Det 2 (Y) = {res['fwhm_y_ch']:.3f} ± {res['fwhm_y_ch_err']:.3f} ch")
    else:
        print(f"  2D Centroid (ch)    : ({res['centroid_x_ch']:.3f} ± {res['centroid_x_ch_err']:.3f}, {res['centroid_y_ch']:.3f} ± {res['centroid_y_ch_err']:.3f}) ch")
        print(f"  FWHM (ch)           : Det 1 (X) = {res['fwhm_x_ch']:.3f} ± {res['fwhm_x_ch_err']:.3f} ch | Det 2 (Y) = {res['fwhm_y_ch']:.3f} ± {res['fwhm_y_ch_err']:.3f} ch")
    print(f"  Fitted Net Volume   : {res['volume']:10.1f} ± {res['volume_err']:<6.1f} counts  (Integrated 2D Peak)")
    print(f"  Gamba Gate Net (p|p): {res['gamba_net']:10.1f} ± {res['gamba_net_err']:<6.1f} counts  [Π Ratio: {res['pi_ratio_percent']:.1f}%]")
    print(f"  Peak Amplitude (H)  : {res['amplitude']:10.1f} ± {res['amplitude_err']:<6.1f} counts")
    if res.get("eta_tx") is not None:
        print(f"  Hypermet Tails (η)  : η_X = {res['eta_tx']:.3f} (β_X = {res['beta_x']:.2f}) | η_Y = {res['eta_ty']:.3f} (β_Y = {res['beta_y']:.2f})")
    elif res.get("alpha_x") is not None:
        ax_str = f"{res['alpha_x']:.3f}" + (f" ± {res['alpha_x_err']:.3f}" if res.get('alpha_x_err') else "")
        ay_str = f"{res['alpha_y']:.3f}" + (f" ± {res['alpha_y_err']:.3f}" if res.get('alpha_y_err') else "")
        print(f"  Left Tail Joins (α) : α_X = {ax_str} | α_Y = {ay_str}")
    print(subbar)
    print(f"  Gamba & Morhác 4-Component Background Decomposition:")
    print(f"    • 2D Continuum (bg|bg): {res.get('cont_counts', 0.0):10.1f} counts (b0={res['bg_b0']:.2f}, bx={res['bg_bx']:.4f}, by={res['bg_by']:.4f})")
    print(f"    • Cross-Ridge Det 1 (p|bg): {res.get('ridge_x_counts', 0.0):10.1f} counts (Rx={res['ridge_x']:.1f} ± {res['ridge_x_err']:.1f})")
    print(f"    • Cross-Ridge Det 2 (bg|p): {res.get('ridge_y_counts', 0.0):10.1f} counts (Ry={res['ridge_y']:.1f} ± {res['ridge_y_err']:.1f})")
    print(f"    • Gamba Discrete Gates: n_pp^m={res['n_pp_m']:.1f}, n_pbg^m={res['n_pbg_m']:.1f}, n_bgp^m={res['n_bgp_m']:.1f}, n_bgbg^m={res['n_bgbg_m']:.1f}")
    print(f"    • Total Background    : {res.get('total_bg_counts', 0.0):10.1f} counts | Gross in ROI: {res.get('gross_counts', 0.0):10.1f} counts")
    print(f"  2D Fit ROI Window   : Det 1 (X)=[ch {res['roi_x_min']}..{res['roi_x_max']}], Det 2 (Y)=[ch {res['roi_y_min']}..{res['roi_y_max']}]")
    print(f"  Reduced Chi2 / NDF  : {res['red_chi2']:.3f} (Chi2 = {res['chi2']:.1f}, NDF = {res['ndf']})")
    print(f"{bar}\n", flush=True)


def generate_pdf_1d(spec, ch_start, ch_end, is_log=False, zoom_y=1.0, fit_res=None, cal=None, axis_label=None, title=None):
    """
    Generates a publication-quality 1D spectrum vector PDF with white background,
    Times New Roman font, inward ticks, stepped staircase histogram, and optional calibration.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ch_start = int(max(0, ch_start))
    ch_end = int(min(len(spec) - 1, ch_end))
    sub_x = np.arange(ch_start, ch_end + 1, dtype=np.float64)
    sub_y = spec[ch_start:ch_end + 1]

    is_cal = is_calibrated_coeffs(cal)
    if is_cal:
        plot_x = np.array([ch_to_energy(c, cal) for c in sub_x])
        x_lim_0 = ch_to_energy(ch_start, cal)
        x_lim_1 = ch_to_energy(ch_end, cal)
        x_axis_name = axis_label or "Energy (keV)"
    else:
        plot_x = sub_x
        x_lim_0 = ch_start
        x_lim_1 = ch_end
        x_axis_name = axis_label or "Channel"

    # Stepped histogram centered on bins
    ax.step(plot_x, sub_y, where="mid", color="#111111", linewidth=1.0, label="Data")

    max_val = float(np.max(sub_y)) if len(sub_y) > 0 else 1.0
    min_val = float(np.min(sub_y)) if len(sub_y) > 0 else 0.0

    if is_log:
        ax.set_yscale("log")
        log_min = max(1.0, min_val if min_val > 0 else 1.0)
        log_max = max(10.0, max_val)
        y_max = 10 ** (np.log10(log_min) + (np.log10(log_max) - np.log10(log_min) + 0.5) * zoom_y)
        ax.set_ylim(bottom=log_min, top=y_max)
    else:
        y_max = (max_val * 1.1) * zoom_y
        y_min = min(0.0, min_val * 1.1) if min_val < 0 else 0.0
        ax.set_ylim(bottom=y_min, top=max(1.0, y_max))
        if y_min < 0:
            ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8)

    ax.set_xlim(min(x_lim_0, x_lim_1), max(x_lim_0, x_lim_1))
    ax.set_xlabel(x_axis_name, fontsize=12, labelpad=6)
    ax.set_ylabel("Counts", fontsize=12, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)
    if title:
        ax.set_title(title, fontsize=12, pad=8)

    # Plot fitted peak curve and baseline if available
    if fit_res and fit_res.get("success"):
        curve_x = np.array(fit_res.get("curve_x", []), dtype=np.float64)
        curve_fit = np.array(fit_res.get("curve_fit", []), dtype=np.float64)
        curve_bg = np.array(fit_res.get("curve_bg", []), dtype=np.float64)

        if len(curve_x) > 0 and len(curve_fit) > 0:
            curve_x_plot = np.array([ch_to_energy(c, cal) for c in curve_x]) if is_cal else curve_x
            if len(curve_bg) > 0:
                ax.plot(curve_x_plot, curve_bg, color="#cc0066", linestyle="--", linewidth=1.2, label="Background")
            ax.plot(curve_x_plot, curve_fit, color="#d95f02", linestyle="-", linewidth=1.8, label="Fit")

            if "peaks" in fit_res and len(fit_res["peaks"]) > 0:
                p_list = fit_res["peaks"]
                for p in p_list:
                    p_mu = ch_to_energy(p["centroid_ch"], cal) if is_cal else p["centroid_ch"]
                    ax.axvline(p_mu, color="#d95f02", linestyle=":", linewidth=0.7, alpha=0.65)
                tot_a = sum(p.get("area", 0.0) for p in p_list)
                info_txt = f"Multi-Peak Fit: {len(p_list)} peaks\nTotal Net Area: {tot_a:,.1f} counts\nPeak-Aware Continuum BG"
                ax.text(0.04, 0.94, info_txt, transform=ax.transAxes, verticalalignment="top",
                        fontsize=9, fontfamily="serif", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999", alpha=0.92))
            else:
                mu = fit_res.get("centroid_ch")
                if mu is not None:
                    mu_plot = ch_to_energy(mu, cal) if is_cal else mu
                    ax.axvline(mu_plot, color="#d95f02", linestyle=":", linewidth=1.0)

                area_str = f"{fit_res.get('area', 0):.1f} ± {fit_res.get('area_err', 0):.1f}"
                if is_cal:
                    centroid_val = fit_res.get('centroid_e', ch_to_energy(fit_res.get('centroid_ch', 0), cal))
                    centroid_err = fit_res.get('centroid_e_err', fit_res.get('centroid_ch_err', 0) * de_dch(fit_res.get('centroid_ch', 0), cal))
                    fwhm_val = fit_res.get('fwhm_e', fit_res.get('fwhm_ch', 0) * de_dch(fit_res.get('centroid_ch', 0), cal))
                    fwhm_err = fit_res.get('fwhm_e_err', fit_res.get('fwhm_ch_err', 0) * de_dch(fit_res.get('centroid_ch', 0), cal))
                    unit_str = "keV"
                else:
                    centroid_val = fit_res.get('centroid_ch', 0)
                    centroid_err = fit_res.get('centroid_ch_err', 0)
                    fwhm_val = fit_res.get('fwhm_ch', 0)
                    fwhm_err = fit_res.get('fwhm_ch_err', 0)
                    unit_str = "ch"

                centroid_str = f"{centroid_val:.2f} ± {centroid_err:.2f}"
                fwhm_str = f"{fwhm_val:.2f} ± {fwhm_err:.2f}"
                info_txt = f"Centroid: {centroid_str} {unit_str}\nArea: {area_str} counts\nFWHM: {fwhm_str} {unit_str}\n$\\chi^2_\\nu$: {fit_res.get('red_chi2', 0):.2f}"
                ax.text(0.04, 0.94, info_txt, transform=ax.transAxes, verticalalignment="top",
                        fontsize=9, fontfamily="serif", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999", alpha=0.92))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf_2d(matrix, x0, x1, y0, y1, cmap_name="turbo", scale_mode="log", vmin=0, vmax=100, fit_2d_res=None, cal_x=None, cal_y=None, x_label=None, y_label=None, title=None):
    """
    Generates a publication-quality 2D coincidence matrix vector PDF with white background,
    Times New Roman font, per-axis calibration axes labels, colorbar, and optional fit overlays.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, PowerNorm, Normalize
    from matplotlib.patches import Ellipse
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5

    x0 = int(max(0, x0))
    x1 = int(min(matrix.shape[1], x1))
    y0 = int(max(0, y0))
    y1 = int(min(matrix.shape[0], y1))
    sub_mat = matrix[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if scale_mode == "log":
        norm = LogNorm(vmin=max(1, vmin if vmin > 0 else 1), vmax=max(2, vmax))
    elif scale_mode == "sqrt":
        norm = PowerNorm(gamma=0.5, vmin=max(0, vmin), vmax=max(1, vmax))
    else:
        norm = Normalize(vmin=vmin, vmax=max(1, vmax))

    try:
        cmap = plt.get_cmap(cmap_name)
    except Exception:
        cmap = plt.get_cmap("turbo")

    is_cal_x = is_calibrated_coeffs(cal_x)
    is_cal_y = is_calibrated_coeffs(cal_y)

    ext_x0 = ch_to_energy(x0, cal_x) if is_cal_x else x0
    ext_x1 = ch_to_energy(x1, cal_x) if is_cal_x else x1
    ext_y0 = ch_to_energy(y0, cal_y) if is_cal_y else y0
    ext_y1 = ch_to_energy(y1, cal_y) if is_cal_y else y1

    im = ax.imshow(sub_mat, extent=[ext_x0, ext_x1, ext_y0, ext_y1], origin="lower", cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    lbl_x = x_label or ("Det 1 (X) Energy (keV)" if is_cal_x else "Det 1 / X (Channel)")
    lbl_y = y_label or ("Det 2 (Y) Energy (keV)" if is_cal_y else "Det 2 / Y (Channel)")
    ax.set_xlabel(lbl_x, fontsize=12, labelpad=6)
    ax.set_ylabel(lbl_y, fontsize=12, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)
    if title:
        ax.set_title(title, fontsize=12, pad=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Counts", fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    # Draw 2D fit crosshair, ellipse and ROI box if active
    if fit_2d_res and fit_2d_res.get("success"):
        cx = fit_2d_res.get("centroid_x_ch", 0)
        cy = fit_2d_res.get("centroid_y_ch", 0)
        fwhm_x = fit_2d_res.get("fwhm_x_ch", 4.0)
        fwhm_y = fit_2d_res.get("fwhm_y_ch", 4.0)

        cx_plot = ch_to_energy(cx, cal_x) if is_cal_x else cx
        cy_plot = ch_to_energy(cy, cal_y) if is_cal_y else cy
        fx_plot = fwhm_x * de_dch(cx, cal_x) if is_cal_x else fwhm_x
        fy_plot = fwhm_y * de_dch(cy, cal_y) if is_cal_y else fwhm_y

        rx0 = fit_2d_res.get("roi_x_min", cx - 8)
        rx1 = fit_2d_res.get("roi_x_max", cx + 8) + 1
        ry0 = fit_2d_res.get("roi_y_min", cy - 8)
        ry1 = fit_2d_res.get("roi_y_max", cy + 8) + 1

        rx0_plot = ch_to_energy(rx0, cal_x) if is_cal_x else rx0
        rx1_plot = ch_to_energy(rx1, cal_x) if is_cal_x else rx1
        ry0_plot = ch_to_energy(ry0, cal_y) if is_cal_y else ry0
        ry1_plot = ch_to_energy(ry1, cal_y) if is_cal_y else ry1

        import matplotlib.patches as patches
        rect = patches.Rectangle((rx0_plot, ry0_plot), rx1_plot - rx0_plot, ry1_plot - ry0_plot, linewidth=1.0, edgecolor="#ffd600", facecolor="none", linestyle="--", alpha=0.7)
        ax.add_patch(rect)

        # Crosshair
        cross_w = fx_plot * 0.8
        cross_h = fy_plot * 0.8
        ax.plot([cx_plot - cross_w, cx_plot + cross_w], [cy_plot, cy_plot], color="#ff0066", linewidth=1.2)
        ax.plot([cx_plot, cx_plot], [cy_plot - cross_h, cy_plot + cross_h], color="#ff0066", linewidth=1.2)

        # Ellipse
        ell = Ellipse((cx_plot, cy_plot), width=fx_plot, height=fy_plot, angle=0, edgecolor="#ffd600", facecolor="none", linewidth=1.6)
        ax.add_patch(ell)

        # Text annotation with Net Volume and Centroid
        vol_val = fit_2d_res.get("volume", 0)
        vol_err = fit_2d_res.get("volume_err", 0)
        vol_str = f"Net Vol: {vol_val:,.0f} ± {vol_err:,.0f} cts" if vol_err else f"Net Vol: {vol_val:,.0f} cts"
        unit_x = "keV" if is_cal_x else "ch"
        unit_y = "keV" if is_cal_y else "ch"
        ax.text(0.02, 0.98, f"2D Coincidence Peak\nCentroid: ({cx_plot:.2f} {unit_x}, {cy_plot:.2f} {unit_y})\n{vol_str}",
                transform=ax.transAxes, verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#191c20", edgecolor="#ffd600", alpha=0.85),
                color="#ffffff", fontfamily="monospace")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_1d_ascii(filepath: Path, spec: np.ndarray, cal: list = None, header: str = "") -> None:
    """Export 1D spectrum to ASCII .dat file with columns: Channel, Energy (if calibrated), Counts."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    is_cal = is_calibrated_coeffs(cal)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# python-cmat 1D Spectrum Export\n")
        if header:
            f.write(f"# {header}\n")
        if is_cal:
            f.write("# Channel\tEnergy_keV\tCounts\n")
            for ch, val in enumerate(spec):
                e = ch_to_energy(ch, cal)
                f.write(f"{ch}\t{e:.4f}\t{val:.2f}\n")
        else:
            f.write("# Channel\tCounts\n")
            for ch, val in enumerate(spec):
                f.write(f"{ch}\t{val:.2f}\n")


def export_amat_ascii(filepath: Path, matrix: np.ndarray, header: str = "") -> None:
    """Export 2D matrix to ASCII matrix format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = matrix.shape
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# python-cmat 2D Matrix Export ({nx}x{ny})\n")
        if header:
            f.write(f"# {header}\n")
        for row in matrix:
            f.write(" ".join(str(int(v)) if float(v).is_integer() else f"{v:.2f}" for v in row) + "\n")


def parse_gate_ranges(param_str: str) -> list:
    """
    Parses gate intervals from JSON string (e.g. [[100, 110], [120, 130]])
    or semicolon/space/comma separated tokens.
    """
    if not param_str:
        return []
    param_str = param_str.strip()
    if param_str.startswith("["):
        try:
            parsed = json.loads(param_str)
            if isinstance(parsed, list):
                res = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        res.append([float(item[0]), float(item[1])])
                return res
        except Exception:
            pass
    pairs = []
    for chunk in param_str.replace(";", " ").replace("|", " ").split():
        if "," in chunk:
            parts = chunk.split(",")
        elif "-" in chunk:
            parts = chunk.split("-")
        else:
            continue
        if len(parts) == 2:
            try:
                pairs.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return pairs


def compute_1d_gate(matrix: np.ndarray, axis: int, w_gates: list, b_gates: list) -> dict:
    """
    Compute background-subtracted gated coincidence spectrum along the opposite axis.

    Args:
        matrix: 2D numpy array [res_y, res_x]
        axis: 0 to gate on Det 1 (X) and produce coincidence on Det 2 (Y);
              1 to gate on Det 2 (Y) and produce coincidence on Det 1 (X).
        w_gates: list of [min_ch, max_ch] pairs for peak gates
        b_gates: list of [min_ch, max_ch] pairs for background gates

    Returns:
        dict with net_spec, raw_spec, bg_spec, w_width, b_width, scale, etc.
    """
    res_y, res_x = matrix.shape
    dest_len = res_y if axis == 0 else res_x
    max_gate_ch = (res_x - 1) if axis == 0 else (res_y - 1)

    raw_w = np.zeros(dest_len, dtype=np.float64)
    raw_b = np.zeros(dest_len, dtype=np.float64)

    valid_w = []
    total_w_ch = 0
    for w in w_gates:
        c0 = max(0, min(int(round(float(w[0]))), int(round(float(w[1])))))
        c1 = min(max_gate_ch, max(int(round(float(w[0]))), int(round(float(w[1])))))
        if c1 >= c0:
            if axis == 0:
                raw_w += np.sum(matrix[:, c0:c1 + 1], axis=1, dtype=np.float64)
            else:
                raw_w += np.sum(matrix[c0:c1 + 1, :], axis=0, dtype=np.float64)
            width = c1 - c0 + 1
            total_w_ch += width
            valid_w.append([c0, c1])

    valid_b = []
    total_b_ch = 0
    for b in b_gates:
        c0 = max(0, min(int(round(float(b[0]))), int(round(float(b[1])))))
        c1 = min(max_gate_ch, max(int(round(float(b[0]))), int(round(float(b[1])))))
        if c1 >= c0:
            if axis == 0:
                raw_b += np.sum(matrix[:, c0:c1 + 1], axis=1, dtype=np.float64)
            else:
                raw_b += np.sum(matrix[c0:c1 + 1, :], axis=0, dtype=np.float64)
            width = c1 - c0 + 1
            total_b_ch += width
            valid_b.append([c0, c1])

    if total_b_ch > 0 and total_w_ch > 0:
        scale = float(total_w_ch) / float(total_b_ch)
        bg_sub = raw_b * scale
        net_spec = raw_w - bg_sub
    else:
        scale = 0.0
        bg_sub = np.zeros_like(raw_w)
        net_spec = raw_w.copy()

    gross_counts = float(np.sum(raw_w))
    bg_counts = float(np.sum(bg_sub))
    net_counts = float(np.sum(net_spec))
    net_err = float(np.sqrt(np.maximum(0.0, np.sum(raw_w + (scale ** 2) * raw_b))))

    return {
        "success": True,
        "axis": axis,
        "dest_axis": 1 - axis,
        "valid_w": valid_w,
        "valid_b": valid_b,
        "w_gates": valid_w,
        "b_gates": valid_b,
        "total_w_ch": total_w_ch,
        "total_b_ch": total_b_ch,
        "w_width": total_w_ch,
        "b_width": total_b_ch,
        "scale": scale,
        "gross_counts": gross_counts,
        "bg_counts": bg_counts,
        "net_counts": net_counts,
        "net_err": net_err,
        "net_spec": net_spec.tolist(),
        "raw_spec": raw_w.tolist(),
        "bg_spec": bg_sub.tolist(),
    }


def print_gate_terminal_report(gate_res: dict, matrix_name: str):
    src_det = "Det 1 (X)" if gate_res["axis"] == 0 else "Det 2 (Y)"
    dst_det = "Det 2 (Y)" if gate_res["axis"] == 0 else "Det 1 (X)"
    w_strs = [f"[{w[0]}..{w[1]}]" for w in gate_res["valid_w"]] if gate_res["valid_w"] else ["None"]
    b_strs = [f"[{b[0]}..{b[1]}]" for b in gate_res["valid_b"]] if gate_res["valid_b"] else ["None"]
    w_text = ", ".join(w_strs)
    b_text = ", ".join(b_strs)
    scale_text = f"{gate_res['scale']:.4f}" if gate_res["total_b_ch"] > 0 else "0.0000"

    print(f"\n[1D Gate Cut] {matrix_name} -> Gated {src_det} => {dst_det} Coincidence Spectrum:")
    print(f"  • Gate W: {w_text} (total width: {gate_res['total_w_ch']} ch)")
    print(f"  • Bg B:   {b_text} (total width: {gate_res['total_b_ch']} ch, scale: {scale_text})")
    print(f"  • Counts: Gross={gate_res['gross_counts']:,.0f} | Bg={gate_res['bg_counts']:,.1f} | Net={gate_res['net_counts']:,.1f} ± {gate_res['net_err']:,.1f} cts\n", flush=True)


def ricker_wavelet(points: int, a: float) -> np.ndarray:
    """Analytical Ricker (Mexican Hat) wavelet for CWT convolution."""
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2.0
    xsq = vec ** 2
    mod = (1.0 - xsq / wsq)
    gauss = np.exp(-xsq / (2.0 * wsq))
    return A * mod * gauss


def find_peaks_1d(
    spec: np.ndarray,
    ch_min: int = 0,
    ch_max: int = None,
    method: str = "cwt",
    widths: np.ndarray = None,
    min_snr: float = 9.0,
    min_counts: float = 10.0,
    fwhm_est: float = 4.0,
    cal: list = None,
) -> dict:
    """
    Modular 1D peak search engine for gamma-ray spectroscopy.
    Supports:
      1. 'cwt' (Default & Recommended): Continuous Wavelet Transform (Ricker wavelet) with local Poisson variance normalization.
      2. 'prominence': Topographic prominence with local Poisson statistical significance, doublet resolution checks, and Non-Maximum Suppression (NMS).
      3. 'mariscotti': GASPware native 5-fold smoothed 2nd difference (trackn.F) with Poisson variance normalization and NMS.
    """
    t0 = time.perf_counter()
    spec = np.asarray(spec, dtype=np.float64)
    tot_len = len(spec)
    c_start = max(0, min(tot_len - 1, int(ch_min)))
    c_end = min(tot_len, max(c_start + 1, int(ch_max if ch_max is not None else tot_len)))

    if tot_len < 5:
        return {
            "success": True,
            "method": method,
            "engine": "none",
            "ch_min": c_start,
            "ch_max": c_end,
            "peaks": [],
            "count": 0,
            "elapsed_ms": 0.0,
        }

    fwhm_val = max(1.5, float(fwhm_est))
    min_dist = max(3, int(round(fwhm_val * 0.8)))
    doublet_dist = max(min_dist + 1, int(round(fwhm_val * 1.4)))

    # 3-point binomial smoothing [0.25, 0.5, 0.25] to suppress single-channel Poisson noise spikes
    sm = np.convolve(spec, [0.25, 0.5, 0.25], mode="same")

    # Expanded search range with margin so boundary peaks have complete local context
    margin = max(16, int(round(fwhm_val * 4.0)))
    search_start = max(2, c_start - margin)
    search_end = min(tot_len - 2, c_end + margin)

    candidates = []
    used_engine = "numpy"

    if method == "mariscotti":
        # GASPware Native trackn.F 5-fold boxcar smoothed 2nd difference with Poisson variance
        m = max(1, int(round(fwhm_val * 0.3)))
        k_size = 65
        IC = np.zeros(k_size, dtype=np.float64)
        mid = 32
        IC[mid - 1] = 1.0; IC[mid] = -2.0; IC[mid + 1] = 1.0
        for _ in range(5):
            IB = np.zeros(k_size, dtype=np.float64)
            for ii in range(mid - 1 - 5 * m, mid + 1):
                if 0 <= ii - m and ii + m < k_size:
                    IB[ii] = np.sum(IC[max(0, ii - m):min(k_size, ii + m + 1)])
            for ii in range(mid + 1):
                IC[ii] = IB[ii]; IC[k_size - 1 - ii] = IB[ii]

        kernel = IC[IC != 0]
        ss = np.convolve(spec, kernel, mode="same")
        ff_sq = np.convolve(np.maximum(0.0, spec), kernel ** 2, mode="same")
        ff = np.sqrt(np.maximum(1e-6, ff_sq))
        poisson_snr = -ss / ff

        for i in range(search_start, search_end):
            if poisson_snr[i] >= min_snr and spec[i] >= min_counts:
                if (sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]) or (spec[i] > spec[i - 1] and spec[i] >= spec[i + 1]):
                    candidates.append({
                        "channel": float(i),
                        "counts": int(round(spec[i])),
                        "snr": float(poisson_snr[i]),
                        "score": float(poisson_snr[i])
                    })

    elif method == "cwt":
        # CWT with local Poisson noise variance across scales (prevents global noise skew)
        sigma_est = max(1.0, fwhm_val / 2.355)
        if widths is None:
            widths = np.linspace(max(1.5, sigma_est * 0.8), min(10.0, sigma_est * 2.0), 6)

        cwt_matrix = np.zeros((len(widths), tot_len), dtype=np.float64)
        var_matrix = np.zeros((len(widths), tot_len), dtype=np.float64)

        for i, w in enumerate(widths):
            pts = min(int(round(6.0 * w)), tot_len)
            if pts % 2 == 0:
                pts += 1
            wav = ricker_wavelet(pts, w)
            wav -= np.mean(wav)
            w_norm = np.sum(np.abs(wav)) or 1.0
            wav /= w_norm
            cwt_matrix[i, :] = np.convolve(spec, wav, mode="same")
            var_matrix[i, :] = np.convolve(np.maximum(0.0, spec), wav ** 2, mode="same")

        snr_matrix = cwt_matrix / np.sqrt(np.maximum(1e-6, var_matrix))
        best_snr = np.max(snr_matrix, axis=0)

        for i in range(search_start, search_end):
            if best_snr[i] >= min_snr and spec[i] >= min_counts:
                if (sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]) or (spec[i] > spec[i - 1] and spec[i] >= spec[i + 1]):
                    candidates.append({
                        "channel": float(i),
                        "counts": int(round(spec[i])),
                        "snr": float(best_snr[i]),
                        "score": float(best_snr[i])
                    })

    else:
        # 'prominence' (Default & Recommended for Gamma Spectra)
        # Topographic prominence with local Poisson noise normalization
        w_rad = max(12, int(round(fwhm_val * 3.5)))
        for i in range(search_start, search_end):
            if spec[i] >= min_counts:
                if (sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]) or (spec[i] > spec[i - 1] and spec[i] >= spec[i + 1]):
                    i_l = max(0, i - w_rad)
                    i_r = min(tot_len, i + w_rad + 1)
                    l_min = spec[i]; l_idx = i - 1
                    while l_idx >= i_l and spec[l_idx] <= spec[i]:
                        if spec[l_idx] < l_min:
                            l_min = spec[l_idx]
                        l_idx -= 1
                    if l_idx >= i_l and spec[l_idx] > spec[i]:
                        l_min = np.min(spec[l_idx:i])

                    r_min = spec[i]; r_idx = i + 1
                    while r_idx < i_r and spec[r_idx] <= spec[i]:
                        if spec[r_idx] < r_min:
                            r_min = spec[r_idx]
                        r_idx += 1
                    if r_idx < i_r and spec[r_idx] > spec[i]:
                        r_min = np.min(spec[i+1:r_idx+1])

                    base = max(l_min, r_min)
                    prom = spec[i] - base
                    snr = prom / np.sqrt(max(1.0, base))

                    if snr >= min_snr and prom >= max(5.0, 1.2 * np.sqrt(spec[i])):
                        candidates.append({
                            "channel": float(i),
                            "counts": int(round(spec[i])),
                            "snr": float(snr),
                            "score": float(prom)
                        })

    # Non-Maximum Suppression (NMS) & Doublet Separation Filter:
    # Solves:
    #   1. Intra-peak noise ripples on top/shoulders of real peaks (< 12% dip suppressed)
    #   2. Resolves true close doublets if valley dip >= 12% of peak height
    candidates.sort(key=lambda c: (c["score"], c["counts"]), reverse=True)
    kept = []
    for c in candidates:
        ch = c["channel"]
        y_c = c["counts"]
        is_sub = False
        for k in kept:
            k_ch = k["channel"]
            k_y = k["counts"]
            dist = abs(ch - k_ch)
            if dist < min_dist:
                is_sub = True
                break
            if dist <= doublet_dist:
                c1 = int(round(min(ch, k_ch)))
                c2 = int(round(max(ch, k_ch)))
                valley = np.min(spec[c1:c2+1])
                min_h = min(y_c, k_y)
                dip = min_h - valley
                if dip < 0.12 * min_h:  # Intra-peak noise ripple
                    is_sub = True
                    break

        if not is_sub:
            # Centroid refinement via 3-point parabolic interpolation
            c_int = int(round(ch))
            if 1 <= c_int < tot_len - 1:
                y0, y1, y2 = spec[c_int - 1], spec[c_int], spec[c_int + 1]
                denom = y0 - 2.0 * y1 + y2
                if denom < 0:
                    delta = 0.5 * (y0 - y2) / denom
                    if -0.6 <= delta <= 0.6:
                        ch += delta

            # Calibrated energy
            if cal and len(cal) >= 2:
                energy = cal[0] + cal[1] * ch + (cal[2] * (ch ** 2) if len(cal) > 2 else 0.0)
            else:
                energy = ch

            kept.append({
                "channel": round(float(ch), 2),
                "energy": round(float(energy), 2),
                "counts": int(round(spec[c_int])),
                "snr": round(float(c["snr"]), 1),
            })

    # Filter to requested viewport [c_start, c_end]
    peaks = [p for p in kept if c_start <= p["channel"] <= c_end]
    peaks.sort(key=lambda p: p["channel"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "success": True,
        "method": method,
        "engine": used_engine,
        "ch_min": c_start,
        "ch_max": c_end,
        "peaks": peaks,
        "count": len(peaks),
        "elapsed_ms": round(elapsed_ms, 2),
    }


def print_peaks_terminal_report(peaks_res: dict, det_name: str, matrix_name: str, is_cal: bool):
    """Print aligned tabular summary of found peaks to the terminal."""
    peaks = peaks_res.get("peaks", [])
    count = len(peaks)
    method_name = peaks_res.get("method", "prominence").capitalize()
    engine = peaks_res.get("engine", "numpy").upper()
    elapsed = peaks_res.get("elapsed_ms", 0.0)
    ch_range = f"[{peaks_res.get('ch_min', 0)}..{peaks_res.get('ch_max', 4095)}]"

    print(f"\n[1D Peak Search] {matrix_name} -> {det_name} (Range: {ch_range}, Method: {method_name}, {engine} engine):", flush=True)
    if count == 0:
        print(f"  No peaks detected above SNR threshold in {elapsed:.1f} ms.\n", flush=True)
        return

    print(f"  Detected {count} candidate peaks in {elapsed:.1f} ms:", flush=True)
    unit = "keV" if is_cal else "ch"
    print(f"  {'#':<4} {'Centroid (ch)':<15} {'Energy (' + unit + ')':<15} {'Counts':<10} {'SNR':<8}", flush=True)
    print(f"  {'-'*4} {'-'*15} {'-'*15} {'-'*10} {'-'*8}", flush=True)

    # Show up to top 20 peaks by count
    top_peaks = sorted(peaks, key=lambda p: p["counts"], reverse=True)[:20]
    # Re-sort top 20 by channel for readability
    top_peaks.sort(key=lambda p: p["channel"])
    for idx, p in enumerate(top_peaks, 1):
        e_str = f"{p['energy']:.2f}"
        print(f"  {idx:<4} {p['channel']:<15.2f} {e_str:<15} {p['counts']:<10d} {p['snr']:<8.1f}", flush=True)

    if count > 20:
        print(f"  ... and {count - 20} more peaks (displayed in Web Viewer).\n", flush=True)
    else:
        print("", flush=True)


def compute_snip_background(
    spectrum: np.ndarray,
    ch_min: int = 0,
    ch_max: int = None,
    fwhm_est: float = 4.0,
    iterations: int = None,
    smoothing: bool = True
) -> np.ndarray:
    """
    Computes the continuous background baseline using the SNIP
    (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm
    with LLS (Log-Log-Square-Root) variance-stabilizing transformation
    (Ryan et al. 1988; Morhác et al. 1997; ROOT TSpectrum::Background).

    Parameters:
      spectrum: 1D array of channel counts
      ch_min, ch_max: Region of interest bounds
      fwhm_est: Expected peak FWHM in channels (used to determine clipping window M)
      iterations: Clipping order M (if None, M = max(3, int(round(1.4 * fwhm_est))))
      smoothing: Apply 3-point binomial smoothing [0.25, 0.5, 0.25] to prevent Poisson noise ripples

    Returns:
      bg: 1D array of background counts for [ch_min..ch_max] of length (ch_max - ch_min + 1)
    """
    tot_len = len(spectrum)
    ch_min = max(0, int(ch_min))
    ch_max = tot_len - 1 if ch_max is None else min(tot_len - 1, int(ch_max))
    if ch_min >= ch_max:
        return np.zeros(max(1, ch_max - ch_min + 1), dtype=np.float64)

    fwhm = max(1.5, float(fwhm_est))
    margin = max(24, int(round(4.0 * fwhm)))
    s_min = max(0, ch_min - margin)
    s_max = min(tot_len - 1, ch_max + margin)

    y_sub = np.asarray(spectrum[s_min:s_max + 1], dtype=np.float64)
    N = len(y_sub)
    if N < 3:
        return y_sub.copy()

    # LLS transformation: z = ln(ln(sqrt(max(0, y) + 1) + 1) + 1)
    z = np.log(np.log(np.sqrt(np.maximum(0.0, y_sub) + 1.0) + 1.0) + 1.0)

    # Clipping window order M
    if iterations is None or iterations <= 0:
        M = max(3, int(round(1.4 * fwhm)))
    else:
        M = int(iterations)
    M = min(M, (N - 1) // 2)

    v = z.copy()
    # Decreasing order clipping (Morhác 1997): p = M down to 1
    for p in range(M, 0, -1):
        v_clipped = 0.5 * (v[:-2*p] + v[2*p:])
        v[p:N-p] = np.minimum(v[p:N-p], v_clipped)

    # Invert LLS transformation: B = (exp(exp(v) - 1) - 1)^2 - 1
    bg = (np.exp(np.exp(v) - 1.0) - 1.0)**2 - 1.0
    bg = np.clip(bg, 0.0, y_sub)

    if smoothing and N >= 5:
        bg_sm = bg.copy()
        bg_sm[1:-1] = 0.25 * bg[:-2] + 0.5 * bg[1:-1] + 0.25 * bg[2:]
        bg = np.clip(bg_sm, 0.0, y_sub)

    offset = ch_min - s_min
    span = ch_max - ch_min + 1
    return bg[offset:offset + span]


def compute_peak_aware_background(
    spectrum: np.ndarray,
    ch_min: int = 0,
    ch_max: int = None,
    peak_channels: list = None,
    fwhm_est: float = 4.0
) -> np.ndarray:
    """
    Computes the continuous background baseline across [ch_min, ch_max] by distinguishing
    between pure continuum regions (where peak search 'P' confirmed no peaks exist) and
    photopeak regions.

    In pure background channels, the baseline fits the central expected value (average)
    of the noise grass using rolling smoothing, completely eliminating the underside under-fit.
    Underneath candidate peak clusters, the baseline connects smoothly between the continuum
    levels on either side of the cluster. Handles positive, zero, and negative count values
    (e.g. coincidence-gated net spectra with background subtraction).
    """
    tot_len = len(spectrum)
    ch_min = max(0, int(ch_min))
    ch_max = tot_len - 1 if ch_max is None else min(tot_len - 1, int(ch_max))
    x_total = np.arange(ch_min, ch_max + 1, dtype=np.float64)
    y_total = np.asarray(spectrum[ch_min:ch_max + 1], dtype=np.float64)
    n_ch = len(x_total)

    if n_ch < 5:
        return y_total.copy()

    fwhm = max(1.5, float(fwhm_est))
    half_w = max(4, int(round(2.2 * fwhm)))

    valid_peaks = sorted([float(p) for p in (peak_channels or []) if ch_min <= p <= ch_max])

    if not valid_peaks:
        # No candidate peaks in window: entire window is pure background!
        # Smooth the entire spectrum to get the central average continuum
        win = max(7, min(n_ch // 2 * 2 + 1, int(round(5.0 * fwhm))))
        if win % 2 == 0:
            win += 1
        pad = win // 2
        padded = np.pad(y_total, pad, mode="edge")
        w_kernel = np.bartlett(win)
        w_kernel /= np.sum(w_kernel)
        return np.convolve(padded, w_kernel, mode="valid")[:n_ch]

    # Cluster peaks separated by <= 3.2 * FWHM
    clusters = []
    c_thresh = 3.2 * fwhm
    for p in valid_peaks:
        if not clusters or p - clusters[-1][-1] > c_thresh:
            clusters.append([p])
        else:
            clusters[-1].append(p)

    raw_rois = []
    for cl in clusters:
        i1 = max(0, int(np.floor(min(cl) - half_w - ch_min)))
        i2 = min(n_ch - 1, int(np.ceil(max(cl) + half_w - ch_min)))
        raw_rois.append([i1, i2])

    raw_rois.sort(key=lambda r: r[0])
    cluster_rois = []
    for r in raw_rois:
        if not cluster_rois or r[0] > cluster_rois[-1][1]:
            cluster_rois.append(r)
        else:
            cluster_rois[-1][1] = max(cluster_rois[-1][1], r[1])

    mask_peaks = np.zeros(n_ch, dtype=bool)
    for i1, i2 in cluster_rois:
        mask_peaks[i1:i2+1] = True

    bg_indices = np.where(~mask_peaks)[0]

    # If almost all channels (> 92%) are peaks (dense peak forest across whole window)
    if len(bg_indices) < 5 or len(bg_indices) < 0.08 * n_ch:
        return compute_snip_background(spectrum, ch_min, ch_max, fwhm)

    # 1. Linear interpolation across peak regions to form peak-removed continuum
    y_continuum = y_total.copy()
    y_continuum[mask_peaks] = np.interp(x_total[mask_peaks], x_total[bg_indices], y_total[bg_indices])

    # 2. Smooth continuum using rolling Bartlett average with window ~ 5 * FWHM
    win = max(7, min(n_ch // 2 * 2 + 1, int(round(5.0 * fwhm))))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(y_continuum, pad, mode="edge")
    w_kernel = np.bartlett(win)
    w_kernel /= np.sum(w_kernel)
    bg_smooth = np.convolve(padded, w_kernel, mode="valid")[:n_ch]

    # 3. For each peak cluster, the baseline under the cluster linearly connects
    # the smoothed continuum level at i1 to the smoothed continuum level at i2
    for i1, i2 in cluster_rois:
        if i2 > i1:
            if i1 == 0:
                bg_smooth[0:i2+1] = bg_smooth[i2]
            elif i2 >= n_ch - 1:
                bg_smooth[i1:n_ch] = bg_smooth[i1]
            else:
                bg_smooth[i1:i2+1] = np.linspace(bg_smooth[i1], bg_smooth[i2], i2 - i1 + 1)

    return bg_smooth


def fit_all_peaks_1d(
    spectrum: np.ndarray,
    ch_min: int,
    ch_max: int,
    peak_channels: list,
    fit_type: str = "gaussian",
    fwhm_est: float = 4.0,
    cal: list = [0.0, 1.0, 0.0],
    fwhm_mult: float = 4.0,
    bg_method: str = "peak_aware",
    snip_iter: int = None,
    region: list = None
) -> dict:
    """
    Fits all candidate peaks in the visible display window [ch_min, ch_max] by distinguishing
    between pure continuum regions (where peak search 'P' confirmed no peaks exist) and photopeaks.

    If an explicit fit region is specified via `region=[r_left, r_right]`, the baseline is strictly
    computed as a straight line by averaging a 1-FWHM window on the left and right outside the region:
      - Left Background: average over [r_left - FWHM, r_left - 1]
      - Right Background: average over [r_right + 1, r_right + FWHM]
      - Baseline: exact linear connecting line across the region and background wings.
    All peaks inside the region are fitted simultaneously as a coupled multiplet on this straight line.
    """
    ch_min = max(0, int(ch_min))
    ch_max = min(len(spectrum) - 1, int(ch_max))
    fwhm_clean = max(1.5, float(fwhm_est))
    region_info = None

    if region is not None and len(region) >= 2:
        r_left = min(float(region[0]), float(region[1]))
        r_right = max(float(region[0]), float(region[1]))
        w_bg = max(1, int(round(fwhm_clean)))

        # Left background: 1 FWHM immediately left outside the region
        c_L0 = max(0, int(np.floor(r_left)) - w_bg)
        c_L1 = max(0, int(np.floor(r_left)) - 1)
        if c_L1 < c_L0:
            c_L1 = c_L0
        y_L_avg = float(np.mean(spectrum[c_L0:c_L1 + 1]))
        x_L_center = (c_L0 + c_L1) / 2.0 + 0.5

        # Right background: 1 FWHM immediately right outside the region
        c_R0 = min(len(spectrum) - 1, int(np.ceil(r_right)) + 1)
        c_R1 = min(len(spectrum) - 1, int(np.ceil(r_right)) + w_bg)
        if c_R1 < c_R0:
            c_R1 = c_R0
        y_R_avg = float(np.mean(spectrum[c_R0:c_R1 + 1]))
        x_R_center = (c_R0 + c_R1) / 2.0 + 0.5

        dx = max(1e-6, x_R_center - x_L_center)
        slope = (y_R_avg - y_L_avg) / dx
        intercept = y_L_avg - slope * x_L_center

        ch_min = c_L0
        ch_max = c_R1
        x_total = np.arange(ch_min, ch_max + 1, dtype=np.int64)
        y_total = np.asarray(spectrum[ch_min:ch_max + 1], dtype=np.float64)
        n_ch = len(x_total)

        x_ch_arr = np.arange(ch_min, ch_max + 1, dtype=np.float64) + 0.5
        bg_baseline = slope * x_ch_arr + intercept
        used_bg_method = "linear_region_average"
        region_info = {
            "r_left": r_left,
            "r_right": r_right,
            "left_avg": round(y_L_avg, 2),
            "right_avg": round(y_R_avg, 2),
            "slope": round(slope, 4),
            "c_L0": c_L0,
            "c_L1": c_L1,
            "c_R0": c_R0,
            "c_R1": c_R1
        }

        # Filter candidate peaks: only peaks within [r_left - 1.0, r_right + 1.0]
        reg_peaks = [float(c) for c in (peak_channels or []) if (r_left - 1.0) <= c <= (r_right + 1.0)]
        if not reg_peaks:
            sub_s = spectrum[int(np.floor(r_left)):int(np.ceil(r_right)) + 1]
            if len(sub_s) > 0:
                apex = int(np.floor(r_left)) + int(np.argmax(sub_s))
                reg_peaks = [float(apex)]
            else:
                reg_peaks = [(r_left + r_right) / 2.0]
        valid_peaks = sorted(reg_peaks)
    else:
        x_total = np.arange(ch_min, ch_max + 1, dtype=np.int64)
        y_total = np.asarray(spectrum[ch_min:ch_max + 1], dtype=np.float64)
        n_ch = len(x_total)

        if n_ch < 5:
            return {"success": False, "error": "Display range too small for peak fitting (< 5 channels)."}

        if peak_channels is None:
            peak_channels = []
        valid_peaks = sorted([float(c) for c in peak_channels if ch_min <= c <= ch_max])
        if not valid_peaks:
            # Fallback to auto-detecting peaks in window if none passed
            search_res = find_peaks_1d(spectrum, ch_min=ch_min, ch_max=ch_max, method="cwt", min_snr=9.0, fwhm_est=fwhm_est, cal=cal)
            valid_peaks = sorted([p["channel"] for p in search_res.get("peaks", []) if ch_min <= p["channel"] <= ch_max])
            if not valid_peaks:
                return {"success": False, "error": f"No candidate peaks found within display range [{ch_min}..{ch_max}]."}

    # Clean candidate peaks: suppress duplicate candidates closer than 0.8 * FWHM without >= 10% dip
    # When explicit peaks were supplied (e.g. user set up peaks with 'G' to fit a multiplet with 'V'),
    # preserve all distinct peaks (>= 0.6 channels apart) so user-marked multiplets are never discarded.
    is_explicit_peaks = bool(peak_channels and len(peak_channels) > 0)
    min_spacing = 0.6 if is_explicit_peaks else (0.8 * fwhm_clean)
    clean_peaks = []
    for p in valid_peaks:
        if not clean_peaks:
            clean_peaks.append(p)
        else:
            prev = clean_peaks[-1]
            if p - prev < min_spacing:
                ch1 = int(np.clip(round(prev), 0, len(spectrum) - 1))
                ch2 = int(np.clip(round(p), 0, len(spectrum) - 1))
                if ch2 > ch1:
                    mid_val = np.min(spectrum[ch1:ch2 + 1])
                    peak_val = min(spectrum[ch1], spectrum[ch2])
                    if mid_val > 0.90 * peak_val and not is_explicit_peaks:
                        if spectrum[ch2] > spectrum[ch1]:
                            clean_peaks[-1] = p
                        continue
            clean_peaks.append(p)

    if region_info is not None:
        clusters = [ clean_peaks ]
    else:
        # Cluster peaks: peaks within threshold are grouped into the same multiplet
        # When explicit multiplet peaks (<= 6 peaks) are passed to be fitted together,
        # expand grouping threshold to 4.5 * FWHM to ensure all components fit as one cluster.
        clusters = []
        c_threshold = (4.5 * fwhm_clean) if (is_explicit_peaks and len(clean_peaks) <= 6) else (3.2 * fwhm_clean)
        for p in clean_peaks:
            if not clusters or p - clusters[-1][-1] > c_threshold:
                clusters.append([p])
            else:
                clusters[-1].append(p)

    # 1. Compute baseline across the display range (if not already computed via region)
    if region_info is None:
        if bg_method == "snip":
            m_iter = max(3, int(round(1.4 * fwhm_clean))) if snip_iter is None else int(snip_iter)
            bg_baseline = compute_snip_background(spectrum, ch_min=ch_min, ch_max=ch_max, fwhm_est=fwhm_clean, iterations=m_iter)
            used_bg_method = "snip"
        else:
            bg_baseline = compute_peak_aware_background(spectrum, ch_min=ch_min, ch_max=ch_max, peak_channels=clean_peaks, fwhm_est=fwhm_clean)
            used_bg_method = "peak_aware_average"

    x_spec = np.arange(len(spectrum), dtype=np.float64) + 0.5
    y_spec = np.asarray(spectrum, dtype=np.float64)

    a0, a1, a2 = (cal if cal and len(cal) == 3 else [0.0, 1.0, 0.0])
    def ch_to_e(c): return a0 + a1 * c + a2 * (c**2)
    def de_dch(c): return abs(a1 + 2.0 * a2 * c)

    all_fitted_peaks = []
    cluster_results = []
    is_hypermet = (fit_type == "hypermet")
    is_tail = (fit_type == "gaussian_tail")

    sigma_init = max(0.5, fwhm_clean / 2.355)
    half_w = max(4, int(round(max(1.8, 0.5 * fwhm_mult) * fwhm_clean)))

    for cluster in clusters:
        k = len(cluster)
        if region_info is not None:
            i_min = ch_min
            i_max = ch_max
        else:
            i_min = max(ch_min, int(np.floor(min(cluster) - half_w)))
            i_max = min(ch_max, int(np.ceil(max(cluster) + half_w)))

        roi_x = x_spec[i_min:i_max + 1]
        roi_y = y_spec[i_min:i_max + 1]
        roi_bg = bg_baseline[i_min - ch_min:i_max - ch_min + 1]
        N = len(roi_x)
        if N < 3:
            continue

        weights = 1.0 / np.sqrt(np.maximum(1.0, np.abs(roi_y)))

        if is_hypermet:
            p0 = []
            for p_ch in cluster:
                idx = int(np.clip(round(p_ch - roi_x[0]), 0, N - 1))
                h_init = max(10.0, float(roi_y[idx]) - roi_bg[idx])
                fT_init = 0.15 * (h_init * sigma_init * math.sqrt(2.0 * math.pi))
                beta_init = 1.5 * sigma_init
                AS_init = 0.02 * h_init
                p0.extend([h_init, float(p_ch), sigma_init, fT_init, beta_init, AS_init])
            n_per_peak = 6
        elif is_tail:
            p0 = []
            for p_ch in cluster:
                idx = int(np.clip(round(p_ch - roi_x[0]), 0, N - 1))
                h_init = max(10.0, float(roi_y[idx]) - roi_bg[idx])
                p0.extend([h_init, float(p_ch), sigma_init, 1.5])
            n_per_peak = 4
        else:  # Standard symmetric Gaussian
            p0 = []
            for p_ch in cluster:
                idx = int(np.clip(round(p_ch - roi_x[0]), 0, N - 1))
                h_init = max(10.0, float(roi_y[idx]) - roi_bg[idx])
                p0.extend([h_init, float(p_ch), sigma_init])
            n_per_peak = 3

        p_curr = np.array(p0, dtype=np.float64)
        lam = 0.001

        def model(p):
            tot = roi_bg.copy()
            for j in range(k):
                base = n_per_peak * j
                if is_hypermet:
                    H = max(0.0, p[base])
                    mu = p[base + 1]
                    sig = max(0.2, abs(p[base + 2]))
                    fT = max(0.0, p[base + 3])
                    beta = max(0.2, p[base + 4])
                    AS = max(0.0, p[base + 5])
                    dx = roi_x - mu
                    z = dx / sig
                    g = H * np.exp(np.clip(-0.5 * z**2, -50.0, 0.0))
                    u = np.clip(dx / beta + 0.5 * (sig / beta)**2, -50.0, 50.0)
                    v = np.clip(dx / (math.sqrt(2.0) * sig) + sig / (math.sqrt(2.0) * beta), -20.0, 20.0)
                    t = (fT / (2.0 * beta)) * np.exp(u) * _vec_erfc(v)
                    s = 0.5 * AS * _vec_erfc(np.clip(z / math.sqrt(2.0), -20.0, 20.0))
                    s_loc = np.where((roi_x >= i_min) & (roi_x <= mu + 4.0 * sig), s, 0.0)
                    tot += (g + t + s_loc)
                elif is_tail:
                    H = max(0.0, p[base])
                    mu = p[base + 1]
                    sig = max(0.2, abs(p[base + 2]))
                    alpha = max(0.3, p[base + 3])
                    dx = roi_x - mu
                    z = dx / sig
                    is_g = (z >= -alpha)
                    g_val = H * np.exp(-0.5 * z**2)
                    t_val = H * np.exp(np.clip(0.5 * alpha**2 + alpha * z, -50.0, 50.0))
                    tail_win = (roi_x >= mu - 15.0 * sig) & (roi_x <= mu + 6.0 * sig)
                    pk_val = np.where(is_g, g_val, t_val)
                    tot += np.where(tail_win, pk_val, 0.0)
                else:
                    H = max(0.0, p[base])
                    mu = p[base + 1]
                    sig = max(0.2, abs(p[base + 2]))
                    dx = roi_x - mu
                    tot += H * np.exp(np.clip(-0.5 * (dx / sig)**2, -50.0, 0.0))
            return tot

        for it in range(80):
            mod = model(p_curr)
            r_curr = (mod - roi_y) * weights
            J = np.zeros((N, len(p_curr)), dtype=np.float64)
            for idx_p in range(len(p_curr)):
                dp = max(1e-5, abs(p_curr[idx_p]) * 1e-4)
                p_up = p_curr.copy()
                p_up[idx_p] += dp
                J[:, idx_p] = (model(p_up) - mod) / dp * weights

            JTJ = J.T @ J
            JTr = J.T @ r_curr
            A_mat = JTJ + lam * np.diag(np.diag(JTJ) + 1e-4)
            try:
                delta = np.linalg.solve(A_mat, -JTr)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(A_mat) @ (-JTr)

            p_next = p_curr + delta
            for j in range(k):
                base = n_per_peak * j
                p_next[base] = max(0.0, p_next[base])
                p_next[base + 1] = float(np.clip(p_next[base + 1], cluster[j] - 2.5, cluster[j] + 2.5))
                p_next[base + 2] = max(0.3, abs(p_next[base + 2]))
                if is_hypermet:
                    p_next[base + 3] = max(0.0, p_next[base + 3])
                    p_next[base + 4] = max(0.3, p_next[base + 4])
                    p_next[base + 5] = max(0.0, p_next[base + 5])
                elif is_tail:
                    p_next[base + 3] = max(0.3, p_next[base + 3])

            r_next = (model(p_next) - roi_y) * weights
            if np.sum(r_next**2) < np.sum(r_curr**2):
                p_curr = p_next
                lam = max(1e-7, lam * 0.3)
                if np.max(np.abs(delta)) < 1e-4:
                    break
            else:
                lam = min(1e4, lam * 3.0)

        fit_cluster = model(p_curr)
        res_cluster = (fit_cluster - roi_y) * weights
        chi2_c = float(np.sum(res_cluster**2))

        # Parameter Covariance Matrix
        J = np.zeros((N, len(p_curr)), dtype=np.float64)
        for idx_p in range(len(p_curr)):
            dp = max(1e-5, abs(p_curr[idx_p]) * 1e-4)
            p_up = p_curr.copy()
            p_up[idx_p] += dp
            J[:, idx_p] = (model(p_up) - fit_cluster) / dp * weights

        JTJ = J.T @ J
        ndf_c = max(1, N - len(p_curr))
        s2_c = chi2_c / ndf_c
        try:
            cov_c = np.linalg.inv(JTJ) * s2_c
        except np.linalg.LinAlgError:
            cov_c = np.linalg.pinv(JTJ) * s2_c

        c_peaks = []
        for j in range(k):
            base = n_per_peak * j
            H_fit = float(p_curr[base])
            mu_fit = float(p_curr[base + 1])
            sig_fit = float(abs(p_curr[base + 2]))

            var_H = max(0.0, float(cov_c[base, base]))
            var_mu = max(0.0, float(cov_c[base + 1, base + 1]))
            var_sig = max(0.0, float(cov_c[base + 2, base + 2]))
            cov_H_sig = float(cov_c[base, base + 2])

            H_err = math.sqrt(var_H)
            mu_err = math.sqrt(var_mu)
            sig_err = math.sqrt(var_sig)

            fwhm_fit = sig_fit * 2.354820045
            fwhm_err = sig_err * 2.354820045

            if is_hypermet:
                fT_fit = float(p_curr[base + 3])
                AS_fit = float(p_curr[base + 5])
                net_area = H_fit * sig_fit * math.sqrt(2.0 * math.pi) + fT_fit + AS_fit * (sig_fit * math.sqrt(2.0 * math.pi))
                area_err = math.sqrt(max(1.0, 2.0 * math.pi * (sig_fit**2 * var_H + H_fit**2 * var_sig + 2.0 * H_fit * sig_fit * cov_H_sig)))
            elif is_tail:
                alpha_fit = float(p_curr[base + 3])
                A_tail = (1.0 / alpha_fit) * math.exp(0.5 * alpha_fit**2) * _vec_erfc(alpha_fit / math.sqrt(2.0))
                factor = math.sqrt(2.0 * math.pi) * (1.0 - 0.5 * _vec_erfc(alpha_fit / math.sqrt(2.0))) + A_tail
                net_area = H_fit * sig_fit * factor
                area_err = math.sqrt(max(1.0, factor**2 * (sig_fit**2 * var_H + H_fit**2 * var_sig + 2.0 * H_fit * sig_fit * cov_H_sig)))
            else:
                net_area = H_fit * sig_fit * math.sqrt(2.0 * math.pi)
                var_area = 2.0 * math.pi * (sig_fit**2 * var_H + H_fit**2 * var_sig + 2.0 * H_fit * sig_fit * cov_H_sig)
                area_err = math.sqrt(max(1.0, var_area))

            # Centroid and FWHM in energy
            e_c = ch_to_e(mu_fit)
            disp_c = de_dch(mu_fit)
            e_err = mu_err * disp_c
            fwhm_e = fwhm_fit * disp_c
            fwhm_e_err = fwhm_err * disp_c

            # Peak-to-background ratio
            idx_ch = int(np.clip(round(mu_fit - ch_min), 0, len(bg_baseline) - 1))
            bg_val = float(bg_baseline[idx_ch])
            p_to_bg = (H_fit / max(1.0, bg_val)) if bg_val > 0 else 999.0

            c_peaks.append({
                "cluster_idx": len(cluster_results),
                "peak_idx": j,
                "centroid_ch": round(mu_fit, 3),
                "centroid_ch_err": round(mu_err, 3),
                "centroid_e": round(e_c, 2),
                "centroid_e_err": round(e_err, 2),
                "fwhm_ch": round(fwhm_fit, 3),
                "fwhm_ch_err": round(fwhm_err, 3),
                "fwhm_e": round(fwhm_e, 2),
                "fwhm_e_err": round(fwhm_e_err, 2),
                "area": round(net_area, 1),
                "area_err": round(area_err, 1),
                "amplitude": round(H_fit, 1),
                "amplitude_err": round(H_err, 1),
                "bg_level": round(bg_val, 1),
                "peak_to_bg": round(p_to_bg, 2)
            })
            all_fitted_peaks.append(c_peaks[-1])

        cluster_results.append({
            "roi_ch_min": i_min,
            "roi_ch_max": i_max,
            "p_opt": p_curr.copy(),
            "k": k,
            "fit": fit_cluster,
            "peaks": c_peaks,
            "chi2": chi2_c
        })

    all_fitted_peaks.sort(key=lambda p: p["centroid_ch"])

    # Overall reduced chi-squared over fitted cluster regions
    curve_fit_bins = bg_baseline.copy()
    filled_mask = np.zeros(n_ch, dtype=bool)
    for cl in cluster_results:
        s = cl["roi_ch_min"] - ch_min
        e = cl["roi_ch_max"] - ch_min
        curve_fit_bins[s:e+1] = cl["fit"]
        filled_mask[s:e+1] = True

    known_idx = np.where(filled_mask)[0]
    if len(known_idx) > 0:
        res_diff = y_total[known_idx] - curve_fit_bins[known_idx]
        var_y = np.maximum(1.0, np.abs(y_total[known_idx]))
        tot_params = sum([len(cl["peaks"]) * (4 if is_tail else (6 if is_hypermet else 3)) for cl in cluster_results])
        ndf_tot = max(1, len(known_idx) - tot_params)
        red_chi2 = round(float(np.sum((res_diff**2) / var_y) / ndf_tot), 2)
    else:
        red_chi2 = 1.0
        ndf_tot = 1

    # Continuous dense grid evaluation for smooth drawing and exact bin-center alignment
    n_span = ch_max - ch_min + 1
    n_pts = min(3000, max(600, n_span * 10))
    x_dense = np.linspace(float(ch_min), float(ch_max + 1.0), n_pts)
    x_bins = np.arange(ch_min, ch_max + 1, dtype=np.float64) + 0.5
    bg_dense = np.interp(x_dense, x_bins, bg_baseline)
    fit_dense = bg_dense.copy()

    # Add continuous analytical peak profiles on top of the baseline
    for cl in cluster_results:
        p_c = cl["p_opt"]
        k = cl["k"]
        for j in range(k):
            base = n_per_peak * j
            if is_hypermet:
                H = max(0.0, p_c[base])
                mu = p_c[base + 1]
                sig = max(0.2, abs(p_c[base + 2]))
                fT = max(0.0, p_c[base + 3])
                beta = max(0.2, p_c[base + 4])
                AS = max(0.0, p_c[base + 5])
                dx = x_dense - mu
                z = dx / sig
                g = H * np.exp(np.clip(-0.5 * z**2, -50.0, 0.0))
                u = np.clip(dx / beta + 0.5 * (sig / beta)**2, -50.0, 50.0)
                v = np.clip(dx / (math.sqrt(2.0) * sig) + sig / (math.sqrt(2.0) * beta), -20.0, 20.0)
                t = (fT / (2.0 * beta)) * np.exp(u) * _vec_erfc(v)
                s = 0.5 * AS * _vec_erfc(np.clip(z / math.sqrt(2.0), -20.0, 20.0))
                s_loc = np.where((x_dense >= cl["roi_ch_min"]) & (x_dense <= mu + 4.0 * sig), s, 0.0)
                fit_dense += (g + t + s_loc)
            elif is_tail:
                H = max(0.0, p_c[base])
                mu = p_c[base + 1]
                sig = max(0.2, abs(p_c[base + 2]))
                alpha = max(0.3, p_c[base + 3])
                dx = x_dense - mu
                z = dx / sig
                is_g = (z >= -alpha)
                g_val = H * np.exp(-0.5 * z**2)
                t_val = H * np.exp(np.clip(0.5 * alpha**2 + alpha * z, -50.0, 50.0))
                tail_win = (x_dense >= mu - 15.0 * sig) & (x_dense <= mu + 6.0 * sig)
                pk_val = np.where(is_g, g_val, t_val)
                fit_dense += np.where(tail_win, pk_val, 0.0)
            else:
                H = max(0.0, p_c[base])
                mu = p_c[base + 1]
                sig = max(0.2, abs(p_c[base + 2]))
                dx = x_dense - mu
                fit_dense += H * np.exp(np.clip(-0.5 * (dx / sig)**2, -50.0, 0.0))

    ret = {
        "success": True,
        "bg_method": used_bg_method,
        "fit_type": fit_type,
        "ch_min": ch_min,
        "ch_max": ch_max,
        "total_peaks": len(all_fitted_peaks),
        "peaks_fitted": len(all_fitted_peaks),
        "total_clusters": len(clusters),
        "spline_bg_knots": len(clusters),
        "red_chi2": red_chi2,
        "ndf": ndf_tot,
        "curve_x": [round(float(v), 3) for v in x_dense],
        "curve_bg": [round(float(v), 2) for v in bg_dense],
        "curve_fit": [round(float(v), 2) for v in fit_dense],
        "peaks": all_fitted_peaks
    }
    if region_info is not None:
        ret["region_info"] = region_info
    return ret


def print_multi_fit_terminal_report(res: dict, det_name: str, matrix_name: str, is_cal: bool):
    """Print aligned tabular summary of all fitted peaks to the terminal."""
    peaks = res.get("peaks", [])
    count = len(peaks)
    ft = res.get("fit_type", "gaussian")
    model_name = "Hypermet" if ft == "hypermet" else ("RadWare (Left Tail)" if ft == "gaussian_tail" else "Gaussian (Symmetric)")
    ch_min = res.get("ch_min", 0)
    ch_max = res.get("ch_max", 4095)
    clusters = res.get("total_clusters", 1)
    bg_method = res.get("bg_method", "peak_aware_average")
    r_info = res.get("region_info")
    if bg_method == "linear_region_average" and r_info:
        bg_name = f"Straight-Line Region Average [ch {r_info['r_left']:.1f}..{r_info['r_right']:.1f}]"
    elif bg_method == "snip":
        bg_name = f"SNIP Background (M={res.get('snip_iterations', 6)})"
    else:
        bg_name = "Peak-Aware Continuum Average"

    bar = "═" * 96
    subbar = "─" * 96
    print(f"\n{bar}", flush=True)
    print(f"[GASPware 1D Multi-Peak Fit] [{det_name}] - {matrix_name}", flush=True)
    print(f"Model: {model_name} | {bg_name} ({clusters} clusters) | Display Window: ch [{ch_min}..{ch_max}]", flush=True)
    if r_info:
        print(f"Region: [ch {r_info['r_left']:.1f}..{r_info['r_right']:.1f}] | Background: Left Avg = {r_info['left_avg']:.1f} (ch {r_info['c_L0']}..{r_info['c_L1']}) | Right Avg = {r_info['right_avg']:.1f} (ch {r_info['c_R0']}..{r_info['c_R1']}) | Slope = {r_info['slope']:.4f}", flush=True)
    print(subbar, flush=True)
    if count == 0:
        print("  No peaks fitted.", flush=True)
        print(f"{bar}\n", flush=True)
        return

    unit = "keV" if is_cal else "ch"
    print(f"  {'#':<4} {'Centroid (' + unit + ')':<22} {'FWHM (' + unit + ')':<18} {'Net Area (counts)':<24} {'Peak/BG':<8}", flush=True)
    print(f"  {'-'*4} {'-'*22} {'-'*18} {'-'*24} {'-'*8}", flush=True)

    tot_area = 0.0
    for idx, p in enumerate(peaks, 1):
        if is_cal:
            c_str = f"{p['centroid_e']:8.2f} ± {p['centroid_e_err']:<5.2f}"
            w_str = f"{p['fwhm_e']:6.2f} ± {p['fwhm_e_err']:<5.2f}"
        else:
            c_str = f"{p['centroid_ch']:8.3f} ± {p['centroid_ch_err']:<5.3f}"
            w_str = f"{p['fwhm_ch']:6.3f} ± {p['fwhm_ch_err']:<5.3f}"
        a_str = f"{p['area']:10.1f} ± {p['area_err']:<7.1f}"
        pbg_str = f"{p['peak_to_bg']:6.2f}"
        tot_area += p["area"]
        print(f"  {idx:<4} {c_str:<22} {w_str:<18} {a_str:<24} {pbg_str:<8}", flush=True)

    print(subbar, flush=True)
    print(f"  Total Peaks Fitted: {count} | Total Net Area: {tot_area:,.1f} counts | χ²_red = {res.get('red_chi2', 1.0)}", flush=True)
    print(f"{bar}\n", flush=True)


def parse_cmd_tokens(tokens: list) -> tuple:
    """
    Parse a list of string tokens into (positional_arguments, options_dict).
    Supports boolean flags (--fit), single-value flags (--roi 16), and multi-value flags (--range 100 200).
    """
    pos = []
    flags = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") or (tok.startswith("-") and len(tok) == 2 and not (len(tok) > 1 and tok[1].isdigit())):
            flag_name = tok.lstrip("-").replace("-", "_").lower()
            vals = []
            i += 1
            while i < len(tokens):
                next_tok = tokens[i]
                if next_tok.startswith("--") or (next_tok.startswith("-") and len(next_tok) == 2 and not (len(next_tok) > 1 and next_tok[1].isdigit())):
                    break
                vals.append(next_tok)
                i += 1
            if len(vals) == 0:
                flags[flag_name] = True
            elif len(vals) == 1:
                flags[flag_name] = vals[0]
            else:
                flags[flag_name] = vals
        else:
            pos.append(tok)
            i += 1
    return pos, flags


def parse_gate_args(tokens: list, session: "CMATSession"):
    """Parse coincidence gate command tokens into (action, axis, w_gates, b_gates)."""
    if not tokens:
        return "show", None, [], []
    first = tokens[0].lower()
    if first in ("clear", "reset", "ungate"):
        axis = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else None
        return "clear", axis, [], []
    if first in ("show", "info"):
        return "show", None, [], []

    try:
        axis = int(tokens[0])
    except ValueError:
        axis = 0

    w_gates = []
    b_gates = []
    current_type = None
    curr_pair = []
    is_energy = False

    for tok in tokens[1:]:
        t_low = tok.lower()
        if t_low in ("--energy", "-e"):
            is_energy = True
            continue
        if t_low in ("--channel", "--ch", "-c"):
            is_energy = False
            continue
        if t_low in ("w", "gate", "peak", "roi"):
            if len(curr_pair) == 2:
                (w_gates if current_type == "w" else b_gates).append(curr_pair)
                curr_pair = []
            current_type = "w"
            continue
        if t_low in ("b", "bg", "background"):
            if len(curr_pair) == 2:
                (w_gates if current_type == "w" else b_gates).append(curr_pair)
                curr_pair = []
            current_type = "b"
            continue
        try:
            val = float(tok)
            curr_pair.append(val)
            if len(curr_pair) == 2:
                if current_type == "w":
                    w_gates.append(curr_pair)
                elif current_type == "b":
                    b_gates.append(curr_pair)
                curr_pair = []
        except ValueError:
            pass

    if len(curr_pair) == 2 and current_type:
        (w_gates if current_type == "w" else b_gates).append(curr_pair)

    if is_energy:
        cal = session.get_cal(axis)
        w_gates = [[energy_to_ch(p[0], cal), energy_to_ch(p[1], cal)] for p in w_gates]
        b_gates = [[energy_to_ch(p[0], cal), energy_to_ch(p[1], cal)] for p in b_gates]

    return "apply", axis, w_gates, b_gates


def parse_cal_args(tokens: list):
    """Parse calibration command tokens into (action, axis, coeffs)."""
    if not tokens or tokens[0].lower() in ("show", "info"):
        axis = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else None
        return "show", axis, None
    if tokens[0].lower() in ("clear", "reset"):
        axis = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else None
        return "clear", axis, None

    first = tokens[0].lower()
    if first.isdigit():
        axis = int(first)
        coeffs = [float(x) for x in tokens[1:]]
        return "set", axis, coeffs
    elif first in ("x", "det1", "det_1"):
        axis = 0
        coeffs = [float(x) for x in tokens[1:]]
        return "set", axis, coeffs
    elif first in ("y", "det2", "det_2"):
        axis = 1
        coeffs = [float(x) for x in tokens[1:]]
        return "set", axis, coeffs
    else:
        coeffs = [float(x) for x in tokens]
        return "set", None, coeffs


class CMATSession:
    """
    Central state container for python-cmat spectroscopy sessions.
    Maintains loaded matrices, per-axis calibration parameters, active gates,
    1D and 2D fits, and user preferences.
    """
    def __init__(self, config: dict = None, config_path: Path = None):
        self.matrices = []
        self.active_index = 0
        self.config = config.copy() if config else DEFAULT_CONFIG.copy()
        self.config_path = config_path

        c0 = parse_cal_coefficients(self.config.get("cal_0", self.config.get("cal", [0.0, 1.0, 0.0])))
        c1 = parse_cal_coefficients(self.config.get("cal_1", self.config.get("cal", [0.0, 1.0, 0.0])))
        self.cal = {0: c0, 1: c1}
        self.gates = {0: None, 1: None}
        self.fits_1d = {0: None, 1: None}
        self.fit_2d = None

    def add_matrix_file(self, path: Path, name: str = None, cal: dict = None) -> int:
        path = Path(path).resolve()
        for idx, m in enumerate(self.matrices):
            if m["path"] == str(path):
                if name:
                    m["name"] = name
                return idx

        reader = CMATReader(path)
        mat = reader.to_numpy()
        proj = reader.get_projection()
        matrix_name = name or path.name

        matrix_cal = {0: list(self.cal[0]), 1: list(self.cal[1])}
        if cal:
            if isinstance(cal, dict):
                for k, v in cal.items():
                    matrix_cal[int(k)] = parse_cal_coefficients(v)
            elif isinstance(cal, (list, tuple)):
                if len(cal) == 6:
                    matrix_cal[0] = parse_cal_coefficients(cal[:3])
                    matrix_cal[1] = parse_cal_coefficients(cal[3:])
                elif len(cal) >= 3:
                    matrix_cal[0] = parse_cal_coefficients(cal[:3])
                    matrix_cal[1] = parse_cal_coefficients(cal[:3])

        entry = {
            "index": len(self.matrices),
            "name": matrix_name,
            "filename": path.name,
            "path": str(path),
            "reader": reader,
            "matrix": mat,
            "proj": proj,
            "shape": list(mat.shape),
            "total_counts": int(np.sum(mat)),
            "max_count": int(np.max(mat)),
            "nonzero_bins": int(np.count_nonzero(mat)),
            "is_symmetric": bool(reader.is_symmetric),
            "cal": matrix_cal,
        }
        self.matrices.append(entry)
        return len(self.matrices) - 1

    def get_active_matrix(self) -> dict:
        if self.matrices and 0 <= self.active_index < len(self.matrices):
            return self.matrices[self.active_index]
        return None

    def select_matrix(self, identifier) -> int:
        if not self.matrices:
            raise ValueError("No matrices loaded in session.")
        try:
            val = int(identifier)
            if 0 <= val < len(self.matrices):
                self.active_index = val
                return val
            elif 1 <= val <= len(self.matrices):
                self.active_index = val - 1
                return val - 1
        except ValueError:
            pass

        ident_str = str(identifier).strip().lower()
        for idx, m in enumerate(self.matrices):
            if m["name"].lower() == ident_str or m["filename"].lower() == ident_str:
                self.active_index = idx
                return idx
        for idx, m in enumerate(self.matrices):
            if ident_str in m["name"].lower() or ident_str in m["filename"].lower():
                self.active_index = idx
                return idx
        raise KeyError(f"Matrix '{identifier}' not found in loaded matrices.")

    def close_matrix(self, identifier=None) -> None:
        if not self.matrices:
            return
        idx = self.active_index if identifier is None else self.select_matrix(identifier)
        self.matrices.pop(idx)
        for i, m in enumerate(self.matrices):
            m["index"] = i
        if self.active_index >= len(self.matrices):
            self.active_index = max(0, len(self.matrices) - 1)

    def get_cal(self, axis: int = 0) -> list:
        axis = int(axis)
        m = self.get_active_matrix()
        if m and "cal" in m:
            mc = m["cal"]
            if isinstance(mc, dict) and axis in mc:
                return mc[axis]
            elif isinstance(mc, (list, tuple)):
                if len(mc) > 0 and isinstance(mc[0], (list, tuple)):
                    return mc[axis] if axis < len(mc) else [0.0, 1.0, 0.0]
                return list(mc)
        return self.cal.get(axis, [0.0, 1.0, 0.0])

    def set_cal(self, axis, coeffs: list) -> None:
        c = parse_cal_coefficients(coeffs)
        if axis is None:
            self.cal[0] = list(c)
            self.cal[1] = list(c)
            m = self.get_active_matrix()
            if m and "cal" in m and isinstance(m["cal"], dict):
                m["cal"][0] = list(c)
                m["cal"][1] = list(c)
        else:
            axis = int(axis)
            self.cal[axis] = list(c)
            m = self.get_active_matrix()
            if m and "cal" in m and isinstance(m["cal"], dict):
                m["cal"][axis] = list(c)

    def is_calibrated(self, axis: int = 0) -> bool:
        return is_calibrated_coeffs(self.get_cal(axis))

    def get_spectrum(self, axis: int = 0) -> np.ndarray:
        m = self.get_active_matrix()
        if not m:
            return np.array([], dtype=np.float64)
        other_axis = 1 - int(axis)
        gate = self.gates.get(other_axis)
        if gate and "net_spec" in gate:
            return np.array(gate["net_spec"], dtype=np.float64)
        mat = m["matrix"]
        if axis == 0:
            if m["is_symmetric"] and m["proj"] is not None:
                return m["proj"]
            return np.sum(mat, axis=0, dtype=np.float64)
        else:
            if m["is_symmetric"] and m["proj"] is not None:
                return m["proj"]
            return np.sum(mat, axis=1, dtype=np.float64)


class CMATCommandInterpreter:
    """
    Command-line and macro script execution engine for python-cmat.
    Executes commands for CMATSession in headless mode (-m, -c, or -i).
    """
    def __init__(self, session: CMATSession):
        self.session = session
        self.macro_depth = 0
        self.max_macro_depth = 10
        self.last_gated_axis = None
        self.should_exit = False

    def execute_batch(self, cmd_string: str) -> bool:
        """Execute one or more commands (may contain semicolon-separated commands)."""
        self.should_exit = False
        res = self.execute_line(cmd_string)
        return True if self.should_exit else res

    def execute_line(self, line: str) -> bool:
        """Execute a single line of commands (may contain semicolon-separated commands)."""
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            return True

        parts = []
        try:
            in_quote = None
            cur = []
            for ch in line:
                if ch in ('"', "'"):
                    if in_quote == ch:
                        in_quote = None
                    elif in_quote is None:
                        in_quote = ch
                    cur.append(ch)
                elif ch == ';' and in_quote is None:
                    part_str = "".join(cur).strip()
                    if part_str:
                        parts.append(part_str)
                    cur = []
                else:
                    cur.append(ch)
            part_str = "".join(cur).strip()
            if part_str:
                parts.append(part_str)
        except Exception:
            parts = [line]

        for part in parts:
            if not self._execute_single_command(part):
                return False
        return True

    def _execute_single_command(self, cmd_str: str) -> bool:
        cmd_str = cmd_str.strip()
        if not cmd_str or cmd_str.startswith("#") or cmd_str.startswith("//"):
            return True

        import shlex
        try:
            tokens = shlex.split(cmd_str)
        except Exception as e:
            print(f"[!] Syntax error in command '{cmd_str}': {e}", file=sys.stderr)
            return True

        if not tokens:
            return True

        verb = tokens[0].lower()
        args = tokens[1:]

        if verb in ("quit", "exit", "q"):
            self.should_exit = True
            return False
        elif verb in ("help", "?"):
            self.cmd_help(args)
        elif verb in ("echo", "print"):
            self.cmd_echo(args)
        elif verb in ("sleep", "wait"):
            self.cmd_sleep(args)
        elif verb in ("load", "open"):
            self.cmd_load(args)
        elif verb in ("matrix", "select", "use"):
            self.cmd_matrix(args)
        elif verb in ("list", "matrices", "ls"):
            self.cmd_list(args)
        elif verb in ("info", "metadata"):
            self.cmd_info(args)
        elif verb in ("close", "unload"):
            self.cmd_close(args)
        elif verb == "cal":
            self.cmd_cal(args)
        elif verb in ("gate", "ungate"):
            self.cmd_gate(args if verb == "gate" else ["clear"] + args)
        elif verb in ("search", "find"):
            self.cmd_search(args)
        elif verb == "fit_1d":
            self.cmd_fit_1d(args)
        elif verb in ("fit_multiplet", "fit_multi"):
            self.cmd_fit_multiplet(args)
        elif verb in ("fit_all", "fitall"):
            self.cmd_fit_all(args)
        elif verb in ("clear_fits", "clearfit"):
            self.cmd_clear_fits(args)
        elif verb in ("fit_2d", "fit2d"):
            self.cmd_fit_2d(args)
        elif verb in ("pdf_1d", "pdf1d"):
            self.cmd_pdf_1d(args)
        elif verb in ("pdf_2d", "pdf2d"):
            self.cmd_pdf_2d(args)
        elif verb in ("export_1d", "export1d"):
            self.cmd_export_1d(args)
        elif verb in ("export_amat", "export_mat", "exportamat"):
            self.cmd_export_amat(args)
        elif verb in ("macro", "run") or verb.startswith("@"):
            if verb.startswith("@"):
                args = [verb[1:]] + args
            self.cmd_macro(args)
        else:
            print(f"[!] Unknown command '{verb}'. Type 'help' for a list of available commands.", file=sys.stderr)

        return True

    def cmd_help(self, args: list):
        bar = "─" * 85
        print(f"\n{bar}")
        print(" python-cmat Analysis Commands Reference:")
        print(bar)
        print("  Matrix & Session:")
        print("    load <path> [alias]                 Load .cmat matrix file into session")
        print("    matrix <name_or_index>              Select active matrix for analysis")
        print("    list                                List all loaded matrices with indices and counts")
        print("    info                                Display active matrix metadata and calibrations")
        print("    close [name_or_index]               Unload matrix from session")
        print()
        print("  Energy Calibration (Per-Axis):")
        print("    cal show                            Display current calibration formulas for both axes")
        print("    cal clear [axis]                    Reset calibration to channel coordinates")
        print("    cal <axis> <a0> <a1> [a2]           Set quadratic calibration for Det 1 (0) or Det 2 (1)")
        print("    cal <a0> <a1> [a2]                  Set quadratic calibration for both axes simultaneously")
        print()
        print("  1D Coincidence Gating:")
        print("    gate <axis> w <w0> <w1> [b <b0> <b1>] Coincidence gate with normalized BG subtraction")
        print("    gate clear [axis]                   Clear coincidence gates and restore full projection")
        print("    gate show                           Display active gate windows and BG scale factor")
        print()
        print("  1D Peak Search & Fitting:")
        print("    search [axis] [--method M] [--snr N] Search peaks on 1D/gated spectrum (cwt, prominence)")
        print("    fit_1d <axis> <ch_or_e> [--model M] Fit single photopeak on active 1D spectrum")
        print("    fit_multiplet <axis> <p1> <p2> ...  Simultaneously fit coupled multiplet on straight baseline")
        print("    fit_all [axis] [--range min max]    Auto-fit all candidate peaks with continuum baseline")
        print("    clear_fits [1d|2d|all]              Clear stored fit results")
        print()
        print("  2D Coincidence Fitting:")
        print("    fit_2d <x> <y> [--roi N] [--verbose] 2D coincidence fit (Gamba 4-component decomposition)")
        print()
        print("  Export & Publishing:")
        print("    pdf_1d <axis> <out.pdf> [--fit]     Export vector PDF of 1D (or gated) spectrum")
        print("    pdf_2d <out.pdf> [--x ...] [--fit]  Export vector PDF of 2D coincidence matrix")
        print("    export_1d <axis> <out.dat>          Export 1D spectrum to ASCII data table")
        print("    export_amat <out.mat>               Export 2D matrix to ASCII matrix format")
        print()
        print("  Scripting & Control:")
        print("    macro <filepath>                    Execute commands from macro script file")
        print("    echo <message>                      Print message to terminal")
        print("    sleep <seconds>                     Pause execution")
        print("    quit / exit / q                     Exit shell or stop execution")
        print(f"{bar}\n")

    def cmd_echo(self, args: list):
        print(" ".join(args))

    def cmd_sleep(self, args: list):
        if args:
            try:
                time.sleep(float(args[0]))
            except ValueError:
                pass

    def cmd_load(self, args: list):
        if not args:
            print("[!] Usage: load <filepath> [alias]", file=sys.stderr)
            return
        path_str = args[0]
        alias = args[1] if len(args) > 1 else None
        p = Path(path_str)
        if not p.exists():
            p_alt = Path(__file__).resolve().parent / path_str
            if p_alt.exists():
                p = p_alt
            else:
                print(f"[!] Error: File '{path_str}' not found.", file=sys.stderr)
                return
        idx = self.session.add_matrix_file(p, name=alias)
        self.session.active_index = idx
        m = self.session.matrices[idx]
        print(f"[*] Loaded [{idx + 1}] '{m['name']}' ({m['shape'][0]}×{m['shape'][1]}, {m['total_counts']:,} counts, symmetric={m['is_symmetric']})")

    def cmd_matrix(self, args: list):
        if not args:
            print("[!] Usage: matrix <name_or_index>", file=sys.stderr)
            return
        try:
            idx = self.session.select_matrix(args[0])
            m = self.session.get_active_matrix()
            print(f"[*] Active matrix switched to [{idx + 1}/{len(self.session.matrices)}]: '{m['name']}' ({m['shape'][0]}×{m['shape'][1]}, {m['total_counts']:,} counts)")
        except Exception as e:
            print(f"[!] Error selecting matrix: {e}", file=sys.stderr)

    def cmd_list(self, args: list):
        if not self.session.matrices:
            print("[*] No matrices currently loaded.")
            return
        bar = "─" * 85
        print(f"\n{bar}")
        print(f" {'Idx':<4} {'Act':<4} {'Name':<24} {'Shape':<12} {'Total Counts':<14} {'Max Count':<10} {'Symmetric':<10}")
        print(bar)
        for idx, m in enumerate(self.session.matrices):
            act = "*" if idx == self.session.active_index else " "
            shape_str = f"{m['shape'][0]}×{m['shape'][1]}"
            tot_str = f"{m['total_counts']:,}"
            max_str = f"{m['max_count']:,}"
            sym_str = "Yes" if m['is_symmetric'] else "No"
            print(f"  {idx + 1:<3} {act:^4} {m['name'][:23]:<24} {shape_str:<12} {tot_str:<14} {max_str:<10} {sym_str:<10}")
        print(f"{bar}\n")

    def cmd_info(self, args: list):
        m = self.session.get_active_matrix()
        if not m:
            print("[!] No active matrix loaded.", file=sys.stderr)
            return
        bar = "═" * 70
        subbar = "─" * 70
        print(f"\n{bar}")
        print(f" Active Matrix Metadata: '{m['name']}'")
        print(subbar)
        print(f"  File Path      : {m['path']}")
        print(f"  Dimensions     : {m['shape'][0]} × {m['shape'][1]} channels")
        print(f"  Total Counts   : {m['total_counts']:,}")
        print(f"  Max Bin Count  : {m['max_count']:,}")
        print(f"  Nonzero Bins   : {m['nonzero_bins']:,}")
        print(f"  Symmetric      : {'Yes' if m['is_symmetric'] else 'No'}")
        c0 = self.session.get_cal(0)
        c1 = self.session.get_cal(1)
        c0_cal = "Calibrated" if self.session.is_calibrated(0) else "Uncalibrated"
        c1_cal = "Calibrated" if self.session.is_calibrated(1) else "Uncalibrated"
        print(f"  Det 1 / X Cal  : E = {c0[0]:.4f} + {c0[1]:.6f}*ch + {c0[2]:.8e}*ch^2 [{c0_cal}]")
        print(f"  Det 2 / Y Cal  : E = {c1[0]:.4f} + {c1[1]:.6f}*ch + {c1[2]:.8e}*ch^2 [{c1_cal}]")
        g0 = self.session.gates.get(0)
        g1 = self.session.gates.get(1)
        if g0:
            print(f"  Active Gate 0  : {len(g0.get('w_gates', []))} peak window(s) on Det 1 -> Coinc on Det 2")
        if g1:
            print(f"  Active Gate 1  : {len(g1.get('w_gates', []))} peak window(s) on Det 2 -> Coinc on Det 1")
        print(f"{bar}\n")

    def cmd_close(self, args: list):
        ident = args[0] if args else None
        try:
            self.session.close_matrix(ident)
            print("[*] Matrix closed.")
        except Exception as e:
            print(f"[!] Error closing matrix: {e}", file=sys.stderr)

    def cmd_cal(self, args: list):
        action, axis, coeffs = parse_cal_args(args)
        if action == "show":
            bar = "═" * 84
            print(f"\n{bar}")
            print(" Energy Calibration (Quadratic: E = a0 + a1*ch + a2*ch^2)")
            print("─" * 84)
            axes = [axis] if axis is not None else sorted(self.session.cal.keys())
            if not axes:
                axes = [0, 1]
            for ax in axes:
                c = self.session.get_cal(ax)
                status = "Calibrated (keV)" if self.session.is_calibrated(ax) else "Uncalibrated (Channel)"
                det_label = f"Det {ax + 1} / Axis {ax}" + (" (X)" if ax == 0 else (" (Y)" if ax == 1 else ""))
                print(f"  {det_label:<24}: E = {c[0]:.4f} + {c[1]:.6f}*ch + {c[2]:.8e}*ch^2  [{status}]")
            print(f"{bar}\n")
        elif action == "clear":
            self.session.set_cal(axis, [0.0, 1.0, 0.0])
            target = f"Axis {axis}" if axis is not None else "all axes"
            print(f"[*] Cleared energy calibration for {target} (reset to 1 ch/keV).")
        elif action == "set":
            self.session.set_cal(axis, coeffs)
            c = parse_cal_coefficients(coeffs)
            if axis is None:
                print(f"[*] Calibration set for both axes: E = {c[0]:.4f} + {c[1]:.6f}*ch + {c[2]:.8e}*ch^2")
                print("    [Tip: Set per-axis calibration with 'cal <axis> <a0> <a1> [a2]', e.g. 'cal 0 0.5 1.002 0.000001']")
            else:
                det = f"Det {axis + 1} / Axis {axis}" + (" (X)" if axis == 0 else (" (Y)" if axis == 1 else ""))
                print(f"[*] Calibration set for {det}: E = {c[0]:.4f} + {c[1]:.6f}*ch + {c[2]:.8e}*ch^2")

    def cmd_gate(self, args: list):
        action, axis, w_gates, b_gates = parse_gate_args(args, self.session)
        if action == "clear":
            if axis is None:
                self.session.gates = {0: None, 1: None}
                print("[*] Cleared all active coincidence gates.")
            else:
                self.session.gates[axis] = None
                print(f"[*] Cleared coincidence gate on Axis {axis}.")
            self.last_gated_axis = None
            return

        if action == "show":
            bar = "─" * 70
            print(f"\n{bar}\n Active Coincidence Gates:\n{bar}")
            for ax in (0, 1):
                g = self.session.gates.get(ax)
                if g:
                    det = "Det 1 (X)" if ax == 0 else "Det 2 (Y)"
                    opp = "Det 2 (Y)" if ax == 0 else "Det 1 (X)"
                    w_str = ", ".join(f"[{w[0]}..{w[1]}]" for w in g.get("w_gates", []))
                    b_str = ", ".join(f"[{b[0]}..{b[1]}]" for b in g.get("b_gates", [])) if g.get("b_gates") else "None"
                    print(f"  Gate on {det} -> Slices on {opp}:")
                    print(f"    Peak Windows (W)      : {w_str} (width: {g.get('w_width', 0)} ch)")
                    print(f"    Background Windows (B): {b_str} (width: {g.get('b_width', 0)} ch)")
                    print(f"    BG Scale Factor       : {g.get('scale', 0.0):.4f}")
                    print(f"    Net Gated Counts      : {g.get('net_counts', 0):,}")
                else:
                    det = "Det 1 (X)" if ax == 0 else "Det 2 (Y)"
                    print(f"  {det}: No active gate")
            print(f"{bar}\n")
            return

        m = self.session.get_active_matrix()
        if not m:
            print("[!] No active matrix loaded.", file=sys.stderr)
            return

        if not w_gates:
            print("[!] Error: At least one peak window 'w <min> <max>' is required.", file=sys.stderr)
            return

        gate_res = compute_1d_gate(m["matrix"], axis, w_gates, b_gates)
        self.session.gates[axis] = gate_res
        self.last_gated_axis = 1 - axis

        src_det = "Det 1 (X)" if axis == 0 else "Det 2 (Y)"
        dst_det = "Det 2 (Y)" if axis == 0 else "Det 1 (X)"
        w_str = ", ".join(f"[{w[0]}..{w[1]}]" for w in gate_res["w_gates"])
        b_str = ", ".join(f"[{b[0]}..{b[1]}]" for b in gate_res["b_gates"]) if gate_res["b_gates"] else "None"

        print(f"[*] Applied coincidence gate on {src_det}:")
        print(f"    Peak Windows (W)      : {w_str} (total width: {gate_res['w_width']} ch)")
        if gate_res["b_gates"]:
            print(f"    Background Windows (B): {b_str} (total width: {gate_res['b_width']} ch)")
            print(f"    BG Normalization Scale: {gate_res['scale']:.4f}")
        print(f"    Gated Coincidence on  : {dst_det} (Total Net Counts: {gate_res['net_counts']:,})")

    def cmd_search(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if pos:
            try:
                axis = int(pos[0])
            except ValueError:
                axis = 0
        else:
            axis = self.last_gated_axis if self.last_gated_axis is not None else 0

        method = str(flags.get("method", "cwt")).lower()
        snr = float(flags.get("snr", flags.get("min_snr", 9.0)))
        fwhm = float(flags.get("fwhm", flags.get("fwhm_est", 4.0)))

        c_min = 0
        c_max = None
        if "range" in flags:
            r = flags["range"] if isinstance(flags["range"], (list, tuple)) else [flags["range"]]
            if len(r) >= 2:
                c_min, c_max = int(float(r[0])), int(float(r[1]))

        spec = self.session.get_spectrum(axis)
        if len(spec) == 0:
            print("[!] No spectrum available for peak search.", file=sys.stderr)
            return

        axis_cal = self.session.get_cal(axis)
        res = find_peaks_1d(spec, ch_min=c_min, ch_max=c_max, method=method, min_snr=snr, fwhm_est=fwhm, cal=axis_cal)

        peaks = res.get("peaks", [])
        is_gated = bool(self.session.gates.get(1 - axis))
        det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'}{' Gated' if is_gated else ''})"
        bar = "─" * 75
        print(f"\n{bar}")
        print(f" Peak Search Results ({det_name}, Method: {method.upper()}, Min SNR: {snr})")
        print(bar)
        print(f"  {'#':<4} {'Centroid (ch)':<16} {'Energy (keV)':<16} {'Amplitude (cts)':<18} {'SNR':<8}")
        for idx, p in enumerate(peaks):
            ch_val = p.get('centroid_ch', p.get('channel', 0.0))
            e_val = p.get('centroid_e', p.get('energy', ch_to_energy(ch_val, axis_cal)))
            amp_val = p.get('amplitude', p.get('counts', 0.0))
            c_ch = f"{ch_val:10.2f}"
            c_e = f"{e_val:10.2f}"
            amp = f"{amp_val:12,.1f}"
            snr_val = f"{p.get('snr', 0.0):6.1f}"
            print(f"  {idx + 1:<4} {c_ch:<16} {c_e:<16} {amp:<18} {snr_val:<8}")
        print(bar)
        print(f" Found {len(peaks)} candidate peak(s) in {res.get('elapsed_ms', 0.0):.1f} ms.\n")

    def cmd_fit_1d(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if len(pos) < 2:
            print("[!] Usage: fit_1d <axis> <center_ch_or_energy> [--model gaussian|gaussian_tail|hypermet] [--fwhm_mult 4.0] [--roi half_width] [--energy]", file=sys.stderr)
            return
        axis = int(pos[0])
        center = float(pos[1])
        axis_cal = self.session.get_cal(axis)
        is_cal = self.session.is_calibrated(axis)

        if flags.get("energy") or flags.get("e"):
            center = energy_to_ch(center, axis_cal)

        model = str(flags.get("model", flags.get("fit_type", "gaussian"))).lower()
        fwhm_mult = float(flags.get("fwhm_mult", 4.0))
        roi = int(float(flags["roi"])) if "roi" in flags else None

        spec = self.session.get_spectrum(axis)
        if len(spec) == 0:
            print("[!] No spectrum available for fitting.", file=sys.stderr)
            return

        m = self.session.get_active_matrix()
        mat_name = m["name"] if m else "matrix"
        is_gated = bool(self.session.gates.get(1 - axis))
        det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'}{' Gated Coincidence' if is_gated else ' Projection'})"

        try:
            res = fit_gaussian_peak(np.arange(len(spec)) + 0.5, spec, center, fit_type=model, fwhm_mult=fwhm_mult, roi_half_width=roi, cal=axis_cal)
            res["axis"] = axis
            self.session.fits_1d[axis] = res
            verbosity = "detailed" if (flags.get("verbose") or flags.get("v")) else "compact"
            print_fit_terminal_report(res, det_name, mat_name, is_cal, verbosity=verbosity)
        except Exception as e:
            print(f"[!] 1D peak fit error: {e}", file=sys.stderr)

    def cmd_fit_multiplet(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if len(pos) < 2:
            print("[!] Usage: fit_multiplet <axis> <ch1> <ch2> ... [--model gaussian|...] [--region left right] [--energy]", file=sys.stderr)
            return
        axis = int(pos[0])
        axis_cal = self.session.get_cal(axis)
        is_cal = self.session.is_calibrated(axis)

        raw_peaks = [float(x) for x in pos[1:]]
        if flags.get("energy") or flags.get("e"):
            peak_channels = [energy_to_ch(p, axis_cal) for p in raw_peaks]
        else:
            peak_channels = raw_peaks

        model = str(flags.get("model", "gaussian")).lower()
        fwhm_est = float(flags.get("fwhm_est", 4.0))
        fwhm_mult = float(flags.get("fwhm_mult", 4.0))

        region = None
        if "region" in flags:
            r = flags["region"] if isinstance(flags["region"], (list, tuple)) else [flags["region"]]
            if len(r) >= 2:
                r0, r1 = float(r[0]), float(r[1])
                if flags.get("energy") or flags.get("e"):
                    r0, r1 = energy_to_ch(r0, axis_cal), energy_to_ch(r1, axis_cal)
                region = [r0, r1]

        spec = self.session.get_spectrum(axis)
        ch_min = int(min(peak_channels) - fwhm_est * 4) if peak_channels else 0
        ch_max = int(max(peak_channels) + fwhm_est * 4) if peak_channels else len(spec) - 1

        res = fit_all_peaks_1d(spec, ch_min, ch_max, peak_channels, fit_type=model, fwhm_est=fwhm_est, cal=axis_cal, fwhm_mult=fwhm_mult, region=region)
        if res.get("success"):
            self.session.fits_1d[axis] = res
            m = self.session.get_active_matrix()
            det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'})"
            print_multi_fit_terminal_report(res, det_name, m["name"] if m else "matrix", is_cal)
        else:
            print(f"[!] Multiplet fit failed: {res.get('error', 'Unknown error')}", file=sys.stderr)

    def cmd_fit_all(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        axis = int(pos[0]) if pos else (self.last_gated_axis if self.last_gated_axis is not None else 0)
        axis_cal = self.session.get_cal(axis)
        is_cal = self.session.is_calibrated(axis)

        model = str(flags.get("model", "gaussian")).lower()
        snr = float(flags.get("snr", 9.0))
        fwhm_est = float(flags.get("fwhm_est", 4.0))
        spec = self.session.get_spectrum(axis)

        ch_min = 0
        ch_max = len(spec) - 1
        if "range" in flags:
            r = flags["range"] if isinstance(flags["range"], (list, tuple)) else [flags["range"]]
            if len(r) >= 2:
                r0, r1 = float(r[0]), float(r[1])
                if flags.get("energy") or flags.get("e"):
                    r0, r1 = energy_to_ch(r0, axis_cal), energy_to_ch(r1, axis_cal)
                ch_min, ch_max = int(min(r0, r1)), int(max(r0, r1))

        search_res = find_peaks_1d(spec, ch_min=ch_min, ch_max=ch_max, method="cwt", min_snr=snr, fwhm_est=fwhm_est, cal=axis_cal)
        peak_chs = [p.get("channel", p.get("centroid_ch", 0.0)) for p in search_res.get("peaks", [])]

        res = fit_all_peaks_1d(spec, ch_min, ch_max, peak_chs, fit_type=model, fwhm_est=fwhm_est, cal=axis_cal)
        if res.get("success"):
            self.session.fits_1d[axis] = res
            m = self.session.get_active_matrix()
            det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'})"
            print_multi_fit_terminal_report(res, det_name, m["name"] if m else "matrix", is_cal)
        else:
            print(f"[!] Auto-fit failed: {res.get('error', 'Unknown error')}", file=sys.stderr)

    def cmd_clear_fits(self, args: list):
        target = args[0].lower() if args else "all"
        if target in ("1d", "all"):
            self.session.fits_1d = {0: None, 1: None}
        if target in ("2d", "all"):
            self.session.fit_2d = None
        print(f"[*] Cleared {target} fit cache.")

    def cmd_fit_2d(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if len(pos) < 2:
            print("[!] Usage: fit_2d <x_center> <y_center> [--model gaussian|gaussian_tail|hypermet] [--roi 16] [--verbose] [--energy]", file=sys.stderr)
            return
        x_c = float(pos[0])
        y_c = float(pos[1])
        cal_x = self.session.get_cal(0)
        cal_y = self.session.get_cal(1)

        if flags.get("energy") or flags.get("e"):
            x_c = energy_to_ch(x_c, cal_x)
            y_c = energy_to_ch(y_c, cal_y)

        model = str(flags.get("model", "gaussian")).lower()
        roi = int(float(flags.get("roi", 16)))
        verbose = bool(flags.get("verbose") or flags.get("v"))

        m = self.session.get_active_matrix()
        if not m:
            print("[!] No active matrix loaded.", file=sys.stderr)
            return

        mat = m["matrix"]
        proj_y = m["proj"] if m["is_symmetric"] else np.sum(mat, axis=1, dtype=np.float64)
        is_cal = (self.session.is_calibrated(0), self.session.is_calibrated(1))

        try:
            res = fit_2d_gaussian_peak(
                mat, x_c, y_c, fit_type=model, cal_x=cal_x, cal_y=cal_y, roi_half_width=roi,
                proj_x=m["proj"], proj_y=proj_y, total_counts=m["total_counts"]
            )
            self.session.fit_2d = res
            print_fit_2d_terminal_report(res, m["name"], is_cal, verbosity="detailed" if verbose else "compact")
        except Exception as e:
            print(f"[!] 2D peak fit error: {e}", file=sys.stderr)

    def cmd_pdf_1d(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if len(pos) < 2:
            print("[!] Usage: pdf_1d <axis> <output_filename.pdf> [--range min max] [--log] [--zoom_y 1.0] [--fit] [--title '...']", file=sys.stderr)
            return
        axis = int(pos[0])
        outfile = Path(pos[1])
        spec = self.session.get_spectrum(axis)
        if len(spec) == 0:
            print("[!] No spectrum available for PDF export.", file=sys.stderr)
            return

        ch_start = 0
        ch_end = len(spec) - 1
        if "range" in flags:
            r = flags["range"] if isinstance(flags["range"], (list, tuple)) else [flags["range"]]
            if len(r) >= 2:
                r0, r1 = float(r[0]), float(r[1])
                if flags.get("energy") or flags.get("e"):
                    axis_cal = self.session.get_cal(axis)
                    r0, r1 = energy_to_ch(r0, axis_cal), energy_to_ch(r1, axis_cal)
                ch_start, ch_end = int(r0), int(r1)

        is_log = bool(flags.get("log"))
        zoom_y = float(flags.get("zoom_y", 1.0))
        fit_res = self.session.fits_1d.get(axis) if flags.get("fit") else None
        title = flags.get("title")

        pdf_bytes = generate_pdf_1d(
            spec, ch_start, ch_end, is_log=is_log, zoom_y=zoom_y, fit_res=fit_res,
            cal=self.session.get_cal(axis), title=title
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(pdf_bytes)
        print(f"[+] Exported 1D spectrum PDF: {outfile.resolve()}")

    def cmd_pdf_2d(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if not pos:
            print("[!] Usage: pdf_2d <output_filename.pdf> [--x min max] [--y min max] [--cmap turbo] [--scale log] [--vmin N] [--vmax N] [--fit] [--title '...']", file=sys.stderr)
            return
        outfile = Path(pos[0])
        m = self.session.get_active_matrix()
        if not m:
            print("[!] No active matrix loaded.", file=sys.stderr)
            return

        mat = m["matrix"]
        H, W = mat.shape
        x0, x1 = 0, W
        y0, y1 = 0, H

        cal_x = self.session.get_cal(0)
        cal_y = self.session.get_cal(1)

        if "x" in flags:
            rx = flags["x"] if isinstance(flags["x"], (list, tuple)) else [flags["x"]]
            if len(rx) >= 2:
                v0, v1 = float(rx[0]), float(rx[1])
                if flags.get("energy") or flags.get("e"):
                    v0, v1 = energy_to_ch(v0, cal_x), energy_to_ch(v1, cal_x)
                x0, x1 = int(v0), int(v1)

        if "y" in flags:
            ry = flags["y"] if isinstance(flags["y"], (list, tuple)) else [flags["y"]]
            if len(ry) >= 2:
                v0, v1 = float(ry[0]), float(ry[1])
                if flags.get("energy") or flags.get("e"):
                    v0, v1 = energy_to_ch(v0, cal_y), energy_to_ch(v1, cal_y)
                y0, y1 = int(v0), int(v1)

        cmap = str(flags.get("cmap", "turbo"))
        scale = str(flags.get("scale", "log"))
        vmin = float(flags.get("vmin", 0))
        vmax = float(flags.get("vmax", 100))
        fit_2d_res = self.session.fit_2d if flags.get("fit") else None
        title = flags.get("title")

        pdf_bytes = generate_pdf_2d(
            mat, x0, x1, y0, y1, cmap_name=cmap, scale_mode=scale, vmin=vmin, vmax=vmax,
            fit_2d_res=fit_2d_res, cal_x=cal_x, cal_y=cal_y, title=title
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(pdf_bytes)
        print(f"[+] Exported 2D coincidence matrix PDF: {outfile.resolve()}")

    def cmd_export_1d(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if len(pos) < 2:
            print("[!] Usage: export_1d <axis> <output_filename.dat>", file=sys.stderr)
            return
        axis = int(pos[0])
        outfile = Path(pos[1])
        spec = self.session.get_spectrum(axis)
        m = self.session.get_active_matrix()
        hdr = f"Matrix: {m['name'] if m else 'unknown'} | Det {axis + 1} ({'X' if axis == 0 else 'Y'})"
        export_1d_ascii(outfile, spec, cal=self.session.get_cal(axis), header=hdr)
        print(f"[+] Exported 1D spectrum data: {outfile.resolve()}")

    def cmd_export_amat(self, args: list):
        pos, flags = parse_cmd_tokens(args)
        if not pos:
            print("[!] Usage: export_amat <output_filename.mat>", file=sys.stderr)
            return
        outfile = Path(pos[0])
        m = self.session.get_active_matrix()
        if not m:
            print("[!] No active matrix loaded.", file=sys.stderr)
            return
        hdr = f"Matrix: {m['name']}"
        export_amat_ascii(outfile, m["matrix"], header=hdr)
        print(f"[+] Exported 2D matrix data: {outfile.resolve()}")

    def cmd_macro(self, args: list):
        if not args:
            print("[!] Usage: macro <filepath>", file=sys.stderr)
            return
        mac_path = Path(args[0])
        if not mac_path.exists():
            mac_alt = Path(__file__).resolve().parent / args[0]
            if mac_alt.exists():
                mac_path = mac_alt
            else:
                print(f"[!] Error: Macro file '{args[0]}' not found.", file=sys.stderr)
                return
        self.execute_script(mac_path)

    def execute_script(self, filepath: Path) -> bool:
        if self.macro_depth >= self.max_macro_depth:
            print(f"[!] Maximum macro recursion depth ({self.max_macro_depth}) exceeded.", file=sys.stderr)
            return False

        self.macro_depth += 1
        print(f"[*] Executing macro: {filepath.name} ...")
        t0 = time.time()
        line_num = 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_num += 1
                    line_clean = line.strip()
                    if not line_clean or line_clean.startswith("#") or line_clean.startswith("//"):
                        continue
                    if not self.execute_line(line_clean):
                        if self.should_exit:
                            print(f"[*] Macro '{filepath.name}' exited cleanly at line {line_num}.")
                            return True
                        print(f"[*] Macro '{filepath.name}' stopped at line {line_num}.")
                        return False
            print(f"[+] Macro '{filepath.name}' completed in {time.time() - t0:.2f}s.")
            return True
        except Exception as e:
            print(f"[!] Error in macro '{filepath.name}' at line {line_num}: {e}", file=sys.stderr)
            return False
        finally:
            self.macro_depth -= 1

    def run_repl(self):
        print("\n" + "═" * 75)
        print("  python-cmat Interactive Headless Shell")
        print("  Type 'help' for command list, 'quit' or Ctrl+D to exit.")
        print("═" * 75 + "\n")
        try:
            import readline
        except ImportError:
            pass

        while True:
            try:
                active = self.session.get_active_matrix()
                mat_prompt = active["name"] if active else "no-matrix"
                line = input(f"cmat [{mat_prompt}]> ")
                if not self.execute_line(line):
                    break
            except (EOFError, KeyboardInterrupt):
                print("\n[*] Exiting cmat shell.")
                break
            except Exception as e:
                print(f"[!] Error: {e}", file=sys.stderr)


class CMATWebHandler(BaseHTTPRequestHandler):
    session: "CMATSession" = None
    matrices: list = []
    active_index: int = 0
    reader: CMATReader = None
    matrix: np.ndarray = None
    proj: np.ndarray = None
    cal: list = [0.0, 1.0, 0.0]
    cal_0: list = [0.0, 1.0, 0.0]
    cal_1: list = [0.0, 1.0, 0.0]
    global_cal: list = [0.0, 1.0, 0.0]
    config: dict = None
    config_path: Path = None

    @classmethod
    def get_session(cls) -> "CMATSession":
        if cls.session is None:
            cls.session = CMATSession(cls.config, cls.config_path)
        return cls.session

    @classmethod
    def add_matrix_file(cls, path: Path, name: str = None, cal=None) -> int:
        session = cls.get_session()
        idx = session.add_matrix_file(path, name=name, cal=cal)
        cls.matrices = session.matrices
        cls.active_index = session.active_index
        cls.sync_class_attrs()
        return idx

    @classmethod
    def get_active_matrix(cls):
        session = cls.get_session()
        return session.get_active_matrix()

    @classmethod
    def sync_class_attrs(cls):
        session = cls.get_session()
        cls.matrices = session.matrices
        cls.active_index = session.active_index
        m = session.get_active_matrix()
        if m:
            cls.reader = m["reader"]
            cls.matrix = m["matrix"]
            cls.proj = m["proj"]
            cls.cal = session.get_cal(0)
            cls.cal_0 = session.get_cal(0)
            cls.cal_1 = session.get_cal(1)

    @property
    def active_matrix_data(self):
        return self.get_active_matrix()

    def get_metadata_dict(self):
        info = self.reader.get_info() if self.reader else {}
        active_mat = self.active_matrix_data
        if active_mat:
            info["filename"] = active_mat["name"]
            info["filepath"] = active_mat["path"]
        session = self.get_session()
        c0 = session.get_cal(0)
        c1 = session.get_cal(1)
        info["cal"] = c0
        info["cal_0"] = c0
        info["cal_1"] = c1
        info["cal_axes"] = {
            0: c0,
            1: c1,
            "0": c0,
            "1": c1,
        }
        info["is_calibrated_0"] = session.is_calibrated(0)
        info["is_calibrated_1"] = session.is_calibrated(1)
        info["config"] = self.config or DEFAULT_CONFIG
        info["config_file"] = self.config_path.name if self.config_path else CONFIG_FILENAME
        info["max_count"] = int(np.max(self.matrix)) if self.matrix is not None else 0
        info["total_counts"] = int(np.sum(self.matrix)) if self.matrix is not None else 0
        info["nonzero_bins"] = int(np.count_nonzero(self.matrix)) if self.matrix is not None else 0
        info["active_index"] = session.active_index
        info["matrices"] = [
            {
                "index": m["index"],
                "name": m["name"],
                "filename": m["filename"],
                "path": m["path"],
                "shape": m["shape"],
                "total_counts": m["total_counts"],
                "max_count": m["max_count"],
                "nonzero_bins": m["nonzero_bins"],
                "is_symmetric": m["is_symmetric"],
                "cal_0": m.get("cal", {}).get(0, c0) if isinstance(m.get("cal"), dict) else c0,
                "cal_1": m.get("cal", {}).get(1, c1) if isinstance(m.get("cal"), dict) else c1,
            }
            for m in session.matrices
        ]
        return info

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_html_content().encode("utf-8"))

        elif self.path == "/api/quit":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "shutting_down"}).encode("utf-8"))
            print("\n[*] Quit request received from browser. Server shutting down gracefully...\n", flush=True)
            threading.Timer(0.15, lambda: os._exit(0)).start()

        elif self.path == "/api/metadata":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            info = self.get_metadata_dict()
            self.wfile.write(json.dumps(info).encode("utf-8"))

        elif self.path.startswith("/api/select_matrix"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            idx_str = query.get("index", [query.get("id", ["0"])[0]])[0]
            try:
                idx = int(idx_str)
            except ValueError:
                idx = 0

            session = self.get_session()
            if 0 <= idx < len(session.matrices):
                session.select_matrix(idx)
                CMATWebHandler.sync_class_attrs()
                m = session.matrices[idx]
                print(f"\n[*] Active matrix switched to [{idx + 1}/{len(session.matrices)}]: {m['name']} (Shape: {m['shape'][0]}×{m['shape'][1]}, Total counts: {m['total_counts']:,})", flush=True)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            info = self.get_metadata_dict()
            self.wfile.write(json.dumps(info).encode("utf-8"))

        elif self.path.startswith("/api/fit_peak_2d"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x = float(query.get("x", [0])[0])
            y = float(query.get("y", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0]
            roi_half_width = int(float(query.get("roi_half_width", [query.get("roi_width", [16])[0]])[0]))
            verbosity = query.get("verbosity", ["compact"])[0].lower()

            session = self.get_session()
            cal_x = session.get_cal(0)
            cal_y = session.get_cal(1)
            is_cal_x = session.is_calibrated(0)
            is_cal_y = session.is_calibrated(1)

            proj_y = self.proj if (self.reader and self.reader.is_symmetric) else np.sum(self.matrix, axis=1, dtype=np.float64)
            tot_counts = float(np.sum(self.proj))
            try:
                res = fit_2d_gaussian_peak(
                    self.matrix, x, y, fit_type=fit_type, cal_x=cal_x, cal_y=cal_y, roi_half_width=roi_half_width,
                    proj_x=self.proj, proj_y=proj_y, total_counts=tot_counts
                )
                print_fit_2d_terminal_report(res, self.reader.filename.name, (is_cal_x, is_cal_y), verbosity=verbosity)
            except Exception as e:
                res = {"success": False, "error": str(e), "is_2d": True}
                print(f"[!] 2D coincidence peak fit error at ({x:.1f}, {y:.1f}): {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif self.path.startswith("/api/fit_peak"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            axis = int(query.get("axis", [0])[0])
            channel = float(query.get("channel", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0]
            fwhm_mult = float(query.get("fwhm_mult", [4.0])[0])
            verbosity = query.get("verbosity", ["compact"])[0].lower()
            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))

            w_str = query.get("w_gates", [""])[0]
            b_str = query.get("b_gates", [""])[0]
            w_gates = parse_gate_ranges(w_str)
            b_gates = parse_gate_ranges(b_str)

            if w_gates:
                # Gated coincidence mode: gate was set on the other axis (1 - axis)
                gate_axis = 1 - axis
                gate_res = compute_1d_gate(self.matrix, gate_axis, w_gates, b_gates)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
                det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'} Gated Coincidence)"
            elif axis == 0:
                if y0 == 0 and y1 >= self.matrix.shape[0]:
                    spec = self.proj
                else:
                    spec = np.sum(self.matrix[y0:y1, :], axis=0, dtype=np.float64)
                det_name = "Det 1 (X Projection)"
            else:
                if x0 == 0 and x1 >= self.matrix.shape[1]:
                    spec = np.sum(self.matrix, axis=1, dtype=np.float64)
                else:
                    spec = np.sum(self.matrix[:, x0:x1], axis=1, dtype=np.float64)
                det_name = "Det 2 (Y Projection)"

            session = self.get_session()
            axis_cal = session.get_cal(axis)
            is_cal = session.is_calibrated(axis)
            try:
                res = fit_gaussian_peak(np.arange(len(spec)) + 0.5, spec, channel, fit_type=fit_type, fwhm_mult=fwhm_mult, cal=axis_cal)
                res["axis"] = axis
                print_fit_terminal_report(res, det_name, self.reader.filename.name, is_cal, verbosity=verbosity)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis}
                print(f"[!] Peak fit error at channel {channel:.1f}: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif self.path.startswith("/api/gate_1d"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            axis = int(query.get("axis", [0])[0])
            w_str = query.get("w_gates", [""])[0]
            b_str = query.get("b_gates", [""])[0]
            w_gates = parse_gate_ranges(w_str)
            b_gates = parse_gate_ranges(b_str)

            gate_res = compute_1d_gate(self.matrix, axis, w_gates, b_gates)
            print_gate_terminal_report(gate_res, self.reader.filename.name)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(gate_res).encode("utf-8"))

        elif self.path.startswith("/api/search_peaks"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            axis = int(query.get("axis", [0])[0])
            method = query.get("method", ["cwt"])[0].lower()
            min_snr = float(query.get("min_snr", [9.0])[0])
            min_counts = float(query.get("min_counts", [10.0])[0])
            fwhm_est = float(query.get("fwhm_est", [4.0])[0])
            range_mode = query.get("range", ["visible"])[0].lower()

            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))

            w_str = query.get("w_gates", [""])[0]
            b_str = query.get("b_gates", [""])[0]
            w_gates = parse_gate_ranges(w_str)
            b_gates = parse_gate_ranges(b_str)

            if w_gates:
                gate_axis = 1 - axis
                gate_res = compute_1d_gate(self.matrix, gate_axis, w_gates, b_gates)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
                det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'} Gated Coincidence)"
            elif axis == 0:
                if y0 == 0 and y1 >= self.matrix.shape[0]:
                    spec = self.proj
                else:
                    spec = np.sum(self.matrix[y0:y1, :], axis=0, dtype=np.float64)
                det_name = "Det 1 (X Projection)"
            else:
                if x0 == 0 and x1 >= self.matrix.shape[1]:
                    spec = np.sum(self.matrix, axis=1, dtype=np.float64)
                else:
                    spec = np.sum(self.matrix[:, x0:x1], axis=1, dtype=np.float64)
                det_name = "Det 2 (Y Projection)"

            if range_mode == "visible":
                if axis == 0:
                    ch_min, ch_max = x0, x1
                else:
                    ch_min, ch_max = y0, y1
            else:
                ch_min, ch_max = 0, len(spec)

            session = self.get_session()
            axis_cal = session.get_cal(axis)
            is_cal = session.is_calibrated(axis)
            try:
                res = find_peaks_1d(
                    spec, ch_min=ch_min, ch_max=ch_max, method=method,
                    min_snr=min_snr, min_counts=min_counts, fwhm_est=fwhm_est, cal=axis_cal
                )
                res["axis"] = axis
                print_peaks_terminal_report(res, det_name, self.reader.filename.name, is_cal)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis, "peaks": [], "count": 0}
                print(f"[!] Peak search error: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif self.path.startswith("/api/fit_all_peaks_1d"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            axis = int(query.get("axis", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0].lower()
            fwhm_est = float(query.get("fwhm_est", [4.0])[0])
            min_snr = float(query.get("min_snr", [9.0])[0])

            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))

            w_str = query.get("w_gates", [""])[0]
            b_str = query.get("b_gates", [""])[0]
            w_gates = parse_gate_ranges(w_str)
            b_gates = parse_gate_ranges(b_str)

            if w_gates:
                gate_axis = 1 - axis
                gate_res = compute_1d_gate(self.matrix, gate_axis, w_gates, b_gates)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
                det_name = f"Det {axis + 1} ({'X' if axis == 0 else 'Y'} Gated Coincidence)"
            elif axis == 0:
                if y0 == 0 and y1 >= self.matrix.shape[0]:
                    spec = self.proj
                else:
                    spec = np.sum(self.matrix[y0:y1, :], axis=0, dtype=np.float64)
                det_name = "Det 1 (X Projection)"
            else:
                if x0 == 0 and x1 >= self.matrix.shape[1]:
                    spec = np.sum(self.matrix, axis=1, dtype=np.float64)
                else:
                    spec = np.sum(self.matrix[:, x0:x1], axis=1, dtype=np.float64)
                det_name = "Det 2 (Y Projection)"

            # Determine display channel bounds
            if axis == 0:
                ch_min = int(float(query.get("ch_min", [x0])[0]))
                ch_max = int(float(query.get("ch_max", [x1])[0]))
            else:
                ch_min = int(float(query.get("ch_min", [y0])[0]))
                ch_max = int(float(query.get("ch_max", [y1])[0]))
            ch_min = max(0, min(len(spec) - 2, ch_min))
            ch_max = max(ch_min + 1, min(len(spec) - 1, ch_max))

            session = self.get_session()
            axis_cal = session.get_cal(axis)
            is_cal = session.is_calibrated(axis)

            peaks_str = query.get("peaks", [""])[0]
            if peaks_str.strip():
                try:
                    peak_channels = [float(c.strip()) for c in peaks_str.split(",") if c.strip()]
                except Exception:
                    peak_channels = []
            else:
                peak_channels = []

            # If no peaks provided or empty, auto-detect peaks in this display window first
            if not peak_channels:
                search_method = query.get("search_method", ["cwt"])[0].lower()
                search_res = find_peaks_1d(
                    spec, ch_min=ch_min, ch_max=ch_max, method=search_method,
                    min_snr=min_snr, fwhm_est=fwhm_est, cal=axis_cal
                )
                peak_channels = [p["channel"] for p in search_res.get("peaks", [])]

            fwhm_mult = float(query.get("fwhm_mult", [4.0])[0])
            bg_method = query.get("bg_method", ["peak_aware"])[0].lower()
            snip_iter_val = query.get("snip_iter", [None])[0]
            snip_iter = int(snip_iter_val) if snip_iter_val is not None and str(snip_iter_val).isdigit() else None
            region_str = query.get("region", [""])[0]
            region_bounds = None
            if region_str.strip():
                try:
                    parts = [float(v.strip()) for v in region_str.split(",") if v.strip()]
                    if len(parts) >= 2:
                        region_bounds = [parts[0], parts[1]]
                except Exception:
                    region_bounds = None

            t0 = time.time()
            try:
                res = fit_all_peaks_1d(
                    spec, ch_min=ch_min, ch_max=ch_max, peak_channels=peak_channels,
                    fit_type=fit_type, fwhm_est=fwhm_est, cal=axis_cal, fwhm_mult=fwhm_mult,
                    bg_method=bg_method, snip_iter=snip_iter, region=region_bounds
                )
                res["axis"] = axis
                res["elapsed_ms"] = round((time.time() - t0) * 1000.0, 1)
                if res.get("success"):
                    print_multi_fit_terminal_report(res, det_name, self.reader.filename.name, is_cal)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis, "peaks": [], "elapsed_ms": round((time.time() - t0) * 1000.0, 1)}
                print(f"[!] Multi-peak fit error: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif self.path.startswith("/api/projection_region"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))

            # X Projection (Det 1): Sum along Y axis between y0 and y1
            if y0 == 0 and y1 >= self.matrix.shape[0]:
                spec_x = self.proj
            else:
                spec_x = np.sum(self.matrix[y0:y1, :], axis=0, dtype=np.int64)

            # Y Projection (Det 2): Sum along X axis between x0 and x1
            if x0 == 0 and x1 >= self.matrix.shape[1]:
                spec_y = np.sum(self.matrix, axis=1, dtype=np.int64)
            else:
                spec_y = np.sum(self.matrix[:, x0:x1], axis=1, dtype=np.int64)

            resp = {
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "specX": spec_x.tolist(),
                "specY": spec_y.tolist(),
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.path.startswith("/api/value"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x", [0])[0]))))
            y = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y", [0])[0]))))
            val = int(self.matrix[y, x])
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"x": x, "y": y, "value": val}).encode("utf-8"))

        elif self.path.startswith("/api/tile"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))

            target_w = max(16, min(2048, int(query.get("w", [800])[0])))
            target_h = max(16, min(2048, int(query.get("h", [600])[0])))

            sub = self.matrix[y0:y1, x0:x1]
            sh_y, sh_x = sub.shape

            # Exact pixel-matched downsampling or 1:1 slice
            if sh_x <= target_w and sh_y <= target_h:
                out = np.ascontiguousarray(sub, dtype=np.int32)
            else:
                step_x = max(1, int(np.ceil(sh_x / target_w)))
                step_y = max(1, int(np.ceil(sh_y / target_h)))

                pad_y = (step_y - (sh_y % step_y)) % step_y
                pad_x = (step_x - (sh_x % step_x)) % step_x
                if pad_y > 0 or pad_x > 0:
                    sub_padded = np.pad(sub, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
                else:
                    sub_padded = sub

                new_h = sub_padded.shape[0] // step_y
                new_w = sub_padded.shape[1] // step_x

                # 2D Max pooling preserves all gamma coincidence peaks
                out = sub_padded.reshape(new_h, step_y, new_w, step_x).max(axis=(1, 3))
                out = np.ascontiguousarray(out[:target_h, :target_w], dtype=np.int32)

            actual_h, actual_w = out.shape
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("X-Shape-W", str(actual_w))
            self.send_header("X-Shape-H", str(actual_h))
            self.send_header("X-Slice-X0", str(x0))
            self.send_header("X-Slice-X1", str(x1))
            self.send_header("X-Slice-Y0", str(y0))
            self.send_header("X-Slice-Y1", str(y1))
            self.end_headers()
            self.wfile.write(out.tobytes())

        elif self.path.startswith("/api/export_pdf_1d"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            axis = int(query.get("axis", [0])[0])
            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))
            is_synced = int(query.get("is_synced", [1])[0]) == 1
            is_log = int(query.get("is_log", [0])[0]) == 1
            zoom_y = float(query.get("zoom_y", [1.0])[0])
            has_fit = int(query.get("has_fit", [0])[0]) == 1
            fit_ch = float(query.get("fit_ch", [0])[0]) if has_fit else None
            fit_type = query.get("fit_type", ["gaussian"])[0]
            fwhm_mult = float(query.get("fwhm_mult", [4.0])[0])

            w_str = query.get("w_gates", [""])[0]
            b_str = query.get("b_gates", [""])[0]
            w_gates = parse_gate_ranges(w_str)
            b_gates = parse_gate_ranges(b_str)

            if w_gates:
                gate_axis = 1 - axis
                gate_res = compute_1d_gate(self.matrix, gate_axis, w_gates, b_gates)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
                det_name = f"Det{axis + 1}_{'X' if axis == 0 else 'Y'}_Gated"
                ch_start = (x0 if axis == 0 else y0) if is_synced else 0
                ch_end = (x1 if axis == 0 else y1) if is_synced else len(spec) - 1
            elif axis == 0:
                if y0 == 0 and y1 >= self.matrix.shape[0]:
                    spec = self.proj
                else:
                    spec = np.sum(self.matrix[y0:y1, :], axis=0, dtype=np.float64)
                det_name = "Det1_X"
                ch_start = x0 if is_synced else 0
                ch_end = x1 if is_synced else len(spec) - 1
            else:
                if x0 == 0 and x1 >= self.matrix.shape[1]:
                    spec = np.sum(self.matrix, axis=1, dtype=np.float64)
                else:
                    spec = np.sum(self.matrix[:, x0:x1], axis=1, dtype=np.float64)
                det_name = "Det2_Y"
                ch_start = y0 if is_synced else 0
                ch_end = y1 if is_synced else len(spec) - 1

            session = self.get_session()
            axis_cal = session.get_cal(axis)

            fit_res = None
            if has_fit and fit_ch is not None:
                try:
                    fit_res = fit_gaussian_peak(np.arange(len(spec)) + 0.5, spec, fit_ch, fit_type=fit_type, fwhm_mult=fwhm_mult, cal=axis_cal)
                except Exception:
                    fit_res = None

            try:
                pdf_bytes = generate_pdf_1d(spec, ch_start, ch_end, is_log=is_log, zoom_y=zoom_y, fit_res=fit_res, cal=axis_cal)
                self.send_response(200)
                self.send_header("Content-type", "application/pdf")
                self.send_header("Content-Disposition", f'attachment; filename="{self.reader.filename.stem}_{det_name}_{ch_start}_{ch_end}.pdf"')
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif self.path.startswith("/api/export_pdf_2d"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))
            cmap = query.get("cmap", ["turbo"])[0]
            scale = query.get("scale", ["log"])[0]
            vmin = int(query.get("vmin", [1])[0])
            vmax = int(query.get("vmax", [0])[0])
            if vmax <= 0:
                sub = self.matrix[y0:y1, x0:x1]
                vmax = int(np.max(sub)) if sub.size > 0 else 100
            has_fit = int(query.get("has_fit", [0])[0]) == 1
            fit_x = float(query.get("fit_x", [0])[0]) if has_fit else None
            fit_y = float(query.get("fit_y", [0])[0]) if has_fit else None
            fit_type = query.get("fit_type", ["gaussian"])[0]
            fwhm_mult = float(query.get("fwhm_mult", [4.0])[0])

            session = self.get_session()
            cal_x = session.get_cal(0)
            cal_y = session.get_cal(1)

            fit_2d_res = None
            if has_fit and fit_x is not None and fit_y is not None:
                try:
                    proj_y = self.proj if (self.reader and self.reader.is_symmetric) else np.sum(self.matrix, axis=1, dtype=np.float64)
                    fit_2d_res = fit_2d_gaussian_peak(
                        self.matrix, fit_x, fit_y, fit_type=fit_type, cal_x=cal_x, cal_y=cal_y, roi_half_width=16,
                        proj_x=self.proj, proj_y=proj_y, total_counts=float(np.sum(self.proj))
                    )
                except Exception:
                    fit_2d_res = None

            try:
                pdf_bytes = generate_pdf_2d(self.matrix, x0, x1, y0, y1, cmap_name=cmap, scale_mode=scale, vmin=vmin, vmax=vmax, fit_2d_res=fit_2d_res, cal_x=cal_x, cal_y=cal_y)
                self.send_response(200)
                self.send_header("Content-type", "application/pdf")
                self.send_header("Content-Disposition", f'attachment; filename="{self.reader.filename.stem}_2D_Matrix_{x0}_{x1}_{y0}_{y1}.pdf"')
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/save_config":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode("utf-8"))

                if CMATWebHandler.config is None:
                    CMATWebHandler.config = DEFAULT_CONFIG.copy()

                CMATWebHandler.config.update(data)
                target_path = CMATWebHandler.config_path if CMATWebHandler.config_path else (Path.cwd() / CONFIG_FILENAME)
                save_config_file(target_path, CMATWebHandler.config)
                print(f"[*] Configuration saved to {target_path.name}", flush=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "message": target_path.name}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

        elif self.path.startswith("/api/upload_matrix"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            filename = query.get("filename", ["uploaded_matrix.cmat"])[0]
            safe_name = Path(filename).name
            if not safe_name.endswith(".cmat"):
                safe_name += ".cmat"

            content_len = int(self.headers.get("Content-Length", 0))
            if content_len <= 0:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Empty upload payload"}).encode("utf-8"))
                return

            upload_dir = Path.cwd() / ".cmat_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            target_path = upload_dir / safe_name

            bytes_read = 0
            with open(target_path, "wb") as f:
                while bytes_read < content_len:
                    chunk = self.rfile.read(min(65536, content_len - bytes_read))
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_read += len(chunk)

            try:
                new_idx = CMATWebHandler.add_matrix_file(target_path, name=safe_name, cal=self.cal)
                CMATWebHandler.active_index = new_idx
                CMATWebHandler.sync_class_attrs()
                m = CMATWebHandler.matrices[new_idx]
                print(f"\n[+] Successfully uploaded and loaded: {safe_name} ({m['shape'][0]}×{m['shape'][1]}, {m['total_counts']:,} counts)", flush=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                info = self.get_metadata_dict()
                self.wfile.write(json.dumps(info).encode("utf-8"))
            except Exception as e:
                print(f"[!] Error loading uploaded matrix: {e}", file=sys.stderr)
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

        elif self.path == "/api/load_matrix_path":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                path_str = data.get("path", "").strip()
                if not path_str:
                    raise ValueError("No path provided")

                target_path = Path(path_str).expanduser()
                if not target_path.is_absolute():
                    target_path = (Path.cwd() / target_path).resolve()
                else:
                    target_path = target_path.resolve()

                if not target_path.exists():
                    raise FileNotFoundError(f"Matrix file not found: {target_path}")

                new_idx = CMATWebHandler.add_matrix_file(target_path, cal=self.cal)
                CMATWebHandler.active_index = new_idx
                CMATWebHandler.sync_class_attrs()
                m = CMATWebHandler.matrices[new_idx]
                print(f"\n[+] Successfully loaded matrix from path: {target_path.name} ({m['shape'][0]}×{m['shape'][1]}, {m['total_counts']:,} counts)", flush=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                info = self.get_metadata_dict()
                self.wfile.write(json.dumps(info).encode("utf-8"))
            except Exception as e:
                print(f"[!] Error loading matrix path: {e}", file=sys.stderr)
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")


HTML_FILE_PATH = Path(__file__).resolve().parent / "cmat_webviewer.html"


def get_html_content() -> str:
    """Reads and returns the HTML viewer template from the standalone HTML file."""
    if not HTML_FILE_PATH.exists():
        raise FileNotFoundError(f"Viewer frontend template '{HTML_FILE_PATH}' was not found.")
    return HTML_FILE_PATH.read_text(encoding="utf-8")

def main():
    config_path = Path.cwd() / CONFIG_FILENAME
    config = load_or_create_config(config_path)

    parser = argparse.ArgumentParser(
        description="Launch modern Web-based interactive 2D viewer or run automated spectroscopy analysis in headless mode."
    )
    parser.add_argument(
        "input",
        nargs="*",
        type=str,
        default=[],
        help="Path to one or more input .cmat file(s) (e.g. run1.cmat run2.cmat or *.cmat)",
    )
    parser.add_argument(
        "-m", "--macro",
        type=str,
        default=None,
        help="Execute a .mac command script file in headless mode and exit",
    )
    parser.add_argument(
        "-c", "--batch", "--command",
        type=str,
        dest="command",
        default=None,
        help="Execute one or more semicolon-separated analysis commands in headless mode and exit",
    )
    parser.add_argument(
        "-i", "--headless", "--interactive", "--shell",
        action="store_true",
        dest="headless",
        default=False,
        help="Start interactive headless CLI shell (REPL) without opening a web browser or server",
    )
    parser.add_argument(
        "-H", "--host",
        type=str,
        default=None,
        help=f"Web server host/interface to bind (default from config: {config.get('host', '0.0.0.0')})",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help=f"Web server port (default from config: {config.get('port', 8080)})",
    )
    parser.add_argument(
        "--browser",
        type=str,
        default=None,
        help="Specify browser name/command to open (e.g. 'firefox', 'chrome', 'default', or 'none')",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=None,
        help="Do not automatically open any web browser",
    )
    parser.add_argument(
        "--cal",
        nargs="+",
        type=float,
        metavar="COEFF",
        default=None,
        help="Global calibration coefficients: a0 a1 [a2] applied to all axes (overrides config)",
    )
    parser.add_argument(
        "--cal-0", "--cal-x",
        nargs="+",
        type=float,
        dest="cal_0",
        metavar="COEFF",
        default=None,
        help="Axis 0 (Det 1 / X) calibration coefficients: a0 a1 [a2] (overrides config)",
    )
    parser.add_argument(
        "--cal-1", "--cal-y",
        nargs="+",
        type=float,
        dest="cal_1",
        metavar="COEFF",
        default=None,
        help="Axis 1 (Det 2 / Y) calibration coefficients: a0 a1 [a2] (overrides config)",
    )

    args = parser.parse_args()

    # Expand any globs/wildcards in input arguments
    file_paths = []
    for item in args.input:
        matched = glob.glob(item)
        if matched:
            file_paths.extend([Path(p) for p in sorted(matched)])
        else:
            file_paths.append(Path(item))

    # De-duplicate while preserving CLI order
    seen = set()
    unique_paths = []
    for p in file_paths:
        res = p.resolve()
        if res not in seen:
            seen.add(res)
            unique_paths.append(p)

    for p in unique_paths:
        if not p.exists():
            print(f"Error: File '{p}' not found.", file=sys.stderr)
            sys.exit(1)

    # Initialize Session
    session = CMATSession(config, config_path)

    # Apply calibrations
    if args.cal is not None:
        session.set_cal(None, args.cal)
    if args.cal_0 is not None:
        session.set_cal(0, args.cal_0)
    if args.cal_1 is not None:
        session.set_cal(1, args.cal_1)

    # Pre-load matrices if provided
    for p in unique_paths:
        session.add_matrix_file(p)

    # Check for Headless Mode: Macro, Batch Command, or Interactive REPL
    is_headless_mode = bool(args.macro or args.command or args.headless)

    if is_headless_mode:
        interpreter = CMATCommandInterpreter(session)
        success = True

        if args.command:
            success = interpreter.execute_batch(args.command)
            if not success:
                sys.exit(1)

        if args.macro:
            mac_path = Path(args.macro)
            if not mac_path.exists():
                mac_alt = Path(__file__).resolve().parent / args.macro
                if mac_alt.exists():
                    mac_path = mac_alt
                else:
                    print(f"Error: Macro file '{args.macro}' not found.", file=sys.stderr)
                    sys.exit(1)
            success = interpreter.execute_script(mac_path)
            if not success:
                sys.exit(1)

        if args.headless:
            interpreter.run_repl()

        sys.exit(0 if success else 1)

    # Otherwise: Web Viewer Mode
    if not unique_paths:
        # Check if default GeE-symm.cmat exists in cwd
        def_file = Path.cwd() / "GeE-symm.cmat"
        if def_file.exists():
            unique_paths.append(def_file)
            session.add_matrix_file(def_file)
        else:
            print("Error: No input .cmat files specified.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    # Resolve web server settings
    host = args.host if args.host is not None else str(config.get("host", "0.0.0.0")).strip()
    port = args.port if args.port is not None else int(config.get("port", 8080))
    browser_choice = args.browser if args.browser is not None else str(config.get("browser", "default")).strip()

    if args.no_browser is True or browser_choice.lower() in ("none", "no", "false", "0", "off"):
        open_browser = False
    else:
        open_br = config.get("open_browser", True)
        open_browser = (open_br in (True, "true", "True", "1", 1))

    in_ssh = is_ssh_session()
    if in_ssh and args.browser is None and open_browser:
        open_browser = False
        ssh_browser_skipped = True
    else:
        ssh_browser_skipped = False

    CMATWebHandler.session = session
    CMATWebHandler.config = config
    CMATWebHandler.config_path = config_path
    CMATWebHandler.sync_class_attrs()

    print(f"[*] Loaded {len(session.matrices)} matrix file{'s' if len(session.matrices) > 1 else ''}:")
    for idx, m in enumerate(session.matrices):
        print(f"    [{idx + 1}/{len(session.matrices)}] '{m['name']}' ({m['shape'][0]}×{m['shape'][1]}, {m['total_counts']:,} counts)")

    # Bind HTTP server
    bind_host = "" if host in ("0.0.0.0", "", "::") else host
    server_address = (bind_host, port)
    httpd = HTTPServer(server_address, CMATWebHandler)

    local_ip = get_local_ip()
    if host in ("0.0.0.0", "", "::"):
        network_url = f"http://{local_ip}:{port}"
        local_url = f"http://localhost:{port}"
    elif host in ("127.0.0.1", "localhost"):
        network_url = None
        local_url = f"http://127.0.0.1:{port}"
    else:
        network_url = f"http://{host}:{port}"
        local_url = f"http://localhost:{port}"

    print(f"\n[+] Interactive 2D CMAT Web Viewer is ready!")
    if network_url and network_url != local_url:
        print(f"[+] Network URL (Remote / LAN): {network_url}")
        print(f"[+] Localhost URL:             {local_url}")
    else:
        print(f"[+] Access the viewer at:      {local_url}")
    print(f"[+] Classic Binned 1D Histogram with real-time mouse inspector.")

    if in_ssh:
        print(f"\n[*] SSH session detected:")
        if ssh_browser_skipped:
            print(f"    - Remote X11 browser auto-launch skipped to keep your SSH session fast.")
        print(f"    - Open the Network URL above ({network_url or local_url}) in your local client browser.")
        print(f"    - (Or use SSH port forwarding: ssh -L {port}:localhost:{port} user@server)")

    print(f"\n[+] Press Ctrl+C in terminal to stop server.\n")

    if open_browser:
        target_url = local_url if host in ("127.0.0.1", "localhost") else (network_url or local_url)
        threading.Timer(0.6, lambda: launch_browser(target_url, browser_choice)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    main()
