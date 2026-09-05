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
import math
import json
import argparse
import webbrowser
import threading
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
    "fit_type": "gaussian",
    "fwhm_mult_1d": 4.0,
    "roi_half_width_2d": 16,
    "fit_verbosity": "compact",
    "colormap": "turbo",
    "scale_mode": "log",
    "vmax": 500,
    "vmin": 1,
    "scroll_sensitivity": 4,
    "proj_range": "synced",
    "proj_scale": "linear",
    "port": 8080,
    "open_browser": True,
}


def parse_cal_string(cal_val) -> list:
    """Parse calibration string (e.g. '0.0, 1.0, 0.0' or '0 1 0') into a list of floats."""
    if isinstance(cal_val, (list, tuple)):
        return [float(v) for v in cal_val]
    if isinstance(cal_val, str):
        parts = cal_val.replace(",", " ").split()
        return [float(p) for p in parts] if parts else [0.0, 1.0, 0.0]
    return [0.0, 1.0, 0.0]


def generate_config_content(cfg: dict) -> str:
    """Generate clean, commented INI-style python-cmat-config.txt text."""
    cal_str = cfg.get("cal", "0.0, 1.0, 0.0")
    if isinstance(cal_str, (list, tuple)):
        cal_str = ", ".join(str(v) for v in cal_str)

    open_br = cfg.get("open_browser", True)
    open_br_str = "true" if open_br in (True, "true", "True", "1", 1) else "false"

    return f"""# ==============================================================================
# python-cmat configuration file
# Automatically generated when no config file is present in the working directory.
# You can edit these values directly or click "Save Config" in the Web Viewer.
# ==============================================================================

# Energy Calibration: a0 a1 a2 for E = a0 + a1*ch + a2*ch^2
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

# 1D Projection Display Range: synced, full
proj_range = {cfg.get('proj_range', 'synced')}

# 1D Projection Y-Scale: linear, log
proj_scale = {cfg.get('proj_scale', 'linear')}

# Web Server Port
port = {cfg.get('port', 8080)}

# Automatically open web browser on launch: true, false
open_browser = {open_br_str}
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
                    if key in cfg:
                        if key == "fwhm_mult_1d":
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
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    i_center = int(round(x_center))

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
    bg_at_apex = b0_init + b1_init * (mu_init - x_center)
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
            dx_c = roi_x - x_center
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
                dx_s, dx_cs = roi_x - mu_s, roi_x - x_center
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
        dx_c = roi_x - x_center
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
    bg_roi_sum = np.sum(b0 + b1 * (roi_x[win_mask] - x_center))
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
    dx_c_dense = x_dense - x_center
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
        "roi_ch_min": int(roi_x[0]),
        "roi_ch_max": int(roi_x[-1]),
        "curve_x": [round(float(v), 2) for v in x_dense],
        "curve_fit": [round(float(v), 2) for v in fit_dense],
        "curve_bg": [round(float(v), 2) for v in bg_dense],
    }


def print_fit_terminal_report(res, det_name, filename, is_cal, verbosity="compact"):
    ft = res.get("fit_type", "gaussian")
    model_tag = "Hypermet" if ft == "hypermet" else ("RadWare" if ft == "gaussian_tail" else "Gaussian")
    if verbosity == "compact":
        if is_cal:
            print(f"⚛ 1D Fit [{det_name}] ({model_tag}): Centroid: {res['centroid_e']:.2f}({res['centroid_e_err']:.2f}) keV   Area: {res['area']:.1f}({res['area_err']:.1f}) counts   FWHM: {res['fwhm_e']:.2f}({res['fwhm_e_err']:.2f}) keV", flush=True)
        else:
            print(f"⚛ 1D Fit [{det_name}] ({model_tag}): Centroid: {res['centroid_ch']:.3f}({res['centroid_ch_err']:.3f}) ch   Area: {res['area']:.1f}({res['area_err']:.1f}) counts   FWHM: {res['fwhm_ch']:.3f}({res['fwhm_ch_err']:.3f}) ch", flush=True)
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
    print(f"⚛ GASPware 1D Peak Fit [{det_name}] - {filename} ({model_name})")
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
    proj_x=None, proj_y=None, total_counts=None, **kwargs
):
    mat = np.asarray(matrix, dtype=np.float64)
    H_mat, W_mat = mat.shape

    ix = int(round(x_center))
    iy = int(round(y_center))

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

    xs = np.arange(x_min, x_max + 1, dtype=np.float64)
    ys = np.arange(y_min, y_max + 1, dtype=np.float64)
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

            dxc = x_flat - ix
            dyc = y_flat - iy
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

    a0 = cal[0] if len(cal) > 0 else 0.0
    a1 = cal[1] if len(cal) > 1 else 1.0
    a2 = cal[2] if len(cal) > 2 else 0.0
    def ch_to_e(c): return a0 + a1 * c + a2 * (c**2)
    def de_dch(c): return abs(a1 + 2.0 * a2 * c)

    e_x = ch_to_e(mx)
    e_x_err = param_errors[6] * de_dch(mx)
    e_y = ch_to_e(my)
    e_y_err = param_errors[7] * de_dch(my)

    fwhm_e_x = fwhm_x * de_dch(mx)
    fwhm_e_x_err = fwhm_x_err * de_dch(mx)
    fwhm_e_y = fwhm_y * de_dch(my)
    fwhm_e_y_err = fwhm_y_err * de_dch(my)

    ndf = max(1, N_pixels - n_params)
    red_chi2 = chi2 / ndf

    # Background decomposition counts in ROI from continuous model
    dx = x_flat - mx
    dy = y_flat - my
    dxc = x_flat - ix
    dyc = y_flat - iy
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
    ix_c = int(round(mx)) - x_min
    iy_c = int(round(my)) - y_min

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
    proj_x=None, proj_y=None, total_counts=None, recenter=True, **kwargs
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
        proj_x=proj_x, proj_y=proj_y, total_counts=total_counts, **kwargs
    )

    if not recenter or not res_prelim.get("success"):
        return res_prelim

    # Check fitted centroid coordinates
    mx = res_prelim.get("centroid_x_ch", x_center)
    my = res_prelim.get("centroid_y_ch", y_center)
    ix_orig = int(round(x_center))
    iy_orig = int(round(y_center))
    ix_new = int(round(mx))
    iy_new = int(round(my))

    # If peak is offset from initial click position, recenter ROI and perform refined Pass 2
    if (ix_new != ix_orig or iy_new != iy_orig) or (abs(mx - x_center) > 0.4 or abs(my - y_center) > 0.4):
        ix_clamped = max(roi_half_width + 1, min(W_mat - 1 - roi_half_width - 1, ix_new))
        iy_clamped = max(roi_half_width + 1, min(H_mat - 1 - roi_half_width - 1, iy_new))
        try:
            res_refined = _fit_2d_gaussian_single_roi(
                mat, ix_clamped, iy_clamped, fit_type=fit_type, cal=cal, roi_half_width=roi_half_width,
                proj_x=proj_x, proj_y=proj_y, total_counts=total_counts, **kwargs
            )
            if res_refined.get("success"):
                return res_refined
        except Exception:
            pass

    return res_prelim


def print_fit_2d_terminal_report(res, filename, is_cal, verbosity="compact"):
    ft = res.get("fit_type", "gaussian")
    model_tag = "Hypermet" if ft == "hypermet" else ("RadWare" if ft == "gaussian_tail" else "Gaussian")
    if verbosity == "compact":
        if is_cal:
            print(f"⚛ 2D Fit [{filename}] ({model_tag} + Gamba BG):", flush=True)
            print(f"  Det 1 (X): Centroid: {res['centroid_x_e']:.2f}({res['centroid_x_e_err']:.2f}) keV   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_x_e']:.2f}({res['fwhm_x_e_err']:.2f}) keV", flush=True)
            print(f"  Det 2 (Y): Centroid: {res['centroid_y_e']:.2f}({res['centroid_y_e_err']:.2f}) keV   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_y_e']:.2f}({res['fwhm_y_e_err']:.2f}) keV", flush=True)
            print(f"  Gamba Net Area (p|p^t): {res['gamba_net']:.1f} ± {res['gamba_net_err']:.1f} counts   Peak/Total-BG Ratio (Π): {res['pi_ratio_percent']:.1f}%\n", flush=True)
        else:
            print(f"⚛ 2D Fit [{filename}] ({model_tag} + Gamba BG):", flush=True)
            print(f"  Det 1 (X): Centroid: {res['centroid_x_ch']:.3f}({res['centroid_x_ch_err']:.3f}) ch   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_x_ch']:.3f}({res['fwhm_x_ch_err']:.3f}) ch", flush=True)
            print(f"  Det 2 (Y): Centroid: {res['centroid_y_ch']:.3f}({res['centroid_y_ch_err']:.3f}) ch   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_y_ch']:.3f}({res['fwhm_y_ch_err']:.3f}) ch", flush=True)
            print(f"  Gamba Net Area (p|p^t): {res['gamba_net']:.1f} ± {res['gamba_net_err']:.1f} counts   Peak/Total-BG Ratio (Π): {res['pi_ratio_percent']:.1f}%\n", flush=True)
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
    print(f"⚛ GASPware 2D Coincidence Peak Fit - {filename} ({model_name})")
    print(f"   [Self-Consistent 4-Component BG Decomposition: Gamba et al., NIM A 928 (2019) 93]")
    print(subbar)
    if is_cal:
        print(f"  2D Centroid (Energy): ({res['centroid_x_e']:.2f} ± {res['centroid_x_e_err']:.2f}, {res['centroid_y_e']:.2f} ± {res['centroid_y_e_err']:.2f}) keV")
        print(f"  2D Centroid (ch)    : ({res['centroid_x_ch']:.3f} ± {res['centroid_x_ch_err']:.3f}, {res['centroid_y_ch']:.3f} ± {res['centroid_y_ch_err']:.3f}) ch")
        print(f"  FWHM (Energy)       : Det 1 (X) = {res['fwhm_x_e']:.2f} ± {res['fwhm_x_e_err']:.2f} keV | Det 2 (Y) = {res['fwhm_y_e']:.2f} ± {res['fwhm_y_e_err']:.2f} keV")
        print(f"  FWHM (ch)           : Det 1 (X) = {res['fwhm_x_ch']:.3f} ± {res['fwhm_x_ch_err']:.3f} ch  | Det 2 (Y) = {res['fwhm_y_ch']:.3f} ± {res['fwhm_y_ch_err']:.3f} ch")
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


def generate_pdf_1d(spec, ch_start, ch_end, is_log=False, zoom_y=1.0, fit_res=None):
    """
    Generates a publication-quality 1D spectrum vector PDF with white background,
    Times New Roman font, inward ticks, stepped staircase histogram, and Energy (keV) axis.
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

    # Stepped histogram centered on channels (1 keV/channel calibrated)
    ax.step(sub_x, sub_y, where="mid", color="#111111", linewidth=1.0, label="Data")

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
        ax.set_ylim(bottom=0, top=max(1.0, y_max))

    ax.set_xlim(ch_start, ch_end)
    ax.set_xlabel("Energy (keV)", fontsize=12, labelpad=6)
    ax.set_ylabel("Counts", fontsize=12, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)

    # Plot fitted peak curve and baseline if available
    if fit_res and fit_res.get("success"):
        curve_x = np.array(fit_res.get("curve_x", []), dtype=np.float64)
        curve_fit = np.array(fit_res.get("curve_fit", []), dtype=np.float64)
        curve_bg = np.array(fit_res.get("curve_bg", []), dtype=np.float64)

        if len(curve_x) > 0 and len(curve_fit) > 0:
            if len(curve_bg) > 0:
                ax.plot(curve_x, curve_bg, color="#cc0066", linestyle="--", linewidth=1.2, label="Background")
            ax.plot(curve_x, curve_fit, color="#d95f02", linestyle="-", linewidth=1.8, label="Fit")

            mu = fit_res.get("centroid_ch")
            if mu is not None:
                ax.axvline(mu, color="#d95f02", linestyle=":", linewidth=1.0)

            area_str = f"{fit_res.get('area', 0):.1f} ± {fit_res.get('area_err', 0):.1f}"
            fwhm_str = f"{fit_res.get('fwhm_ch', 0):.2f} ± {fit_res.get('fwhm_ch_err', 0):.2f}"
            centroid_str = f"{fit_res.get('centroid_ch', 0):.2f} ± {fit_res.get('centroid_ch_err', 0):.2f}"
            info_txt = f"Centroid: {centroid_str} keV\nArea: {area_str}\nFWHM: {fwhm_str} keV\n$\\chi^2_\\nu$: {fit_res.get('red_chi2', 0):.2f}"
            ax.text(0.04, 0.94, info_txt, transform=ax.transAxes, verticalalignment="top",
                    fontsize=9, fontfamily="serif", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999", alpha=0.92))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf_2d(matrix, x0, x1, y0, y1, cmap_name="turbo", scale_mode="log", vmin=0, vmax=100, fit_2d_res=None):
    """
    Generates a publication-quality 2D coincidence matrix vector PDF with white background,
    Times New Roman font, Energy (keV) axes labels, colorbar, and optional fit overlays.
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

    im = ax.imshow(sub_mat, extent=[x0, x1, y0, y1], origin="lower", cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xlabel("Energy (keV)", fontsize=12, labelpad=6)
    ax.set_ylabel("Energy (keV)", fontsize=12, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Counts", fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    # Draw 2D fit crosshair, ellipse and ROI box if active
    if fit_2d_res and fit_2d_res.get("success"):
        cx = fit_2d_res.get("centroid_x_ch", 0) + 0.5
        cy = fit_2d_res.get("centroid_y_ch", 0) + 0.5
        fwhm_x = fit_2d_res.get("fwhm_x_ch", 4.0)
        fwhm_y = fit_2d_res.get("fwhm_y_ch", 4.0)

        # ROI box
        rx0 = fit_2d_res.get("roi_x_min", cx - 8)
        rx1 = fit_2d_res.get("roi_x_max", cx + 8) + 1
        ry0 = fit_2d_res.get("roi_y_min", cy - 8)
        ry1 = fit_2d_res.get("roi_y_max", cy + 8) + 1
        import matplotlib.patches as patches
        rect = patches.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, linewidth=1.0, edgecolor="#ffd600", facecolor="none", linestyle="--", alpha=0.7)
        ax.add_patch(rect)

        # Crosshair
        ax.plot([cx - 4, cx + 4], [cy, cy], color="#ff0066", linewidth=1.2)
        ax.plot([cx, cx], [cy - 4, cy + 4], color="#ff0066", linewidth=1.2)

        # Ellipse
        ell = Ellipse((cx, cy), width=fwhm_x, height=fwhm_y, angle=0, edgecolor="#ffd600", facecolor="none", linewidth=1.6)
        ax.add_patch(ell)

        # Text annotation with Net Volume and Centroid
        vol_val = fit_2d_res.get("volume", 0)
        vol_err = fit_2d_res.get("volume_err", 0)
        vol_str = f"Net Vol: {vol_val:,.0f} ± {vol_err:,.0f} cts" if vol_err else f"Net Vol: {vol_val:,.0f} cts"
        ax.text(0.02, 0.98, f"2D Coincidence Peak\nCentroid: ({cx-0.5:.1f}, {cy-0.5:.1f})\n{vol_str}",
                transform=ax.transAxes, verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#191c20", edgecolor="#ffd600", alpha=0.85),
                color="#ffffff", fontfamily="monospace")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


class CMATWebHandler(BaseHTTPRequestHandler):
    reader: CMATReader = None
    matrix: np.ndarray = None
    proj: np.ndarray = None
    cal: list = [0.0, 1.0, 0.0]
    config: dict = None
    config_path: Path = None

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
            info = self.reader.get_info()
            info["cal"] = self.cal
            info["config"] = self.config or DEFAULT_CONFIG
            info["config_file"] = self.config_path.name if self.config_path else CONFIG_FILENAME
            info["max_count"] = int(np.max(self.matrix))
            info["total_counts"] = int(np.sum(self.matrix))
            info["nonzero_bins"] = int(np.count_nonzero(self.matrix))
            self.wfile.write(json.dumps(info).encode("utf-8"))

        elif self.path.startswith("/api/fit_peak_2d"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x = float(query.get("x", [0])[0])
            y = float(query.get("y", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0]
            roi_half_width = int(float(query.get("roi_half_width", [query.get("roi_width", [16])[0]])[0]))
            verbosity = query.get("verbosity", ["compact"])[0].lower()

            is_cal = self.cal and (self.cal[0] != 0.0 or self.cal[1] != 1.0 or self.cal[2] != 0.0)
            proj_y = self.proj if (self.reader and self.reader.is_symmetric) else np.sum(self.matrix, axis=1, dtype=np.float64)
            tot_counts = float(np.sum(self.proj))
            try:
                res = fit_2d_gaussian_peak(
                    self.matrix, x, y, fit_type=fit_type, cal=self.cal, roi_half_width=roi_half_width,
                    proj_x=self.proj, proj_y=proj_y, total_counts=tot_counts
                )
                print_fit_2d_terminal_report(res, self.reader.filename.name, is_cal, verbosity=verbosity)
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

            if axis == 0:
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

            is_cal = self.cal and (self.cal[0] != 0.0 or self.cal[1] != 1.0 or self.cal[2] != 0.0)
            try:
                res = fit_gaussian_peak(np.arange(len(spec)), spec, channel, fit_type=fit_type, fwhm_mult=fwhm_mult, cal=self.cal)
                res["axis"] = axis
                print_fit_terminal_report(res, det_name, self.reader.filename.name, is_cal, verbosity=verbosity)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis}
                print(f"[!] Peak fit error at channel {channel:.1f}: {e}", file=sys.stderr)

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

            if axis == 0:
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

            fit_res = None
            if has_fit and fit_ch is not None:
                try:
                    fit_res = fit_gaussian_peak(np.arange(len(spec)), spec, fit_ch, fit_type=fit_type, fwhm_mult=fwhm_mult, cal=self.cal)
                except Exception:
                    fit_res = None

            try:
                pdf_bytes = generate_pdf_1d(spec, ch_start, ch_end, is_log=is_log, zoom_y=zoom_y, fit_res=fit_res)
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

            fit_2d_res = None
            if has_fit and fit_x is not None and fit_y is not None:
                try:
                    proj_y = self.proj if (self.reader and self.reader.is_symmetric) else np.sum(self.matrix, axis=1, dtype=np.float64)
                    fit_2d_res = fit_2d_gaussian_peak(
                        self.matrix, fit_x, fit_y, fit_type=fit_type, cal=self.cal, roi_half_width=16,
                        proj_x=self.proj, proj_y=proj_y, total_counts=float(np.sum(self.proj))
                    )
                except Exception:
                    fit_2d_res = None

            try:
                pdf_bytes = generate_pdf_2d(self.matrix, x0, x1, y0, y1, cmap_name=cmap, scale_mode=scale, vmin=vmin, vmax=vmax, fit_2d_res=fit_2d_res)
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
        description="Launch modern Web-based interactive 2D viewer with classic binned 1D histogram."
    )
    parser.add_argument("input", type=str, help="Path to input .cmat file")
    parser.add_argument("-p", "--port", type=int, default=None, help=f"Web server port (default from config: {config.get('port', 8080)})")
    parser.add_argument("--no-browser", action="store_true", default=None, help="Do not automatically open the web browser")
    parser.add_argument(
        "--cal",
        nargs="+",
        type=float,
        metavar="COEFF",
        default=None,
        help="Energy calibration coefficients: a0 a1 a2 for E = a0 + a1*ch + a2*ch^2 (overrides config)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Resolve settings: CLI arguments override config file defaults
    port = args.port if args.port is not None else int(config.get("port", 8080))
    if args.no_browser is True:
        open_browser = False
    else:
        open_br = config.get("open_browser", True)
        open_browser = (open_br in (True, "true", "True", "1", 1))

    if args.cal is not None:
        cal = args.cal
    else:
        cal = parse_cal_string(config.get("cal", [0.0, 1.0, 0.0]))

    print(f"[*] Loading matrix from {input_path} ...")
    reader = CMATReader(input_path)
    mat = reader.to_numpy()
    proj = reader.get_projection()

    CMATWebHandler.reader = reader
    CMATWebHandler.matrix = mat
    CMATWebHandler.proj = proj
    CMATWebHandler.cal = cal
    CMATWebHandler.config = config
    CMATWebHandler.config_path = config_path

    server_address = ("", port)
    httpd = HTTPServer(server_address, CMATWebHandler)
    url = f"http://localhost:{port}"
    print(f"\n[+] Interactive 2D CMAT Web Viewer is ready!")
    print(f"[+] Access the viewer at: {url}")
    print(f"[+] Classic Binned 1D Histogram with real-time mouse inspector.")
    print(f"[+] Press Ctrl+C in terminal to stop server.\n")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    main()
