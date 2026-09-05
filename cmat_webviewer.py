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
    if verbosity == "compact":
        if is_cal:
            print(f"⚛ 1D Fit [{det_name}]: Centroid: {res['centroid_e']:.2f}({res['centroid_e_err']:.2f}) keV   Area: {res['area']:.1f}({res['area_err']:.1f}) counts   FWHM: {res['fwhm_e']:.2f}({res['fwhm_e_err']:.2f}) keV", flush=True)
        else:
            print(f"⚛ 1D Fit [{det_name}]: Centroid: {res['centroid_ch']:.3f}({res['centroid_ch_err']:.3f}) ch   Area: {res['area']:.1f}({res['area_err']:.1f}) counts   FWHM: {res['fwhm_ch']:.3f}({res['fwhm_ch_err']:.3f}) ch", flush=True)
        return

    bar = "═" * 80
    subbar = "─" * 80
    ft = res.get("fit_type")
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


def fit_2d_gaussian_peak(
    matrix, x_center, y_center, fit_type="gaussian", cal=[0.0, 1.0, 0.0], roi_half_width=16,
    proj_x=None, proj_y=None, total_counts=None, **kwargs
):
    """
    Fits a true 2D coincidence peak (Symmetric Gaussian, RadWare Tail, or Hypermet Model)
    on the 2D gamma-gamma coincidence matrix using the self-consistent 4-component background
    decomposition established by Gamba et al. (NIM A 928, 2019, 93-103) & Morhác et al. (NIM A 401, 1997, 113):
      - bg|bg: 2D Compton continuum + accidental random coincidences: b0 + bx*(x - x_c) + by*(y - y_c)
      - p|bg : Det 1 peak with Det 2 Compton/random continuum ridge: R_x * P_X(x)
      - bg|p : Det 2 peak with Det 1 Compton/random continuum ridge: R_y * P_Y(y)
      - p|p^t: True 2D coincidence peak volume: H * P_X(x) * P_Y(y)
    
    All background components are extracted self-consistently from the 2D spectrum without arbitrary parameters.
    """
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


def print_fit_2d_terminal_report(res, filename, is_cal, verbosity="compact"):
    if verbosity == "compact":
        if is_cal:
            print(f"⚛ 2D Fit [{filename}] (Gamba & Morhác BG Decomposition):", flush=True)
            print(f"  Det 1 (X): Centroid: {res['centroid_x_e']:.2f}({res['centroid_x_e_err']:.2f}) keV   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_x_e']:.2f}({res['fwhm_x_e_err']:.2f}) keV", flush=True)
            print(f"  Det 2 (Y): Centroid: {res['centroid_y_e']:.2f}({res['centroid_y_e_err']:.2f}) keV   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_y_e']:.2f}({res['fwhm_y_e_err']:.2f}) keV", flush=True)
            print(f"  Gamba Net Area (p|p^t): {res['gamba_net']:.1f} ± {res['gamba_net_err']:.1f} counts   Peak/Total-BG Ratio (Π): {res['pi_ratio_percent']:.1f}%\n", flush=True)
        else:
            print(f"⚛ 2D Fit [{filename}] (Gamba & Morhác BG Decomposition):", flush=True)
            print(f"  Det 1 (X): Centroid: {res['centroid_x_ch']:.3f}({res['centroid_x_ch_err']:.3f}) ch   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_x_ch']:.3f}({res['fwhm_x_ch_err']:.3f}) ch", flush=True)
            print(f"  Det 2 (Y): Centroid: {res['centroid_y_ch']:.3f}({res['centroid_y_ch_err']:.3f}) ch   Area: {res['volume']:.1f}({res['volume_err']:.1f}) counts   FWHM: {res['fwhm_y_ch']:.3f}({res['fwhm_y_ch_err']:.3f}) ch", flush=True)
            print(f"  Gamba Net Area (p|p^t): {res['gamba_net']:.1f} ± {res['gamba_net_err']:.1f} counts   Peak/Total-BG Ratio (Π): {res['pi_ratio_percent']:.1f}%\n", flush=True)
        return

    bar = "═" * 80
    subbar = "─" * 80
    ft = res.get("fit_type")
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

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

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


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GASPware 2D Matrix Interactive Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  body {
    background: #111;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  header {
    background: #1a1a1a;
    padding: 8px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #2d2d2d;
  }
  .title { font-size: 1.1rem; font-weight: bold; color: #00e5ff; display: flex; align-items: center; gap: 8px; }
  .badge { background: #263238; color: #80cbc4; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; }
  .hud { font-family: monospace; font-size: 0.88rem; color: #ffd600; background: #212121; padding: 4px 12px; border-radius: 4px; border: 1px solid #424242; }

  main { display: flex; flex: 1; overflow: hidden; }

  aside {
    width: 270px;
    background: #161616;
    border-right: 1px solid #262626;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }
  .control-group {
    background: #1f1f1f;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #2f2f2f;
  }
  .control-group h3 {
    font-size: 0.78rem;
    text-transform: uppercase;
    color: #90caf9;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
  }
  label { font-size: 0.78rem; color: #aaa; display: block; margin-bottom: 3px; }
  select, input[type=range], button {
    width: 100%;
    padding: 5px 8px;
    background: #2b2b2b;
    border: 1px solid #3d3d3d;
    color: #fff;
    border-radius: 4px;
    font-size: 0.82rem;
    outline: none;
  }
  select:focus, input:focus { border-color: #00e5ff; }
  button {
    cursor: pointer;
    background: #00838f;
    border: none;
    font-weight: bold;
    transition: background 0.15s;
    margin-top: 5px;
  }
  button:hover { background: #00acc1; }
  button.secondary { background: #37474f; }
  button.secondary:hover { background: #546e7a; }
  .range-val { font-family: monospace; font-size: 0.78rem; color: #80d8ff; float: right; }

  .workspace { flex: 1; display: flex; flex-direction: row; padding: 8px; gap: 8px; overflow: hidden; }
  .panel-2d {
    flex: 6;
    background: #000;
    border-radius: 6px;
    position: relative;
    border: 1px solid #2a2a2a;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #canvas2d { width: 100%; flex: 1; min-height: 0; display: block; cursor: crosshair; }
  .panel-2d-footer {
    height: 26px;
    background: #141414;
    border-top: 1px solid #282828;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 10px;
    font-size: 0.74rem;
    font-family: monospace;
    color: #00e5ff;
    flex-shrink: 0;
  }
  .panel-1d-container {
    flex: 4.8;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow: hidden;
  }
  .panel-1d {
    flex: 1;
    min-height: 0;
    background: #181818;
    border-radius: 6px;
    border: 1px solid #2a2a2a;
    display: flex;
    flex-direction: column;
    padding: 6px 8px;
  }
  .panel-1d canvas { width: 100%; flex: 1; min-height: 0; background: #111; border-radius: 4px; cursor: crosshair; display: block; }
  .panel-1d-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .panel-1d-title { font-size: 0.78rem; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .fit-results-card {
    background: #191c20;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 0.74rem;
    font-family: monospace;
    flex-shrink: 0;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    margin-top: 6px;
  }
  .fit-card-1d {
    border: 1px solid #00e5ff;
  }
  .fit-card-2d {
    border: 1px solid #ffd600;
  }
  .fit-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
    font-weight: bold;
    color: #ffd600;
  }
  .fit-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 12px;
  }
  .fit-item { display: flex; justify-content: space-between; }
  .fit-item .lbl { color: #888; margin-right: 6px; }
  .fit-item .val { color: #fff; font-weight: bold; }
  .fit-btn-close {
    background: transparent;
    border: 1px solid #555;
    color: #bbb;
    padding: 1px 6px;
    font-size: 0.7rem;
    border-radius: 3px;
    cursor: pointer;
    width: auto;
    margin-top: 0;
  }
  .fit-btn-close:hover { background: #333; color: #fff; }
  .fit-btn-toggle {
    background: transparent;
    border: 1px solid #444;
    color: #aaa;
    padding: 1px 6px;
    font-size: 0.68rem;
    border-radius: 3px;
    cursor: pointer;
    width: auto;
    margin-top: 0;
  }
  .fit-btn-toggle:hover { background: #333; color: #fff; }
  .fit-compact-box {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 2px 0;
  }
  .fit-compact-line {
    font-family: monospace;
    font-size: 0.74rem;
    color: #eee;
    white-space: normal;
    word-break: break-word;
    line-height: 1.4;
  }
  .fit-compact-line strong {
    color: #ffd600;
  }

  .help-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #1e1e1e;
    border: 2px solid #00e5ff;
    border-radius: 8px;
    padding: 20px;
    max-width: 520px;
    width: 90%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    display: none;
    z-index: 100;
  }
  .help-modal h2 { font-size: 1.1rem; color: #00e5ff; margin-bottom: 12px; }
  .help-modal table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .help-modal td { padding: 5px 8px; border-bottom: 1px solid #333; }
  .help-modal td:first-child { font-family: monospace; color: #ffd600; font-weight: bold; width: 42%; }
</style>
</head>
<body>

<header>
  <div class="title">
    <span>⚛ GASPware 2D Matrix Visualizer</span>
    <span class="badge" id="matInfo">Loading...</span>
  </div>
  <div class="hud" id="hudCoords">Det 1 (X): - | Det 2 (Y): - | Counts: -</div>
</header>

<main>
  <aside>
    <!-- 1. Navigation & Scroll (Top) -->
    <div class="control-group">
      <h3>Navigation &amp; Scroll</h3>
      <label>Scroll Zoom Step <span class="range-val" id="scrollSensLabel">4%</span></label>
      <input type="range" id="scrollSensSlider" min="1" max="15" value="4">
      <button id="btnHelp" class="secondary" style="background:#4a148c; margin-top:8px;">Help / Shortcuts [?]</button>
      <button id="btnQuit" class="secondary" style="background:#b71c1c; margin-top:5px;">Quit Viewer [Q]</button>
    </div>

    <!-- 2. 1D Spectrum Fitting -->
    <div class="control-group">
      <h3 style="color: #00e5ff;">1D Histogram Peak Fit</h3>
      <label>1D Peak Function</label>
      <select id="fitTypeSelect1D">
        <option value="gaussian" selected>Gaussian (Symmetric)</option>
        <option value="gaussian_tail">Gaussian + Left Tail (RadWare)</option>
        <option value="hypermet">Hypermet (Convolved Tail + Step)</option>
      </select>

      <label style="margin-top: 6px;">1D Fit Region <span class="range-val" id="fwhmMultLabel1D">4.0× FWHM</span></label>
      <input type="range" id="fwhmMultSlider1D" min="1.0" max="10.0" step="0.5" value="4.0">
      <small style="color: #888; font-size: 0.68rem; display: block; margin-top: 3px;">
        Fits 1D histogram peaks on Det 1 or Det 2 (Ctrl+Click on spectrum or [G]).
      </small>
    </div>

    <!-- 3. 2D Coincidence Fitting -->
    <div class="control-group">
      <h3 style="color: #ffd600;">2D Coincidence Peak Fit</h3>
      <label>2D Peak Function</label>
      <select id="fitTypeSelect2D">
        <option value="gaussian" selected>Gaussian (Symmetric)</option>
        <option value="gaussian_tail">Gaussian + Left Tail (RadWare)</option>
        <option value="hypermet">Hypermet (Convolved Tail + Step)</option>
      </select>

      <label style="margin-top: 6px;">2D ROI Half-Width <span class="range-val" id="roiWidthLabel2D">±16 ch</span></label>
      <input type="range" id="roiWidthSlider2D" min="6" max="36" step="2" value="16">

      <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #333;">
        <label>Results Verbosity</label>
        <select id="fitVerbositySelect">
          <option value="compact" selected>Compact (1 line / axis)</option>
          <option value="detailed">Detailed (Full breakdown)</option>
        </select>
        <small style="color: #888; font-size: 0.68rem; display: block; margin-top: 3px;">
          Gamba &amp; Morh&aacute;c 4-component BG decomposition (Ctrl+Click on 2D or [G]).
        </small>
      </div>
    </div>

    <!-- 4. Color & 1D Projections -->
    <div class="control-group">
      <h3>Color &amp; Projections</h3>
      <label>Colormap [C]</label>
      <select id="cmapSelect">
        <option value="turbo" selected>Turbo (Rainbow Pro)</option>
        <option value="viridis">Viridis</option>
        <option value="plasma">Plasma</option>
        <option value="inferno">Inferno</option>
        <option value="hot">Hot Thermal</option>
        <option value="jet">Jet Classic</option>
        <option value="gray">Grayscale</option>
      </select>

      <label style="margin-top: 6px;">2D Scale Mode [1, 2, 4]</label>
      <select id="scaleSelect">
        <option value="log" selected>Logarithmic (LogNorm)</option>
        <option value="sqrt">Power / Sqrt</option>
        <option value="linear">Linear</option>
      </select>

      <label style="margin-top: 6px;">Max Contrast <span class="range-val" id="vmaxLabel">500</span></label>
      <input type="range" id="vmaxSlider" min="0" max="1000" step="1" value="500">

      <label style="margin-top: 6px;">Min Threshold <span class="range-val" id="vminLabel">1</span></label>
      <input type="range" id="vminSlider" min="0" max="50" value="1">

      <label style="margin-top: 6px;">1D Energy / Channel Range</label>
      <select id="projRangeSelect">
        <option value="synced" selected>Synced with 2D Window Zoom</option>
        <option value="full">Full 0..4095 Range</option>
      </select>

      <label style="margin-top: 6px;">1D Vertical Scale</label>
      <select id="projScaleSelect">
        <option value="linear" selected>Linear</option>
        <option value="log">Logarithmic</option>
      </select>
    </div>
  </aside>

  <div class="workspace">
    <div class="panel-2d">
      <canvas id="canvas2d" tabindex="0"></canvas>
      <div class="panel-2d-footer">
        <span id="overlayStatus">Limits: [L: - | R: - | D: - | U: -] &nbsp;•&nbsp; 'E' Expand &nbsp;•&nbsp; 'F' Full View</span>
        <div style="display: flex; gap: 8px; align-items: center;">
          <button id="btnPrintPDF2D" style="width: auto; padding: 2px 8px; font-size: 0.72rem; background: #1b5e20; border-color: #4caf50; color: #fff;">📄 Print PDF</button>
          <span style="color: #777;">Click &amp; Drag to box-zoom</span>
        </div>
      </div>
    </div>
    <div class="panel-1d-container">
      <div class="panel-1d">
        <div class="panel-1d-header">
          <span class="panel-1d-title" id="specTitleX" style="color: #00e5ff;">Det 1 (X Projection, Sliced Y)</span>
          <div style="display: flex; gap: 6px;">
            <button id="btnPrintPDF1DX" style="width: auto; padding: 2px 7px; font-size: 0.72rem; background: #1b5e20; border-color: #4caf50; color: #fff;">📄 Print PDF</button>
            <button id="btnExport1DX" style="width: auto; padding: 2px 7px; font-size: 0.72rem;">Export .dat</button>
          </div>
        </div>
        <canvas id="canvas1dX"></canvas>
      </div>
      <div class="panel-1d">
        <div class="panel-1d-header">
          <span class="panel-1d-title" id="specTitleY" style="color: #ff9800;">Det 2 (Y Projection, Sliced X)</span>
          <div style="display: flex; gap: 6px;">
            <button id="btnPrintPDF1DY" style="width: auto; padding: 2px 7px; font-size: 0.72rem; background: #1b5e20; border-color: #4caf50; color: #fff;">📄 Print PDF</button>
            <button id="btnExport1DY" style="width: auto; padding: 2px 7px; font-size: 0.72rem;">Export .dat</button>
          </div>
        </div>
        <canvas id="canvas1dY"></canvas>
      </div>

      <!-- 1D Fit Results Card -->
      <div class="fit-results-card fit-card-1d" id="fitCard1D">
        <div class="fit-header" style="color: #00e5ff;">
          <span id="fitTitle1D">⚛ 1D Peak Fit Results</span>
          <div style="display: flex; gap: 6px;">
            <button class="fit-btn-toggle" id="btnToggleDetails1D">▾ Details</button>
            <button class="fit-btn-close" id="btnCloseFit1D">✕ Clear 1D Fit</button>
          </div>
        </div>
        <div class="fit-compact-box" id="fitCompactBox1D">
          <div class="fit-compact-line" id="fitCompactLine1D">-</div>
        </div>
        <div class="fit-grid" id="fitGrid1D" style="display: none; margin-top: 6px; padding-top: 6px; border-top: 1px solid #333;">
          <div class="fit-item"><span class="lbl">Centroid:</span><span class="val" id="fitCentroid1D">-</span></div>
          <div class="fit-item"><span class="lbl">Peak Area:</span><span class="val" id="fitArea1D">-</span></div>
          <div class="fit-item"><span class="lbl">FWHM:</span><span class="val" id="fitFWHM1D">-</span></div>
          <div class="fit-item"><span class="lbl">Amplitude:</span><span class="val" id="fitAmp1D">-</span></div>
          <div class="fit-item" id="fitTailItem1D"><span class="lbl">Left Tail (α):</span><span class="val" id="fitTail1D">-</span></div>
          <div class="fit-item" id="fitHypermetItem1D" style="display: none;"><span class="lbl">Hypermet (fT/β):</span><span class="val" id="fitHypermet1D">-</span></div>
          <div class="fit-item"><span class="lbl">Gross / Bg:</span><span class="val" id="fitGrossBg1D">-</span></div>
          <div class="fit-item"><span class="lbl">Reduced χ²:</span><span class="val" id="fitChi21D">-</span></div>
        </div>
      </div>

      <!-- 2D Coincidence Fit Results Card -->
      <div class="fit-results-card fit-card-2d" id="fitCard2D">
        <div class="fit-header" style="color: #ffd600;">
          <span id="fitTitle2D">⚛ 2D Coincidence Peak Fit</span>
          <div style="display: flex; gap: 6px;">
            <button class="fit-btn-toggle" id="btnToggleDetails2D">▾ Details</button>
            <button class="fit-btn-close" id="btnCloseFit2D">✕ Clear 2D Fit</button>
          </div>
        </div>
        <div class="fit-compact-box" id="fitCompactBox2D">
          <div class="fit-compact-line" id="fitCompactLine2D_X">-</div>
          <div class="fit-compact-line" id="fitCompactLine2D_Y">-</div>
          <div class="fit-compact-line" id="fitCompactLine2D_Gamba" style="color: #81c784; font-size: 0.72rem; margin-top: 3px;">-</div>
        </div>
        <div class="fit-grid" id="fitGrid2D" style="display: none; margin-top: 6px; padding-top: 6px; border-top: 1px solid #333;">
          <div class="fit-item"><span class="lbl">2D Centroid:</span><span class="val" id="fitCentroid2D">-</span></div>
          <div class="fit-item"><span class="lbl">Fitted Net Vol:</span><span class="val" id="fitVolume2D" style="color: #4caf50;">-</span></div>
          <div class="fit-item"><span class="lbl">Gamba Gate Net:</span><span class="val" id="fitGambaNet2D" style="color: #81c784;">-</span></div>
          <div class="fit-item"><span class="lbl">Peak/Total BG (Π):</span><span class="val" id="fitPiRatio2D" style="color: #64b5f6;">-</span></div>
          <div class="fit-item"><span class="lbl">FWHM (X / Y):</span><span class="val" id="fitFWHM2D">-</span></div>
          <div class="fit-item"><span class="lbl">Amplitude (H):</span><span class="val" id="fitAmp2D">-</span></div>
          <div class="fit-item" id="fitTailItem2D"><span class="lbl">Left Tails (α):</span><span class="val" id="fitTail2D">-</span></div>
          <div class="fit-item" id="fitHypermetItem2D" style="display: none;"><span class="lbl">Hypermet Tails:</span><span class="val" id="fitHypermet2D">-</span></div>
          <div class="fit-item"><span class="lbl">2D Cont. (bg|bg):</span><span class="val" id="fitContBg2D">-</span></div>
          <div class="fit-item"><span class="lbl">Cross-Ridges:</span><span class="val" id="fitRidges2D">-</span></div>
          <div class="fit-item"><span class="lbl">Gross / Total BG:</span><span class="val" id="fitGrossBg2D">-</span></div>
          <div class="fit-item"><span class="lbl">Reduced χ² / NDF:</span><span class="val" id="fitChi22D">-</span></div>
        </div>
      </div>
    </div>
  </div>
</main>

<div class="help-modal" id="helpModal">
  <h2>GASPware cmat Navigation Shortcuts</h2>
  <table>
    <tr><td>Click & Drag (2D)</td><td>Box Zoom into 2D rectangle</td></tr>
    <tr><td>Click & Drag (1D)</td><td>Box Zoom on 1D spectrum X-axis (syncs 2D &amp; other 1D spectrum)</td></tr>
    <tr><td>Mouse Wheel (2D)</td><td>Zoom In / Out centered on crosshair in equal steps</td></tr>
    <tr><td>Mouse Wheel (1D)</td><td>Zoom In / Out on Y-axis (fixed Ymin, adjusts Ymax)</td></tr>
    <tr><td>Double Click (1D)</td><td>Reset 1D Y-axis scale to default auto-scale</td></tr>
    <tr><td>Ctrl / Cmd + Click (2D)</td><td>2D Coincidence Peak Fit (Gaussian/HPGe + BG &amp; Random Subtraction)</td></tr>
    <tr><td>Ctrl / Cmd + Click (1D)</td><td>1D Histogram Peak Fit + Linear Background (Det 1 or Det 2)</td></tr>
    <tr><td>G / g</td><td>Fit peak at current cursor position (2D coincidence or 1D histogram)</td></tr>
    <tr><td>= (Equals) / +</td><td>Clear active peak fits (1D and 2D)</td></tr>
    <tr><td>Shift + Arrows (←, →, ↓, ↑)</td><td>Pan 2D matrix view in steps (uses Scroll Sensitivity)</td></tr>
    <tr><td>Left Arrow (←)</td><td>Set Left limit (Xmin) at cursor</td></tr>
    <tr><td>Right Arrow (→)</td><td>Set Right limit (Xmax) at cursor</td></tr>
    <tr><td>Down Arrow (↓)</td><td>Set Down limit (Ymin) at cursor</td></tr>
    <tr><td>Up Arrow (↑)</td><td>Set Up limit (Ymax) at cursor</td></tr>
    <tr><td>E / e</td><td>Expand / Zoom into set limits</td></tr>
    <tr><td>F / f</td><td>Full matrix view (zoom out)</td></tr>
    <tr><td>1 / 2 / 4 (or L)</td><td>Switch Linear, Sqrt, Log color scale</td></tr>
    <tr><td>C / c</td><td>Cycle colormaps</td></tr>
    <tr><td>Q / q</td><td>Quit viewer (close browser tab and stop terminal server)</td></tr>
    <tr><td>Esc / ?</td><td>Close Help</td></tr>
  </table>
  <button onclick="document.getElementById('helpModal').style.display='none'" style="margin-top: 15px;">Close</button>
</div>

<script>
  let metadata = null;
  let currentProjSpecX = null;
  let currentProjSpecY = null;
  let currentProjSlice = { x0: 0, x1: 4096, y0: 0, y1: 4096 };

  // Viewport in channel coordinates
  let view = { x0: 0, x1: 4096, y0: 0, y1: 4096 };

  // Scroll Zoom Step (Fraction per wheel event)
  let scrollSensitivity = 0.04;

  // 1D Y-Axis Zoom Multipliers: { 0: Det 1 (X), 1: Det 2 (Y) }
  let zoom1DY = { 0: 1.0, 1: 1.0 };

  // GASPware Markers [Left, Right, Down, Up]
  let markers = { left: null, right: null, down: null, up: null };

  // 2D Mouse / Drag State
  let isBoxZooming = false;
  let isMouseOver2D = false;
  let dragStartPos = { x: 0, y: 0 };
  let mouseCurrentPos = { x: 0, y: 0 };
  let cursorChannel = { x: 0, y: 0 };

  // 1D Box Zooming State: { 0: Det 1 (X), 1: Det 2 (Y) }
  let is1DBoxZooming = { 0: false, 1: false };
  let drag1DStartPos = { 0: { x: 0, y: 0 }, 1: { x: 0, y: 0 } };
  let drag1DCurrentPos = { 0: { x: 0, y: 0 }, 1: { x: 0, y: 0 } };

  // Hovered channels for real-time inspection
  let cursor1DChannelX = null;
  let cursor1DChannelY = null;

  const canvas2d = document.getElementById('canvas2d');
  const ctx2d = canvas2d.getContext('2d');
  const canvas1dX = document.getElementById('canvas1dX');
  const ctx1dX = canvas1dX.getContext('2d');
  const canvas1dY = document.getElementById('canvas1dY');
  const ctx1dY = canvas1dY.getContext('2d');

  // Tile Data & Shape
  let currentTileData = null;
  let currentTileMax = 100;
  let currentVmax = 500;
  let maxCountGlobal = 50000;
  let tileW = 0, tileH = 0;

  function vmaxSliderToValue(sliderVal) {
    const minVal = 2;
    const maxVal = Math.max(minVal + 1, maxCountGlobal);
    const logMin = Math.log10(minVal);
    const logMax = Math.log10(maxVal);
    const frac = Math.max(0, Math.min(1000, sliderVal)) / 1000.0;
    const val = Math.round(Math.pow(10, logMin + frac * (logMax - logMin)));
    return Math.max(minVal, Math.min(maxVal, val));
  }

  function valueToVmaxSlider(val) {
    const minVal = 2;
    const maxVal = Math.max(minVal + 1, maxCountGlobal);
    const logMin = Math.log10(minVal);
    const logMax = Math.log10(maxVal);
    const clamped = Math.max(minVal, Math.min(maxVal, val));
    const frac = (Math.log10(clamped) - logMin) / (logMax - logMin);
    return Math.round(Math.max(0, Math.min(1, frac)) * 1000);
  }

  // Plot Margins
  const margin2D = { left: 55, bottom: 40, top: 12, right: 15 };
  const margin1D = { left: 55, bottom: 28, top: 10, right: 14 };

  // Precalculated 256-entry Uint32 Color LUT
  let colorLUT = new Uint32Array(256);

  function chToEnergy(ch) {
    if (!metadata || !metadata.cal) return ch;
    const a0 = metadata.cal[0] || 0.0;
    const a1 = metadata.cal[1] || 1.0;
    const a2 = metadata.cal[2] || 0.0;
    return a0 + a1 * ch + a2 * (ch * ch);
  }

  function getPlotRect2D() {
    return {
      x: margin2D.left,
      y: margin2D.top,
      w: Math.max(10, canvas2d.width - margin2D.left - margin2D.right),
      h: Math.max(10, canvas2d.height - margin2D.top - margin2D.bottom)
    };
  }

  function getPlotRect1D(canvas) {
    return {
      x: margin1D.left,
      y: margin1D.top,
      w: Math.max(10, canvas.width - margin1D.left - margin1D.right),
      h: Math.max(10, canvas.height - margin1D.top - margin1D.bottom)
    };
  }

  function chToPx2D(chX, chY) {
    const pr = getPlotRect2D();
    const px = pr.x + ((chX - view.x0) / (view.x1 - view.x0)) * pr.w;
    const py = pr.y + pr.h - ((chY - view.y0) / (view.y1 - view.y0)) * pr.h;
    return { x: px, y: py };
  }

  function pxToCh2D(px, py) {
    const pr = getPlotRect2D();
    const chX = view.x0 + ((px - pr.x) / pr.w) * (view.x1 - view.x0);
    const chY = view.y0 + ((pr.y + pr.h - py) / pr.h) * (view.y1 - view.y0);
    return { x: chX, y: chY };
  }

  function rgba32(r, g, b, a = 255) {
    return (a << 24) | (b << 16) | (g << 8) | r;
  }

  function updateColorLUT() {
    const cmap = document.getElementById('cmapSelect').value;
    for (let i = 0; i < 256; i++) {
      let t = i / 255.0;
      let r, g, b;
      if (cmap === 'turbo') {
        r = Math.max(0, Math.min(255, (34.61 + t * (1172.33 + t * (-10791.3 + t * (33300.1 + t * (-38394.4 + t * 14825.2)))))));
        g = Math.max(0, Math.min(255, (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56)))))));
        b = Math.max(0, Math.min(255, (27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66)))))));
      } else if (cmap === 'viridis') {
        r = Math.min(255, Math.max(0, (0.27 + t * (0.01 + t * (-0.8 + t * 1.5))) * 255));
        g = Math.min(255, Math.max(0, (0.00 + t * (1.35 + t * (0.15 - t * 0.5))) * 255));
        b = Math.min(255, Math.max(0, (0.33 + t * (1.30 - t * (2.8 - t * 1.2))) * 255));
      } else if (cmap === 'plasma') {
        r = Math.min(255, Math.max(0, (0.05 + t * 1.5) * 255));
        g = Math.min(255, Math.max(0, (0.01 + t * 0.9) * 255));
        b = Math.min(255, Math.max(0, (0.53 + (1-t) * 0.4) * 255));
      } else if (cmap === 'inferno') {
        r = Math.min(255, Math.max(0, (0.00 + t * (0.5 + t * (1.5 - t * 1.0))) * 255));
        g = Math.min(255, Math.max(0, (0.00 + t * (0.1 + t * (1.2 + t * 0.5))) * 255));
        b = Math.min(255, Math.max(0, (0.00 + t * (0.9 - t * (0.6 + t * 0.3))) * 255));
      } else if (cmap === 'hot') {
        r = Math.min(255, Math.max(0, t * 3.0 * 255));
        g = Math.min(255, Math.max(0, (t * 3.0 - 1.0) * 255));
        b = Math.min(255, Math.max(0, (t * 3.0 - 2.0) * 255));
      } else if (cmap === 'jet') {
        let rVal = Math.min(255, Math.max(0, (1.5 - Math.abs(4 * t - 3)) * 255));
        let gVal = Math.min(255, Math.max(0, (1.5 - Math.abs(4 * t - 2)) * 255));
        let bVal = Math.min(255, Math.max(0, (1.5 - Math.abs(4 * t - 1)) * 255));
        r = rVal; g = gVal; b = bVal;
      } else { // Grayscale
        r = g = b = Math.min(255, Math.max(0, t * 255));
      }
      colorLUT[i] = rgba32(r | 0, g | 0, b | 0);
    }
  }

  function resizeCanvases() {
    const p2d = document.querySelector('.panel-2d');
    const ftr = document.querySelector('.panel-2d-footer');
    const ftrH = ftr ? ftr.offsetHeight : 26;
    const w2d = p2d.clientWidth;
    const h2d = p2d.clientHeight - ftrH;
    if (w2d > 0 && h2d > 0 && (canvas2d.width !== w2d || canvas2d.height !== h2d)) {
      canvas2d.width = w2d;
      canvas2d.height = h2d;
    }

    const p1dX = canvas1dX.parentElement;
    if (p1dX) {
      const w1d = p1dX.clientWidth - 16;
      const h1d = p1dX.clientHeight - 32;
      if (w1d > 0 && h1d > 0 && (canvas1dX.width !== w1d || canvas1dX.height !== h1d)) {
        canvas1dX.width = w1d;
        canvas1dX.height = h1d;
      }
    }

    const p1dY = canvas1dY.parentElement;
    if (p1dY) {
      const w1d = p1dY.clientWidth - 16;
      const h1d = p1dY.clientHeight - 32;
      if (w1d > 0 && h1d > 0 && (canvas1dY.width !== w1d || canvas1dY.height !== h1d)) {
        canvas1dY.width = w1d;
        canvas1dY.height = h1d;
      }
    }
  }

  async function init() {
    updateColorLUT();
    resizeCanvases();
    window.addEventListener('resize', () => {
      resizeCanvases();
      render2D();
      renderBoth1DSpectra();
    });

    const res = await fetch('/api/metadata');
    metadata = await res.json();
    document.getElementById('matInfo').innerText = `${metadata.filename} (${metadata.shape[0]}×${metadata.shape[1]})`;
    view = { x0: 0, x1: metadata.shape[0], y0: 0, y1: metadata.shape[1] };

    maxCountGlobal = Math.max(100, metadata.max_count || 1000);
    currentVmax = Math.min(800, maxCountGlobal);
    const vmaxSlider = document.getElementById('vmaxSlider');
    if (vmaxSlider) {
      vmaxSlider.min = "0";
      vmaxSlider.max = "1000";
      vmaxSlider.step = "1";
      vmaxSlider.value = valueToVmaxSlider(currentVmax);
      document.getElementById('vmaxLabel').innerText = currentVmax;
    }

    setupEvents();
    await fetchTileAndRender();
    await fetch1DProjection();
  }

  async function fetchTileAndRender() {
    const pr = getPlotRect2D();
    const reqW = Math.min(1024, Math.max(128, Math.round(pr.w)));
    const reqH = Math.min(1024, Math.max(128, Math.round(pr.h)));

    const url = `/api/tile?x0=${Math.floor(view.x0)}&x1=${Math.ceil(view.x1)}&y0=${Math.floor(view.y0)}&y1=${Math.ceil(view.y1)}&w=${reqW}&h=${reqH}`;
    const res = await fetch(url);
    tileW = parseInt(res.headers.get('X-Shape-W')) || reqW;
    tileH = parseInt(res.headers.get('X-Shape-H')) || reqH;
    const buf = await res.arrayBuffer();

    currentTileData = new Int32Array(buf);
    let maxVal = 0;
    for (let i = 0; i < currentTileData.length; i++) {
      if (currentTileData[i] > maxVal) maxVal = currentTileData[i];
    }
    currentTileMax = Math.max(2, maxVal);

    render2D();
  }

  async function fetch1DProjection() {
    const x0 = Math.floor(Math.max(0, view.x0));
    const x1 = Math.ceil(Math.min(metadata.shape[0], view.x1));
    const y0 = Math.floor(Math.max(0, view.y0));
    const y1 = Math.ceil(Math.min(metadata.shape[1], view.y1));
    currentProjSlice = { x0, x1, y0, y1 };

    const isFullX = (y0 === 0 && y1 >= metadata.shape[1]);
    const isFullY = (x0 === 0 && x1 >= metadata.shape[0]);

    document.getElementById('specTitleX').innerText = isFullX
      ? `Det 1 (Total X Projection)`
      : `Det 1 (X Projection, Sliced Y: [${y0}..${y1}])`;

    document.getElementById('specTitleY').innerText = isFullY
      ? `Det 2 (Total Y Projection)`
      : `Det 2 (Y Projection, Sliced X: [${x0}..${x1}])`;

    const url = `/api/projection_region?x0=${x0}&x1=${x1}&y0=${y0}&y1=${y1}`;
    const res = await fetch(url);
    const data = await res.json();
    currentProjSpecX = data.specX;
    currentProjSpecY = data.specY;
    renderBoth1DSpectra();
  }

  function render2D() {
    const cw = canvas2d.width, ch = canvas2d.height;
    ctx2d.clearRect(0, 0, cw, ch);
    const pr = getPlotRect2D();

    // 1. Draw Density Heatmap
    if (currentTileData && tileW > 0 && tileH > 0) {
      const imgData = ctx2d.createImageData(tileW, tileH);
      const buf32 = new Uint32Array(imgData.data.buffer);

      const vmin = parseInt(document.getElementById('vminSlider').value) || 0;
      let vmax = currentVmax || 100;
      if (vmax <= vmin) vmax = vmin + 1;
      const scaleMode = document.getElementById('scaleSelect').value;

      const isLog = (scaleMode === 'log');
      const isSqrt = (scaleMode === 'sqrt');
      const logMin = Math.log(Math.max(1, vmin));
      const logScale = 255.0 / (Math.log(Math.max(2, vmax)) - logMin);
      const sqrtScale = 255.0 / Math.sqrt(Math.max(1, vmax));
      const linScale = 255.0 / Math.max(1, vmax - vmin);

      for (let i = 0; i < currentTileData.length; i++) {
        let val = currentTileData[i];
        if (val <= vmin) {
          buf32[i] = 0xFF000000;
        } else {
          let idx;
          if (isLog) idx = (Math.log(val) - logMin) * logScale;
          else if (isSqrt) idx = Math.sqrt(val) * sqrtScale;
          else idx = (val - vmin) * linScale;
          idx = Math.max(0, Math.min(255, idx | 0));
          buf32[i] = colorLUT[idx];
        }
      }

      const offCanvas = document.createElement('canvas');
      offCanvas.width = tileW;
      offCanvas.height = tileH;
      offCanvas.getContext('2d').putImageData(imgData, 0, 0);

      ctx2d.save();
      ctx2d.beginPath();
      ctx2d.rect(pr.x, pr.y, pr.w, pr.h);
      ctx2d.clip();
      ctx2d.imageSmoothingEnabled = false;
      ctx2d.translate(pr.x, pr.y + pr.h);
      ctx2d.scale(1, -1);
      ctx2d.drawImage(offCanvas, 0, 0, pr.w, pr.h);
      ctx2d.restore();
    }

    // 2. Axes Border & Grid Ticks
    ctx2d.strokeStyle = '#444';
    ctx2d.lineWidth = 1;
    ctx2d.strokeRect(pr.x, pr.y, pr.w, pr.h);
    ctx2d.fillStyle = '#999';
    ctx2d.font = '10px monospace';
    ctx2d.textAlign = 'center';

    const numTicks = 6;
    for (let i = 0; i <= numTicks; i++) {
      const frac = i / numTicks;
      const chX = Math.round(view.x0 + frac * (view.x1 - view.x0));
      const chY = Math.round(view.y0 + frac * (view.y1 - view.y0));
      const px = pr.x + frac * pr.w;
      const py = pr.y + pr.h - frac * pr.h;
      ctx2d.beginPath();
      ctx2d.moveTo(px, pr.y + pr.h); ctx2d.lineTo(px, pr.y + pr.h + 5); ctx2d.stroke();
      ctx2d.fillText(chX.toString(), px, pr.y + pr.h + 16);
      ctx2d.beginPath();
      ctx2d.moveTo(pr.x, py); ctx2d.lineTo(pr.x - 5, py); ctx2d.stroke();
      ctx2d.textAlign = 'right';
      ctx2d.fillText(chY.toString(), pr.x - 8, py + 3);
      ctx2d.textAlign = 'center';
    }

    // 3. Limit Markers
    ctx2d.save();
    ctx2d.beginPath();
    ctx2d.rect(pr.x, pr.y, pr.w, pr.h);
    ctx2d.clip();
    ctx2d.lineWidth = 1.5;
    if (markers.left !== null) { let pt = chToPx2D(markers.left, 0); ctx2d.strokeStyle = '#00e5ff'; ctx2d.setLineDash([5, 5]); ctx2d.beginPath(); ctx2d.moveTo(pt.x, pr.y); ctx2d.lineTo(pt.x, pr.y + pr.h); ctx2d.stroke(); }
    if (markers.right !== null) { let pt = chToPx2D(markers.right, 0); ctx2d.strokeStyle = '#00e5ff'; ctx2d.setLineDash([2, 4]); ctx2d.beginPath(); ctx2d.moveTo(pt.x, pr.y); ctx2d.lineTo(pt.x, pr.y + pr.h); ctx2d.stroke(); }
    if (markers.down !== null) { let pt = chToPx2D(0, markers.down); ctx2d.strokeStyle = '#ff4081'; ctx2d.setLineDash([5, 5]); ctx2d.beginPath(); ctx2d.moveTo(pr.x, pt.y); ctx2d.lineTo(pr.x + pr.w, pt.y); ctx2d.stroke(); }
    if (markers.up !== null) { let pt = chToPx2D(0, markers.up); ctx2d.strokeStyle = '#ff4081'; ctx2d.setLineDash([2, 4]); ctx2d.beginPath(); ctx2d.moveTo(pr.x, pt.y); ctx2d.lineTo(pr.x + pr.w, pt.y); ctx2d.stroke(); }
    ctx2d.setLineDash([]);
    ctx2d.restore();

    // 4. Box Zoom
    if (isBoxZooming) {
      const rx = Math.min(dragStartPos.x, mouseCurrentPos.x);
      const ry = Math.min(dragStartPos.y, mouseCurrentPos.y);
      const rw = Math.abs(mouseCurrentPos.x - dragStartPos.x);
      const rh = Math.abs(mouseCurrentPos.y - dragStartPos.y);
      ctx2d.fillStyle = 'rgba(0, 229, 255, 0.28)';
      ctx2d.strokeStyle = '#00e5ff';
      ctx2d.lineWidth = 1.8;
      ctx2d.fillRect(rx, ry, rw, rh);
      ctx2d.strokeRect(rx, ry, rw, rh);
    }

    // 5. Draw 2D Coincidence Peak Fit Indicator & FWHM Ellipse
    if (activeFit2D && activeFit2D.success) {
      const f2d = activeFit2D;
      const ptC = chToPx2D(f2d.centroid_x_ch + 0.5, f2d.centroid_y_ch + 0.5);
      const ptL = chToPx2D(f2d.roi_x_min, f2d.roi_y_max + 1);
      const ptR = chToPx2D(f2d.roi_x_max + 1, f2d.roi_y_min);

      // A. ROI box
      ctx2d.fillStyle = 'rgba(255, 214, 0, 0.08)';
      ctx2d.fillRect(Math.min(ptL.x, ptR.x), Math.min(ptL.y, ptR.y), Math.abs(ptR.x - ptL.x), Math.abs(ptR.y - ptL.y));
      ctx2d.strokeStyle = 'rgba(255, 214, 0, 0.6)';
      ctx2d.lineWidth = 1;
      ctx2d.setLineDash([4, 4]);
      ctx2d.strokeRect(Math.min(ptL.x, ptR.x), Math.min(ptL.y, ptR.y), Math.abs(ptR.x - ptL.x), Math.abs(ptR.y - ptL.y));
      ctx2d.setLineDash([]);

      // B. FWHM Ellipse
      const ptXRadius = chToPx2D(f2d.centroid_x_ch + 0.5 + (f2d.fwhm_x_ch / 2), f2d.centroid_y_ch + 0.5);
      const ptYRadius = chToPx2D(f2d.centroid_x_ch + 0.5, f2d.centroid_y_ch + 0.5 + (f2d.fwhm_y_ch / 2));
      const radX = Math.abs(ptXRadius.x - ptC.x);
      const radY = Math.abs(ptYRadius.y - ptC.y);

      ctx2d.beginPath();
      ctx2d.ellipse(ptC.x, ptC.y, Math.max(2, radX), Math.max(2, radY), 0, 0, 2 * Math.PI);
      ctx2d.strokeStyle = '#ffd600';
      ctx2d.lineWidth = 2.0;
      ctx2d.stroke();

      // C. Crosshair at Centroid
      ctx2d.strokeStyle = '#ff007f';
      ctx2d.lineWidth = 1.6;
      ctx2d.beginPath();
      ctx2d.moveTo(ptC.x - 8, ptC.y); ctx2d.lineTo(ptC.x + 8, ptC.y);
      ctx2d.moveTo(ptC.x, ptC.y - 8); ctx2d.lineTo(ptC.x, ptC.y + 8);
      ctx2d.stroke();

      // D. Callout badge with Centroid & Volume
      const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
      const posStr = isCal
        ? `(${f2d.centroid_x_e}, ${f2d.centroid_y_e}) keV`
        : `(${f2d.centroid_x_ch}, ${f2d.centroid_y_ch}) ch`;
      const lbl = `⚛ ${posStr} | Vol: ${f2d.volume.toLocaleString()} cts`;
      ctx2d.font = 'bold 9px monospace';
      const tW = ctx2d.measureText(lbl).width;
      ctx2d.fillStyle = 'rgba(15, 18, 24, 0.92)';
      ctx2d.fillRect(ptC.x + 8, ptC.y - 17, tW + 8, 16);
      ctx2d.strokeStyle = '#ffd600';
      ctx2d.lineWidth = 1;
      ctx2d.strokeRect(ptC.x + 8, ptC.y - 17, tW + 8, 16);
      ctx2d.fillStyle = '#ffd600';
      ctx2d.textAlign = 'left';
      ctx2d.fillText(lbl, ptC.x + 12, ptC.y - 5);
    }
    updateMarkerStatus();
  }

  let activeFit1D = { 0: null, 1: null };
  let activeFit2D = null;

  async function requestPeakFit2D(x, y) {
    const fitType = document.getElementById('fitTypeSelect2D') ? document.getElementById('fitTypeSelect2D').value : 'gaussian';
    const roiHW = document.getElementById('roiWidthSlider2D') ? parseInt(document.getElementById('roiWidthSlider2D').value, 10) : 16;
    const verbosity = document.getElementById('fitVerbositySelect') ? document.getElementById('fitVerbositySelect').value : 'compact';

    const res = await fetch(`/api/fit_peak_2d?x=${x}&y=${y}&fit_type=${fitType}&roi_half_width=${roiHW}&verbosity=${verbosity}`);
    const data = await res.json();
    if (data.success) {
      activeFit2D = data;

      const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
      const modelTag = (data.fit_type === 'hypermet') ? '2D Hypermet Fit (Convolved Tail + Step)' : ((data.fit_type === 'gaussian_tail') ? '2D Peak Fit (Left-Tail RadWare)' : '2D Gaussian Fit (Symmetric)');
      const posTag = isCal
        ? `(${data.centroid_x_e}, ${data.centroid_y_e}) keV`
        : `(${data.centroid_x_ch}, ${data.centroid_y_ch}) ch`;

      document.getElementById('fitTitle2D').innerText = `⚛ ${modelTag}: ${posTag}`;

      // Compact format (one simple line for each axis + Gamba net summary)
      const volStr = `${data.volume.toLocaleString()}(${data.volume_err.toLocaleString()})`;
      const lineX = isCal
        ? `<strong>Det 1 (X):</strong> Centroid: ${data.centroid_x_e}(${data.centroid_x_e_err}) keV &nbsp;&nbsp; Area: ${volStr} counts &nbsp;&nbsp; FWHM: ${data.fwhm_x_e}(${data.fwhm_x_e_err}) keV`
        : `<strong>Det 1 (X):</strong> Centroid: ${data.centroid_x_ch}(${data.centroid_x_ch_err}) ch &nbsp;&nbsp; Area: ${volStr} counts &nbsp;&nbsp; FWHM: ${data.fwhm_x_ch}(${data.fwhm_x_ch_err}) ch`;
      const lineY = isCal
        ? `<strong>Det 2 (Y):</strong> Centroid: ${data.centroid_y_e}(${data.centroid_y_e_err}) keV &nbsp;&nbsp; Area: ${volStr} counts &nbsp;&nbsp; FWHM: ${data.fwhm_y_e}(${data.fwhm_y_e_err}) keV`
        : `<strong>Det 2 (Y):</strong> Centroid: ${data.centroid_y_ch}(${data.centroid_y_ch_err}) ch &nbsp;&nbsp; Area: ${volStr} counts &nbsp;&nbsp; FWHM: ${data.fwhm_y_ch}(${data.fwhm_y_ch_err}) ch`;
      const lineGamba = `<strong>Gamba Gate Net (p|p<sup>t</sup>):</strong> ${data.gamba_net.toLocaleString()} ± ${data.gamba_net_err.toLocaleString()} counts &nbsp;&nbsp; <strong>Π Ratio:</strong> ${data.pi_ratio_percent}%`;

      document.getElementById('fitCompactLine2D_X').innerHTML = lineX;
      document.getElementById('fitCompactLine2D_Y').innerHTML = lineY;
      document.getElementById('fitCompactLine2D_Gamba').innerHTML = lineGamba;

      // Detailed parameters grid
      document.getElementById('fitCentroid2D').innerText = isCal
        ? `(${data.centroid_x_e} ± ${data.centroid_x_e_err}, ${data.centroid_y_e} ± ${data.centroid_y_e_err}) keV [ch (${data.centroid_x_ch}, ${data.centroid_y_ch})]`
        : `(${data.centroid_x_ch} ± ${data.centroid_x_ch_err}, ${data.centroid_y_ch} ± ${data.centroid_y_ch_err}) ch`;

      document.getElementById('fitVolume2D').innerText = `${data.volume.toLocaleString()} ± ${data.volume_err.toLocaleString()} counts`;
      document.getElementById('fitGambaNet2D').innerText = `${data.gamba_net.toLocaleString()} ± ${data.gamba_net_err.toLocaleString()} counts`;
      document.getElementById('fitPiRatio2D').innerText = `${data.pi_ratio_percent}% (Π = ${data.pi_ratio})`;

      document.getElementById('fitFWHM2D').innerText = isCal
        ? `X: ${data.fwhm_x_e} ± ${data.fwhm_x_e_err} | Y: ${data.fwhm_y_e} ± ${data.fwhm_y_e_err} keV`
        : `X: ${data.fwhm_x_ch} ± ${data.fwhm_x_ch_err} | Y: ${data.fwhm_y_ch} ± ${data.fwhm_y_ch_err} ch`;

      document.getElementById('fitAmp2D').innerText = `${data.amplitude.toLocaleString()} ± ${data.amplitude_err.toLocaleString()} counts`;

      const tailItem = document.getElementById('fitTailItem2D');
      if (tailItem) {
        if (data.fit_type === 'gaussian_tail' && data.alpha_x !== null && data.alpha_y !== null) {
          tailItem.style.display = 'flex';
          const axStr = data.alpha_x_err ? `${data.alpha_x} ± ${data.alpha_x_err}` : `${data.alpha_x}`;
          const ayStr = data.alpha_y_err ? `${data.alpha_y} ± ${data.alpha_y_err}` : `${data.alpha_y}`;
          document.getElementById('fitTail2D').innerText = `αX = ${axStr} | αY = ${ayStr}`;
        } else {
          tailItem.style.display = 'none';
        }
      }

      const hypItem = document.getElementById('fitHypermetItem2D');
      if (hypItem) {
        if (data.fit_type === 'hypermet' && data.eta_tx !== undefined && data.eta_ty !== undefined) {
          hypItem.style.display = 'flex';
          document.getElementById('fitHypermet2D').innerText = `ηX=${data.eta_tx} (β=${data.beta_x}) | ηY=${data.eta_ty} (β=${data.beta_y})`;
        } else {
          hypItem.style.display = 'none';
        }
      }

      document.getElementById('fitContBg2D').innerText = `${data.cont_counts.toLocaleString()} counts (b0=${data.bg_b0}, bx=${data.bg_bx}, by=${data.bg_by})`;
      document.getElementById('fitRidges2D').innerText = `p|bg (X): ${data.ridge_x} ± ${data.ridge_x_err} (${data.ridge_x_counts} cts) | bg|p (Y): ${data.ridge_y} ± ${data.ridge_y_err} (${data.ridge_y_counts} cts)`;
      document.getElementById('fitGrossBg2D').innerText = `${data.gross_counts.toLocaleString()} gross / ${data.total_bg_counts.toLocaleString()} bg counts`;
      document.getElementById('fitChi22D').innerText = `${data.red_chi2} (χ²=${data.chi2}, NDF=${data.ndf})`;

      updateFitCardsVerbosity();
      document.getElementById('fitCard2D').style.display = 'block';

      document.getElementById('hudCoords').innerText = `2D Fit: Centroid = ${posTag} | Net Volume = ${data.volume} ± ${data.volume_err} | Gamba Net = ${data.gamba_net} (Π=${data.pi_ratio_percent}%) | FWHMs = (${isCal ? data.fwhm_x_e + ', ' + data.fwhm_y_e + ' keV' : data.fwhm_x_ch + ', ' + data.fwhm_y_ch + ' ch'})`;
    } else {
      console.warn("2D Peak fit failed:", data.error);
    }
    render2D();
  }

  async function requestPeakFit(axis, ch) {
    const x0 = Math.floor(Math.max(0, view.x0));
    const x1 = Math.ceil(Math.min(metadata.shape[0], view.x1));
    const y0 = Math.floor(Math.max(0, view.y0));
    const y1 = Math.ceil(Math.min(metadata.shape[1], view.y1));
    const fitType = document.getElementById('fitTypeSelect1D') ? document.getElementById('fitTypeSelect1D').value : 'gaussian';
    const fwhmMult = document.getElementById('fwhmMultSlider1D') ? document.getElementById('fwhmMultSlider1D').value : '4.0';
    const verbosity = document.getElementById('fitVerbositySelect') ? document.getElementById('fitVerbositySelect').value : 'compact';

    const res = await fetch(`/api/fit_peak?axis=${axis}&channel=${ch}&x0=${x0}&x1=${x1}&y0=${y0}&y1=${y1}&fit_type=${fitType}&fwhm_mult=${fwhmMult}&verbosity=${verbosity}`);
    const data = await res.json();
    if (data.success) {
      activeFit1D[axis] = data;
      const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
      const detLabel = (axis === 0) ? 'Det 1 (X)' : 'Det 2 (Y)';
      const modelTag = (data.fit_type === 'hypermet') ? '1D Hypermet Fit (Convolved Tail + Step)' : ((data.fit_type === 'gaussian_tail') ? '1D Peak Fit (Left-Tail RadWare)' : '1D Gaussian Fit');
      document.getElementById('fitTitle1D').innerText = `⚛ ${modelTag} [${detLabel}]: ${isCal ? data.centroid_e + ' keV' : 'ch ' + data.centroid_ch}`;

      // Compact line in format: centroid(err)   area(err)   fwhm(err)
      const areaStr = `${data.area.toLocaleString()}(${data.area_err.toLocaleString()})`;
      const compact1D = isCal
        ? `Centroid: ${data.centroid_e}(${data.centroid_e_err}) keV &nbsp;&nbsp; Area: ${areaStr} counts &nbsp;&nbsp; FWHM: ${data.fwhm_e}(${data.fwhm_e_err}) keV`
        : `Centroid: ${data.centroid_ch}(${data.centroid_ch_err}) ch &nbsp;&nbsp; Area: ${areaStr} counts &nbsp;&nbsp; FWHM: ${data.fwhm_ch}(${data.fwhm_ch_err}) ch`;
      document.getElementById('fitCompactLine1D').innerHTML = compact1D;

      // Detailed parameters grid
      document.getElementById('fitCentroid1D').innerText = isCal
        ? `${data.centroid_e} ± ${data.centroid_e_err} keV (${data.centroid_ch} ± ${data.centroid_ch_err} ch)`
        : `${data.centroid_ch} ± ${data.centroid_ch_err} ch`;
      document.getElementById('fitArea1D').innerText = `${data.area.toLocaleString()} ± ${data.area_err.toLocaleString()} counts`;
      document.getElementById('fitFWHM1D').innerText = isCal
        ? `${data.fwhm_e} ± ${data.fwhm_e_err} keV (${data.fwhm_ch} ± ${data.fwhm_ch_err} ch)`
        : `${data.fwhm_ch} ± ${data.fwhm_ch_err} ch`;
      document.getElementById('fitAmp1D').innerText = `${data.amplitude.toLocaleString()} ± ${data.amplitude_err.toLocaleString()} cts`;

      const tailItem = document.getElementById('fitTailItem1D');
      if (tailItem) {
        if (data.fit_type === 'gaussian_tail' && data.alpha !== null) {
          tailItem.style.display = 'flex';
          document.getElementById('fitTail1D').innerText = `${data.alpha} ± ${data.alpha_err}`;
        } else {
          tailItem.style.display = 'none';
        }
      }

      const hypItem = document.getElementById('fitHypermetItem1D');
      if (hypItem) {
        if (data.fit_type === 'hypermet' && data.tail_area !== undefined && data.tail_area !== null) {
          hypItem.style.display = 'flex';
          document.getElementById('fitHypermet1D').innerText = `fT=${data.tail_area} ± ${data.tail_area_err} (β=${data.tail_slope}, As=${data.step_height})`;
        } else {
          hypItem.style.display = 'none';
        }
      }

      document.getElementById('fitGrossBg1D').innerText = `${data.gross_counts.toLocaleString()} / ${data.bg_counts.toLocaleString()}`;
      document.getElementById('fitChi21D').innerText = `${data.red_chi2} (NDF=${data.ndf})`;

      updateFitCardsVerbosity();
      document.getElementById('fitCard1D').style.display = 'block';

      const hudEnergy = isCal ? ` (${data.centroid_e} keV)` : '';
      document.getElementById('hudCoords').innerText = `1D Fit [${detLabel}]: Centroid = ${data.centroid_ch}${hudEnergy} | Area = ${data.area} ± ${data.area_err} | FWHM = ${isCal ? data.fwhm_e + ' keV' : data.fwhm_ch + ' ch'}`;
    } else {
      console.warn("Peak fit failed:", data.error);
    }
    renderBoth1DSpectra();
  }

  function render1DSpectrum(canvas, ctx, spec, axis, cursorCh) {
    const cw = canvas.width, ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);
    if (!spec || spec.length === 0) return;

    const pr = getPlotRect1D(canvas);
    const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
    const isLog = (document.getElementById('projScaleSelect').value === 'log');

    let chStart = 0;
    let chEnd = spec.length - 1;
    if (isSynced) {
      if (axis === 0) {
        chStart = Math.max(0, Math.floor(view.x0));
        chEnd = Math.min(spec.length - 1, Math.ceil(view.x1));
      } else {
        chStart = Math.max(0, Math.floor(view.y0));
        chEnd = Math.min(spec.length - 1, Math.ceil(view.y1));
      }
    }
    if (chEnd <= chStart) chEnd = chStart + 1;

    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = chStart; i <= chEnd; i++) {
      const v = spec[i];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    if (minVal === Infinity) { minVal = 0; maxVal = 1; }
    if (minVal === maxVal) { maxVal = minVal + 1; }

    const range = maxVal - minVal;
    let yMinDisplay, yMaxDisplay;
    const yMult = zoom1DY[axis] !== undefined ? zoom1DY[axis] : 1.0;

    if (isLog) {
      let logMinVal = Math.log10(Math.max(1, minVal > 0 ? minVal : 1));
      let logMaxVal = Math.log10(Math.max(10, maxVal));
      let logRange = logMaxVal - logMinVal;
      if (logRange <= 0.1) logRange = 1;
      yMinDisplay = Math.max(0, logMinVal - logRange * 0.06);
      let baseYMax = logMaxVal + logRange * 0.08;
      yMaxDisplay = yMinDisplay + (baseYMax - yMinDisplay) * yMult;
    } else {
      yMinDisplay = Math.floor(minVal - range * 0.06);
      if (minVal >= 0 && minVal - range * 0.06 < 0) { yMinDisplay = 0; }
      let baseYMax = Math.ceil(maxVal + range * 0.08);
      yMaxDisplay = yMinDisplay + Math.max(1, (baseYMax - yMinDisplay) * yMult);
    }
    if (yMaxDisplay <= yMinDisplay) yMaxDisplay = yMinDisplay + 1;

    // 5. Plot Frame
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.strokeRect(pr.x, pr.y, pr.w, pr.h);

    ctx.save();
    ctx.beginPath();
    ctx.rect(pr.x, pr.y, pr.w, pr.h);
    ctx.clip();

    const span = chEnd - chStart + 1;
    const strokeColor = (axis === 0) ? '#00e5ff' : '#ff9800';
    const fillColor = (axis === 0) ? 'rgba(0, 229, 255, 0.12)' : 'rgba(255, 152, 0, 0.12)';

    function valToPx(val) {
      let norm;
      if (isLog) {
        let lv = Math.log10(Math.max(1, val));
        norm = (lv - yMinDisplay) / (yMaxDisplay - yMinDisplay);
      } else {
        norm = (val - yMinDisplay) / (yMaxDisplay - yMinDisplay);
      }
      norm = Math.max(0, Math.min(1, norm));
      return pr.y + pr.h * (1 - norm);
    }

    // A. Stepped Histogram
    ctx.beginPath();
    ctx.moveTo(pr.x, pr.y + pr.h);
    for (let i = chStart; i <= chEnd; i++) {
      let xL = pr.x + ((i - chStart) / span) * pr.w;
      let xR = pr.x + ((i + 1 - chStart) / span) * pr.w;
      let y = valToPx(spec[i]);
      ctx.lineTo(xL, y); ctx.lineTo(xR, y);
    }
    ctx.lineTo(pr.x + pr.w, pr.y + pr.h);
    ctx.closePath();
    ctx.fillStyle = fillColor; ctx.fill();

    ctx.beginPath();
    for (let i = chStart; i <= chEnd; i++) {
      let xL = pr.x + ((i - chStart) / span) * pr.w;
      let xR = pr.x + ((i + 1 - chStart) / span) * pr.w;
      let y = valToPx(spec[i]);
      if (i === chStart) ctx.moveTo(xL, y); else ctx.lineTo(xL, y);
      ctx.lineTo(xR, y);
    }
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // B. Fitted Peak & Baseline Background Curves
    const fit = activeFit1D[axis];
    if (fit && fit.curve_x && fit.curve_x.length > 0) {
      if (fit.roi_ch_max >= chStart && fit.roi_ch_min <= chEnd) {
        // ROI Shading
        const xRoiL = pr.x + ((fit.roi_ch_min - chStart) / span) * pr.w;
        const xRoiR = pr.x + ((fit.roi_ch_max + 1 - chStart) / span) * pr.w;
        ctx.fillStyle = (axis === 0) ? 'rgba(0, 229, 255, 0.08)' : 'rgba(255, 152, 0, 0.08)';
        ctx.fillRect(Math.max(pr.x, xRoiL), pr.y, Math.min(pr.w, xRoiR - Math.max(pr.x, xRoiL)), pr.h);

        // Baseline Background Curve (Dashed Magenta)
        if (fit.curve_bg && fit.curve_bg.length > 0) {
          ctx.beginPath();
          ctx.strokeStyle = '#ff4081';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          for (let k = 0; k < fit.curve_x.length; k++) {
            const cx = pr.x + ((fit.curve_x[k] + 0.5 - chStart) / span) * pr.w;
            const cy = valToPx(fit.curve_bg[k]);
            if (k === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
          }
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Gaussian Total Fit Curve (Bright Gold)
        ctx.beginPath();
        ctx.strokeStyle = '#ffd600';
        ctx.lineWidth = 2.2;
        for (let k = 0; k < fit.curve_x.length; k++) {
          const cx = pr.x + ((fit.curve_x[k] + 0.5 - chStart) / span) * pr.w;
          const cy = valToPx(fit.curve_fit[k]);
          if (k === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();

        // Centroid Vertical Line
        const cX = pr.x + ((fit.centroid_ch + 0.5 - chStart) / span) * pr.w;
        if (cX >= pr.x && cX <= pr.x + pr.w) {
          ctx.strokeStyle = '#ffd600';
          ctx.lineWidth = 1.2;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(cX, pr.y);
          ctx.lineTo(cX, pr.y + pr.h);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // C. Hover Bin Highlight
    if (cursorCh !== null && cursorCh >= chStart && cursorCh <= chEnd) {
      let xL = pr.x + ((cursorCh - chStart) / span) * pr.w;
      let xR = pr.x + ((cursorCh + 1 - chStart) / span) * pr.w;
      let val = spec[cursorCh] || 0;
      let y = valToPx(val);

      ctx.fillStyle = 'rgba(255, 214, 0, 0.35)';
      ctx.fillRect(xL, y, Math.max(1.5, xR - xL), (pr.y + pr.h) - y);

      ctx.strokeStyle = '#ffd600';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(xL, y, Math.max(1.5, xR - xL), (pr.y + pr.h) - y);

      let xMid = (xL + xR) / 2;
      ctx.strokeStyle = 'rgba(255, 214, 0, 0.75)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xMid, pr.y);
      ctx.lineTo(xMid, pr.y + pr.h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // D. 1D Drag Box Zoom Selection Overlay
    if (is1DBoxZooming[axis]) {
      const xS = Math.min(drag1DStartPos[axis].x, drag1DCurrentPos[axis].x);
      const xE = Math.max(drag1DStartPos[axis].x, drag1DCurrentPos[axis].x);
      const rX = Math.max(pr.x, xS);
      const rW = Math.min(pr.x + pr.w, xE) - rX;
      if (rW > 1) {
        ctx.fillStyle = (axis === 0) ? 'rgba(0, 229, 255, 0.28)' : 'rgba(255, 152, 0, 0.28)';
        ctx.fillRect(rX, pr.y, rW, pr.h);
        ctx.strokeStyle = (axis === 0) ? '#00e5ff' : '#ff9800';
        ctx.lineWidth = 1.8;
        ctx.strokeRect(rX, pr.y, rW, pr.h);

        const frac0 = Math.max(0, Math.min(1, (rX - pr.x) / pr.w));
        const frac1 = Math.max(0, Math.min(1, (rX + rW - pr.x) / pr.w));
        const c0 = Math.round(chStart + frac0 * (span - 1));
        const c1 = Math.round(chStart + frac1 * (span - 1));
        const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
        const badgeLbl = isCal
          ? `Zoom: ${chToEnergy(c0).toFixed(0)} - ${chToEnergy(c1).toFixed(0)} keV (${c0} - ${c1} ch)`
          : `Zoom: ch ${c0} - ${c1}`;

        ctx.save();
        ctx.font = 'bold 9px monospace';
        const tW = ctx.measureText(badgeLbl).width;
        ctx.fillStyle = 'rgba(20, 20, 20, 0.9)';
        ctx.fillRect(rX, pr.y + 4, tW + 8, 15);
        ctx.strokeStyle = (axis === 0) ? '#00e5ff' : '#ff9800';
        ctx.strokeRect(rX, pr.y + 4, tW + 8, 15);
        ctx.fillStyle = '#ffd600';
        ctx.textAlign = 'left';
        ctx.fillText(badgeLbl, rX + 4, pr.y + 15);
        ctx.restore();
      }
    }
    ctx.restore();

    // E. On-Canvas HUD Badge
    const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
    let hudText;
    const detLabel = (axis === 0) ? 'Det 1 (X)' : 'Det 2 (Y)';
    if (cursorCh !== null && cursorCh >= 0 && cursorCh < spec.length) {
      const eVal = chToEnergy(cursorCh).toFixed(1);
      const cVal = spec[cursorCh] || 0;
      hudText = isCal
        ? `${detLabel} | Ch: ${cursorCh} | ${eVal} keV | Cts: ${cVal}`
        : `${detLabel} | Ch: ${cursorCh} | Cts: ${cVal}`;
    } else {
      hudText = `${detLabel} Spectrum (Ctrl+Click to Fit)`;
    }

    ctx.save();
    ctx.font = 'bold 9px monospace';
    const textW = ctx.measureText(hudText).width;
    const badgeX = pr.x + pr.w - textW - 12;
    const badgeY = pr.y + 3;
    ctx.fillStyle = 'rgba(20, 20, 20, 0.88)';
    ctx.strokeStyle = (cursorCh !== null) ? '#ffd600' : '#444';
    ctx.lineWidth = 1;
    ctx.fillRect(badgeX, badgeY, textW + 8, 15);
    ctx.strokeRect(badgeX, badgeY, textW + 8, 15);
    ctx.fillStyle = (cursorCh !== null) ? '#ffd600' : '#888';
    ctx.textAlign = 'left';
    ctx.fillText(hudText, badgeX + 4, badgeY + 11);
    ctx.restore();

    // E. Axes Ticks on X
    ctx.fillStyle = '#999';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
      const frac = i / numTicks;
      const c = Math.round(chStart + frac * (span - 1));
      const x = pr.x + frac * pr.w;
      ctx.beginPath();
      ctx.moveTo(x, pr.y + pr.h);
      ctx.lineTo(x, pr.y + pr.h + 3);
      ctx.stroke();
      if (isCal) {
        const eStr = chToEnergy(c).toFixed(0);
        ctx.fillText(`${eStr}k (${c})`, x, pr.y + pr.h + 12);
      } else {
        ctx.fillText(c.toString(), x, pr.y + pr.h + 12);
      }
    }
    const xLabel = (axis === 0)
      ? (isCal ? 'Det 1 Energy (keV) / Channel' : 'Det 1 Channel')
      : (isCal ? 'Det 2 Energy (keV) / Channel' : 'Det 2 Channel');
    ctx.fillText(xLabel, pr.x + pr.w / 2, pr.y + pr.h + 22);

    // F. Dynamic Ticks on Y
    ctx.textAlign = 'right';
    const numYTicks = 3;
    for (let j = 0; j <= numYTicks; j++) {
      const frac = j / numYTicks;
      const py = pr.y + pr.h - frac * pr.h;
      ctx.beginPath();
      ctx.moveTo(pr.x, py);
      ctx.lineTo(pr.x - 4, py);
      ctx.stroke();

      let labelVal;
      if (isLog) {
        let logV = yMinDisplay + frac * (yMaxDisplay - yMinDisplay);
        let num = Math.round(Math.pow(10, logV));
        labelVal = (num >= 10000) ? num.toExponential(1) : num.toString();
      } else {
        let v = yMinDisplay + frac * (yMaxDisplay - yMinDisplay);
        labelVal = (range > 40) ? Math.round(v).toString() : v.toFixed(1);
      }
      ctx.fillText(labelVal, pr.x - 6, py + 3);
    }
  }

  function renderBoth1DSpectra() {
    render1DSpectrum(canvas1dX, ctx1dX, currentProjSpecX, 0, cursor1DChannelX);
    render1DSpectrum(canvas1dY, ctx1dY, currentProjSpecY, 1, cursor1DChannelY);
  }

  function updateMarkerStatus() {
    const isCal = metadata && metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
    const lVal = markers.left !== null ? markers.left : Math.round(view.x0);
    const rVal = markers.right !== null ? markers.right : Math.round(view.x1);
    const dVal = markers.down !== null ? markers.down : Math.round(view.y0);
    const uVal = markers.up !== null ? markers.up : Math.round(view.y1);

    const lStr = isCal ? `${lVal} (${chToEnergy(lVal).toFixed(0)}k)` : `${lVal}`;
    const rStr = isCal ? `${rVal} (${chToEnergy(rVal).toFixed(0)}k)` : `${rVal}`;
    const dStr = isCal ? `${dVal} (${chToEnergy(dVal).toFixed(0)}k)` : `${dVal}`;
    const uStr = isCal ? `${uVal} (${chToEnergy(uVal).toFixed(0)}k)` : `${uVal}`;

    const statusEl = document.getElementById('overlayStatus');
    if (statusEl) {
      statusEl.innerHTML = `Limits: [L: <span style="color:#ffd600">${lStr}</span> | R: <span style="color:#ffd600">${rStr}</span> | D: <span style="color:#ffd600">${dStr}</span> | U: <span style="color:#ffd600">${uStr}</span>] &nbsp;•&nbsp; 'E' Expand &nbsp;•&nbsp; 'F' Full`;
    }
  }

  function expandMarkers() {
    let xmin = markers.left !== null ? markers.left : view.x0;
    let xmax = markers.right !== null ? markers.right : view.x1;
    let ymin = markers.down !== null ? markers.down : view.y0;
    let ymax = markers.up !== null ? markers.up : view.y1;
    if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
    if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
    if (xmax - xmin >= 1 || ymax - ymin >= 1) {
      view.x0 = Math.max(0, xmin); view.x1 = Math.min(metadata.shape[0], xmax);
      view.y0 = Math.max(0, ymin); view.y1 = Math.min(metadata.shape[1], ymax);
      markers.left = null; markers.right = null; markers.down = null; markers.up = null;
      updateMarkerStatus(); fetchTileAndRender(); fetch1DProjection();
    }
  }

  function zoomFull() {
    view.x0 = 0; view.x1 = metadata.shape[0]; view.y0 = 0; view.y1 = metadata.shape[1];
    markers.left = null; markers.right = null; markers.down = null; markers.up = null;
    updateMarkerStatus(); fetchTileAndRender(); fetch1DProjection();
  }

  function pan2D(dxRatio, dyRatio) {
    const spanX = view.x1 - view.x0;
    const spanY = view.y1 - view.y0;
    const maxSpanX = metadata ? metadata.shape[0] : 4096;
    const maxSpanY = metadata ? metadata.shape[1] : 4096;

    const shiftX = Math.round(spanX * scrollSensitivity * dxRatio);
    const shiftY = Math.round(spanY * scrollSensitivity * dyRatio);

    let newX0 = view.x0 + shiftX;
    let newX1 = view.x1 + shiftX;
    let newY0 = view.y0 + shiftY;
    let newY1 = view.y1 + shiftY;

    if (newX0 < 0) {
      newX1 += -newX0;
      newX0 = 0;
    }
    if (newX1 > maxSpanX) {
      newX0 -= (newX1 - maxSpanX);
      newX1 = maxSpanX;
    }
    if (newY0 < 0) {
      newY1 += -newY0;
      newY0 = 0;
    }
    if (newY1 > maxSpanY) {
      newY0 -= (newY1 - maxSpanY);
      newY1 = maxSpanY;
    }

    view.x0 = Math.max(0, Math.floor(newX0));
    view.x1 = Math.min(maxSpanX, Math.ceil(newX1));
    view.y0 = Math.max(0, Math.floor(newY0));
    view.y1 = Math.min(maxSpanY, Math.ceil(newY1));

    fetchTileAndRender();
    fetch1DProjection();
  }

  async function updateHoverHUD() {
    const res = await fetch(`/api/value?x=${cursorChannel.x}&y=${cursorChannel.y}`);
    const data = await res.json();
    const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
    const ex = isCal ? ` (${chToEnergy(data.x).toFixed(1)} keV)` : '';
    const ey = isCal ? ` (${chToEnergy(data.y).toFixed(1)} keV)` : '';
    document.getElementById('hudCoords').innerText = `Det 1 (X): ch ${data.x}${ex} | Det 2 (Y): ch ${data.y}${ey} | Counts: ${data.value}`;
  }

  let hoverTimer = null;

  function export1DData(axis) {
    const spec = (axis === 0) ? currentProjSpecX : currentProjSpecY;
    if (!spec) return;
    const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
    let chStart = 0, chEnd = spec.length - 1;
    if (isSynced) {
      if (axis === 0) {
        chStart = Math.max(0, Math.floor(view.x0));
        chEnd = Math.min(spec.length - 1, Math.ceil(view.x1));
      } else {
        chStart = Math.max(0, Math.floor(view.y0));
        chEnd = Math.min(spec.length - 1, Math.ceil(view.y1));
      }
    }

    const detName = (axis === 0) ? 'Det1_X' : 'Det2_Y';
    let content = `# 1D Binned Histogram Export from ${metadata.filename}\n`;
    content += `# Spectrum: ${axis === 0 ? 'Det 1 (X Projection, Slice Y)' : 'Det 2 (Y Projection, Slice X)'}\n`;
    content += `# Sliced 2D Window: X=[${currentProjSlice.x0}..${currentProjSlice.x1}], Y=[${currentProjSlice.y0}..${currentProjSlice.y1}]\n`;
    content += `# Channel Energy_keV Counts\n`;

    for (let i = chStart; i <= chEnd; i++) {
      let e = chToEnergy(i).toFixed(3);
      content += `${i} ${e} ${spec[i]}\n`;
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${metadata.filename}_${detName}_${chStart}_${chEnd}.dat`;
    a.click();
  }

  function printPDF1D(axis) {
    const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
    const isLog = (document.getElementById('projScaleSelect').value === 'log') ? 1 : 0;
    const x0 = Math.floor(Math.max(0, view.x0));
    const x1 = Math.ceil(Math.min(metadata.shape[0], view.x1));
    const y0 = Math.floor(Math.max(0, view.y0));
    const y1 = Math.ceil(Math.min(metadata.shape[1], view.y1));
    const zoomY = zoom1DY[axis] || 1.0;
    const fit = activeFit1D[axis];
    const fitType = document.getElementById('fitTypeSelect1D') ? document.getElementById('fitTypeSelect1D').value : 'gaussian';
    const fwhmMult = document.getElementById('fwhmMultSlider1D') ? document.getElementById('fwhmMultSlider1D').value : '4.0';

    let url = `/api/export_pdf_1d?axis=${axis}&x0=${x0}&x1=${x1}&y0=${y0}&y1=${y1}&is_synced=${isSynced ? 1 : 0}&is_log=${isLog}&zoom_y=${zoomY}`;
    if (fit && fit.centroid_ch !== undefined) {
      url += `&has_fit=1&fit_ch=${fit.centroid_ch}&fit_type=${fitType}&fwhm_mult=${fwhmMult}`;
    }
    window.open(url, '_blank');
  }

  function printPDF2D() {
    const x0 = Math.floor(Math.max(0, view.x0));
    const x1 = Math.ceil(Math.min(metadata.shape[0], view.x1));
    const y0 = Math.floor(Math.max(0, view.y0));
    const y1 = Math.ceil(Math.min(metadata.shape[1], view.y1));
    const cmap = document.getElementById('cmapSelect').value;
    const scale = document.getElementById('scaleSelect').value;
    const vmin = document.getElementById('vminSlider').value;
    const vmax = currentVmax || 100;
    const f2d = activeFit2D;
    const fitType = document.getElementById('fitTypeSelect2D') ? document.getElementById('fitTypeSelect2D').value : 'gaussian';

    let url = `/api/export_pdf_2d?x0=${x0}&x1=${x1}&y0=${y0}&y1=${y1}&cmap=${cmap}&scale=${scale}&vmin=${vmin}&vmax=${vmax}`;
    if (f2d && f2d.centroid_x_ch !== undefined) {
      url += `&has_fit=1&fit_x=${f2d.centroid_x_ch}&fit_y=${f2d.centroid_y_ch}&fit_type=${fitType}`;
    }
    window.open(url, '_blank');
  }

  async function quitViewer() {
    try {
      await fetch('/api/quit');
    } catch (e) {}
    document.body.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #0a0a0a; color: #fff; font-family: monospace;">
        <h2 style="color: #00e5ff; margin-bottom: 8px;">⚛ GASPware Web Viewer Closed</h2>
        <p style="color: #888;">Server stopped. You can now close this browser tab.</p>
      </div>
    `;
    window.close();
  }

  function clearFit1D() {
    activeFit1D[0] = null;
    activeFit1D[1] = null;
    document.getElementById('fitCard1D').style.display = 'none';
    renderBoth1DSpectra();
  }

  function clearFit2D() {
    activeFit2D = null;
    document.getElementById('fitCard2D').style.display = 'none';
    render2D();
  }

  function clearAllFits() {
    clearFit1D();
    clearFit2D();
  }

  function updateFitCardsVerbosity() {
    const verbosity = document.getElementById('fitVerbositySelect') ? document.getElementById('fitVerbositySelect').value : 'compact';
    const isDetailed = (verbosity === 'detailed');

    const g1D = document.getElementById('fitGrid1D');
    const b1D = document.getElementById('btnToggleDetails1D');
    if (g1D) g1D.style.display = isDetailed ? 'grid' : 'none';
    if (b1D) b1D.innerText = isDetailed ? '▴ Compact' : '▾ Details';

    const g2D = document.getElementById('fitGrid2D');
    const b2D = document.getElementById('btnToggleDetails2D');
    if (g2D) g2D.style.display = isDetailed ? 'grid' : 'none';
    if (b2D) b2D.innerText = isDetailed ? '▴ Compact' : '▾ Details';
  }

  function setupEvents() {
    document.getElementById('cmapSelect').addEventListener('change', () => { updateColorLUT(); render2D(); });
    document.getElementById('scaleSelect').addEventListener('change', render2D);
    const vmaxSlider = document.getElementById('vmaxSlider');
    if (vmaxSlider) {
      vmaxSlider.addEventListener('input', (e) => {
        currentVmax = vmaxSliderToValue(parseFloat(e.target.value));
        document.getElementById('vmaxLabel').innerText = currentVmax;
        render2D();
      });
    }
    const vmaxLbl = document.getElementById('vmaxLabel');
    if (vmaxLbl) {
      vmaxLbl.title = "Double-click to auto-optimize to visible max";
      vmaxLbl.style.cursor = "pointer";
      vmaxLbl.addEventListener('dblclick', () => {
        if (currentTileMax && vmaxSlider) {
          if (currentTileMax > maxCountGlobal) maxCountGlobal = currentTileMax;
          currentVmax = currentTileMax;
          vmaxSlider.value = valueToVmaxSlider(currentVmax);
          vmaxLbl.innerText = currentVmax;
          render2D();
        }
      });
    }
    document.getElementById('vminSlider').addEventListener('input', (e) => { document.getElementById('vminLabel').innerText = e.target.value; render2D(); });
    document.getElementById('scrollSensSlider').addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      scrollSensitivity = Math.max(0.01, Math.min(0.20, val / 100.0));
      document.getElementById('scrollSensLabel').innerText = val + '%';
    });
    document.getElementById('btnQuit').addEventListener('click', quitViewer);
    document.getElementById('projRangeSelect').addEventListener('change', renderBoth1DSpectra);
    document.getElementById('projScaleSelect').addEventListener('change', renderBoth1DSpectra);
    const btnFull = document.getElementById('btnFull');
    if (btnFull) btnFull.addEventListener('click', zoomFull);
    const btnExpand = document.getElementById('btnExpand');
    if (btnExpand) btnExpand.addEventListener('click', expandMarkers);
    const btnResetMarkers = document.getElementById('btnResetMarkers');
    if (btnResetMarkers) btnResetMarkers.addEventListener('click', () => {
      markers.left = null; markers.right = null;
      markers.down = null; markers.up = null;
      zoom1DY[0] = 1.0; zoom1DY[1] = 1.0;
      updateMarkerStatus(); render2D(); renderBoth1DSpectra();
    });
    document.getElementById('btnExport1DX').addEventListener('click', () => export1DData(0));
    document.getElementById('btnExport1DY').addEventListener('click', () => export1DData(1));
    document.getElementById('btnPrintPDF1DX').addEventListener('click', () => printPDF1D(0));
    document.getElementById('btnPrintPDF1DY').addEventListener('click', () => printPDF1D(1));
    document.getElementById('btnPrintPDF2D').addEventListener('click', printPDF2D);

    document.getElementById('fwhmMultSlider1D').addEventListener('input', (e) => {
      document.getElementById('fwhmMultLabel1D').innerText = parseFloat(e.target.value).toFixed(1) + '× FWHM';
    });
    document.getElementById('roiWidthSlider2D').addEventListener('input', (e) => {
      document.getElementById('roiWidthLabel2D').innerText = '±' + e.target.value + ' ch';
    });
    const selVerb = document.getElementById('fitVerbositySelect');
    if (selVerb) selVerb.addEventListener('change', updateFitCardsVerbosity);

    const btnTog1 = document.getElementById('btnToggleDetails1D');
    if (btnTog1) btnTog1.addEventListener('click', () => {
      const sel = document.getElementById('fitVerbositySelect');
      if (sel) {
        sel.value = (sel.value === 'detailed') ? 'compact' : 'detailed';
        updateFitCardsVerbosity();
      }
    });

    const btnTog2 = document.getElementById('btnToggleDetails2D');
    if (btnTog2) btnTog2.addEventListener('click', () => {
      const sel = document.getElementById('fitVerbositySelect');
      if (sel) {
        sel.value = (sel.value === 'detailed') ? 'compact' : 'detailed';
        updateFitCardsVerbosity();
      }
    });

    document.getElementById('btnCloseFit1D').addEventListener('click', clearFit1D);
    document.getElementById('btnCloseFit2D').addEventListener('click', clearFit2D);

    document.getElementById('btnHelp').addEventListener('click', () => { document.getElementById('helpModal').style.display = 'block'; });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (e.shiftKey) pan2D(-1, 0);
        else { markers.left = cursorChannel.x; render2D(); }
      }
      else if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (e.shiftKey) pan2D(1, 0);
        else { markers.right = cursorChannel.x; render2D(); }
      }
      else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (e.shiftKey) pan2D(0, -1);
        else { markers.down = cursorChannel.y; render2D(); }
      }
      else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (e.shiftKey) pan2D(0, 1);
        else { markers.up = cursorChannel.y; render2D(); }
      }
      else if (e.key === 'e' || e.key === 'E') expandMarkers();
      else if (e.key === 'f' || e.key === 'F') zoomFull();
      else if (e.key === 'q' || e.key === 'Q') quitViewer();
      else if (e.key === '=' || e.key === '+') clearAllFits();
      else if (e.key === 'g' || e.key === 'G') {
        if (isMouseOver2D) requestPeakFit2D(cursorChannel.x, cursorChannel.y);
        else if (cursor1DChannelX !== null) requestPeakFit(0, cursor1DChannelX);
        else if (cursor1DChannelY !== null) requestPeakFit(1, cursor1DChannelY);
      } else if (e.key === 'c' || e.key === 'C') {
        const sel = document.getElementById('cmapSelect');
        sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
        updateColorLUT(); render2D();
      } else if (e.key === '1') {
        document.getElementById('scaleSelect').value = 'linear'; render2D();
      } else if (e.key === '2') {
        document.getElementById('scaleSelect').value = 'sqrt'; render2D();
      } else if (e.key === '4' || e.key === 'l' || e.key === 'L') {
        document.getElementById('scaleSelect').value = 'log'; render2D();
      } else if (e.key === '?' || e.key === 'h' || e.key === 'H') {
        const hm = document.getElementById('helpModal');
        hm.style.display = hm.style.display === 'block' ? 'none' : 'block';
      } else if (e.key === 'Escape') {
        document.getElementById('helpModal').style.display = 'none';
      }
    });

    canvas2d.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas1dX.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas1dY.addEventListener('contextmenu', (e) => e.preventDefault());

    // 2D Mouse Wheel: Centered zoom on where the crosshair is placed
    let wheel2DTimer = null;
    let wheelCenterCh = null;
    canvas2d.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? (1.0 - scrollSensitivity) : (1.0 / (1.0 - scrollSensitivity));

      // Always center the view to where the crosshair is placed in 2D
      if (!wheelCenterCh) {
        const rect = canvas2d.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const pr = getPlotRect2D();

        if (mouseX >= pr.x && mouseX <= pr.x + pr.w && mouseY >= pr.y && mouseY <= pr.y + pr.h) {
          const ch = pxToCh2D(mouseX, mouseY);
          wheelCenterCh = { x: ch.x, y: ch.y };
        } else if (cursorChannel && cursorChannel.x >= 0 && cursorChannel.y >= 0) {
          wheelCenterCh = { x: cursorChannel.x + 0.5, y: cursorChannel.y + 0.5 };
        } else {
          wheelCenterCh = { x: (view.x0 + view.x1) / 2.0, y: (view.y0 + view.y1) / 2.0 };
        }
      }

      const cx = wheelCenterCh.x;
      const cy = wheelCenterCh.y;
      const spanX = (view.x1 - view.x0) * factor;
      const spanY = (view.y1 - view.y0) * factor;

      const minSpan = 8;
      const maxSpanX = metadata ? metadata.shape[0] : 4096;
      const maxSpanY = metadata ? metadata.shape[1] : 4096;

      const newSpanX = Math.max(minSpan, Math.min(maxSpanX, spanX));
      const newSpanY = Math.max(minSpan, Math.min(maxSpanY, spanY));

      let newX0 = cx - newSpanX / 2.0;
      let newX1 = cx + newSpanX / 2.0;
      let newY0 = cy - newSpanY / 2.0;
      let newY1 = cy + newSpanY / 2.0;

      if (newX0 < 0) { newX1 += -newX0; newX0 = 0; }
      if (newX1 > maxSpanX) { newX0 -= (newX1 - maxSpanX); newX1 = maxSpanX; }
      if (newY0 < 0) { newY1 += -newY0; newY0 = 0; }
      if (newY1 > maxSpanY) { newY0 -= (newY1 - maxSpanY); newY1 = maxSpanY; }

      view.x0 = Math.max(0, Math.floor(newX0));
      view.x1 = Math.min(maxSpanX, Math.ceil(newX1));
      view.y0 = Math.max(0, Math.floor(newY0));
      view.y1 = Math.min(maxSpanY, Math.ceil(newY1));

      clearTimeout(wheel2DTimer);
      wheel2DTimer = setTimeout(() => {
        fetchTileAndRender();
        fetch1DProjection();
      }, 20);
    }, { passive: false });

    // 1D Mouse Wheel: Y-Axis zoom (keeping Ymin fixed, adjusting Ymax)
    canvas1dX.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? (1.0 - scrollSensitivity) : (1.0 / (1.0 - scrollSensitivity));
      zoom1DY[0] = Math.max(0.005, Math.min(100.0, (zoom1DY[0] || 1.0) * factor));
      render1DSpectrum(canvas1dX, ctx1dX, currentProjSpecX, 0, cursor1DChannelX);
    }, { passive: false });

    canvas1dY.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? (1.0 - scrollSensitivity) : (1.0 / (1.0 - scrollSensitivity));
      zoom1DY[1] = Math.max(0.005, Math.min(100.0, (zoom1DY[1] || 1.0) * factor));
      render1DSpectrum(canvas1dY, ctx1dY, currentProjSpecY, 1, cursor1DChannelY);
    }, { passive: false });

    canvas1dX.addEventListener('dblclick', () => {
      zoom1DY[0] = 1.0;
      renderBoth1DSpectra();
    });

    canvas1dY.addEventListener('dblclick', () => {
      zoom1DY[1] = 1.0;
      renderBoth1DSpectra();
    });

    // 2D Canvas Handlers
    canvas2d.addEventListener('mouseenter', () => { isMouseOver2D = true; });
    canvas2d.addEventListener('mouseleave', () => {
      isMouseOver2D = false;
      wheelCenterCh = null;
      if (!isBoxZooming) {
        cursor1DChannelX = null;
        cursor1DChannelY = null;
        renderBoth1DSpectra();
      }
    });

    canvas2d.addEventListener('mousedown', (e) => {
      if (e.button !== 0 || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
      const rect = canvas2d.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      dragStartPos = { x: mouseX, y: mouseY };
      mouseCurrentPos = { x: mouseX, y: mouseY };
      isBoxZooming = true;
    });

    canvas2d.addEventListener('click', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) {
        e.preventDefault();
        const rect = canvas2d.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const ch = pxToCh2D(mouseX, mouseY);
        requestPeakFit2D(ch.x, ch.y);
      }
    });

    canvas2d.addEventListener('mousemove', (e) => {
      wheelCenterCh = null;
      const rect = canvas2d.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      mouseCurrentPos = { x: mouseX, y: mouseY };

      const ch = pxToCh2D(mouseX, mouseY);
      cursorChannel = {
        x: Math.max(0, Math.min(metadata.shape[0] - 1, Math.floor(ch.x))),
        y: Math.max(0, Math.min(metadata.shape[1] - 1, Math.floor(ch.y)))
      };

      if (currentProjSpecX && currentProjSpecY) {
        const pr2D = getPlotRect2D();
        if (mouseX >= pr2D.x && mouseX <= pr2D.x + pr2D.w && mouseY >= pr2D.y && mouseY <= pr2D.y + pr2D.h) {
          cursor1DChannelX = cursorChannel.x;
          cursor1DChannelY = cursorChannel.y;
          renderBoth1DSpectra();
        } else if (!isBoxZooming) {
          cursor1DChannelX = null;
          cursor1DChannelY = null;
          renderBoth1DSpectra();
        }
      }

      if (isBoxZooming) render2D();

      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(updateHoverHUD, 20);
    });

    window.addEventListener('mousemove', (e) => {
      if (isBoxZooming) {
        const rect = canvas2d.getBoundingClientRect();
        mouseCurrentPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        render2D();
      }
      if (is1DBoxZooming[0]) {
        const rect = canvas1dX.getBoundingClientRect();
        drag1DCurrentPos[0] = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        render1DSpectrum(canvas1dX, ctx1dX, currentProjSpecX, 0, cursor1DChannelX);
      }
      if (is1DBoxZooming[1]) {
        const rect = canvas1dY.getBoundingClientRect();
        drag1DCurrentPos[1] = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        render1DSpectrum(canvas1dY, ctx1dY, currentProjSpecY, 1, cursor1DChannelY);
      }
    });

    window.addEventListener('mouseup', async (e) => {
      if (isBoxZooming) {
        isBoxZooming = false;
        const xS = Math.min(dragStartPos.x, mouseCurrentPos.x), xE = Math.max(dragStartPos.x, mouseCurrentPos.x);
        const yS = Math.min(dragStartPos.y, mouseCurrentPos.y), yE = Math.max(dragStartPos.y, mouseCurrentPos.y);
        if (xE - xS > 4 && yE - yS > 4) {
          const ch0 = pxToCh2D(xS, yE), ch1 = pxToCh2D(xE, yS);
          view.x0 = Math.max(0, Math.floor(ch0.x)); view.x1 = Math.min(metadata.shape[0], Math.ceil(ch1.x));
          view.y0 = Math.max(0, Math.floor(ch0.y)); view.y1 = Math.min(metadata.shape[1], Math.ceil(ch1.y));
          fetchTileAndRender(); fetch1DProjection();
        } else render2D();
      }

      if (is1DBoxZooming[0]) {
        is1DBoxZooming[0] = false;
        const xS = Math.min(drag1DStartPos[0].x, drag1DCurrentPos[0].x);
        const xE = Math.max(drag1DStartPos[0].x, drag1DCurrentPos[0].x);
        if (xE - xS > 5 && currentProjSpecX) {
          const pr = getPlotRect1D(canvas1dX);
          const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
          let chStart = 0, chEnd = currentProjSpecX.length - 1;
          if (isSynced) {
            chStart = Math.max(0, Math.floor(view.x0));
            chEnd = Math.min(currentProjSpecX.length - 1, Math.ceil(view.x1));
          }
          const span = Math.max(1, chEnd - chStart + 1);
          const frac0 = Math.max(0, Math.min(1, (xS - pr.x) / pr.w));
          const frac1 = Math.max(0, Math.min(1, (xE - pr.x) / pr.w));
          const c0 = Math.round(chStart + frac0 * (span - 1));
          const c1 = Math.round(chStart + frac1 * (span - 1));
          if (c1 > c0) {
            view.x0 = Math.max(0, c0);
            view.x1 = Math.min(metadata.shape[0], c1 + 1);
            fetchTileAndRender();
            fetch1DProjection();
          } else {
            renderBoth1DSpectra();
          }
        } else {
          renderBoth1DSpectra();
        }
      }

      if (is1DBoxZooming[1]) {
        is1DBoxZooming[1] = false;
        const xS = Math.min(drag1DStartPos[1].x, drag1DCurrentPos[1].x);
        const xE = Math.max(drag1DStartPos[1].x, drag1DCurrentPos[1].x);
        if (xE - xS > 5 && currentProjSpecY) {
          const pr = getPlotRect1D(canvas1dY);
          const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
          let chStart = 0, chEnd = currentProjSpecY.length - 1;
          if (isSynced) {
            chStart = Math.max(0, Math.floor(view.y0));
            chEnd = Math.min(currentProjSpecY.length - 1, Math.ceil(view.y1));
          }
          const span = Math.max(1, chEnd - chStart + 1);
          const frac0 = Math.max(0, Math.min(1, (xS - pr.x) / pr.w));
          const frac1 = Math.max(0, Math.min(1, (xE - pr.x) / pr.w));
          const c0 = Math.round(chStart + frac0 * (span - 1));
          const c1 = Math.round(chStart + frac1 * (span - 1));
          if (c1 > c0) {
            view.y0 = Math.max(0, c0);
            view.y1 = Math.min(metadata.shape[1], c1 + 1);
            fetchTileAndRender();
            fetch1DProjection();
          } else {
            renderBoth1DSpectra();
          }
        } else {
          renderBoth1DSpectra();
        }
      }
    });

    // 1D Spectrum Handlers
    function handle1DHover(canvas, spec, axis, e) {
      if (!spec) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const pr = getPlotRect1D(canvas);
      const isSynced = (document.getElementById('projRangeSelect').value === 'synced');

      let chStart = 0, chEnd = spec.length - 1;
      if (isSynced) {
        if (axis === 0) {
          chStart = Math.max(0, Math.floor(view.x0));
          chEnd = Math.min(spec.length - 1, Math.ceil(view.x1));
        } else {
          chStart = Math.max(0, Math.floor(view.y0));
          chEnd = Math.min(spec.length - 1, Math.ceil(view.y1));
        }
      }
      const span = Math.max(1, chEnd - chStart + 1);
      const frac = (mouseX - pr.x) / pr.w;
      if (frac >= 0 && frac <= 1) {
        const ch = Math.max(0, Math.min(spec.length - 1, Math.floor(chStart + frac * span)));
        if (axis === 0) cursor1DChannelX = ch; else cursor1DChannelY = ch;
        const val = spec[ch] || 0;
        const energy = chToEnergy(ch).toFixed(1);
        const detName = (axis === 0) ? 'Det 1 (1D)' : 'Det 2 (1D)';
        document.getElementById('hudCoords').innerText = `${detName}: ch ${ch} (${energy} keV) | Proj Counts: ${val}`;
      } else {
        if (!is1DBoxZooming[axis]) {
          if (axis === 0) cursor1DChannelX = null; else cursor1DChannelY = null;
        }
      }
      renderBoth1DSpectra();
    }

    function handle1DClick(canvas, spec, axis, e) {
      if (!spec) return;
      if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const pr = getPlotRect1D(canvas);
        const isSynced = (document.getElementById('projRangeSelect').value === 'synced');

        let chStart = 0, chEnd = spec.length - 1;
        if (isSynced) {
          if (axis === 0) {
            chStart = Math.max(0, Math.floor(view.x0));
            chEnd = Math.min(spec.length - 1, Math.ceil(view.x1));
          } else {
            chStart = Math.max(0, Math.floor(view.y0));
            chEnd = Math.min(spec.length - 1, Math.ceil(view.y1));
          }
        }
        const span = Math.max(1, chEnd - chStart + 1);
        const frac = (mouseX - pr.x) / pr.w;
        if (frac >= 0 && frac <= 1) {
          const ch = Math.max(0, Math.min(spec.length - 1, Math.floor(chStart + frac * span)));
          requestPeakFit(axis, ch);
        }
      }
    }

    function handle1DMousedown(canvas, axis, e) {
      if (e.button !== 0 || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      is1DBoxZooming[axis] = true;
      drag1DStartPos[axis] = { x: mouseX, y: mouseY };
      drag1DCurrentPos[axis] = { x: mouseX, y: mouseY };
    }

    canvas1dX.addEventListener('mousedown', (e) => handle1DMousedown(canvas1dX, 0, e));
    canvas1dX.addEventListener('mousemove', (e) => handle1DHover(canvas1dX, currentProjSpecX, 0, e));
    canvas1dX.addEventListener('mouseleave', () => { if (!is1DBoxZooming[0]) { cursor1DChannelX = null; renderBoth1DSpectra(); } });
    canvas1dX.addEventListener('click', (e) => handle1DClick(canvas1dX, currentProjSpecX, 0, e));

    canvas1dY.addEventListener('mousedown', (e) => handle1DMousedown(canvas1dY, 1, e));
    canvas1dY.addEventListener('mousemove', (e) => handle1DHover(canvas1dY, currentProjSpecY, 1, e));
    canvas1dY.addEventListener('mouseleave', () => { if (!is1DBoxZooming[1]) { cursor1DChannelY = null; renderBoth1DSpectra(); } });
    canvas1dY.addEventListener('click', (e) => handle1DClick(canvas1dY, currentProjSpecY, 1, e));
  }

  window.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""
def main():
    parser = argparse.ArgumentParser(
        description="Launch modern Web-based interactive 2D viewer with classic binned 1D histogram."
    )
    parser.add_argument("input", type=str, help="Path to input .cmat file")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    parser.add_argument(
        "--cal",
        nargs="+",
        type=float,
        metavar="COEFF",
        default=[0.0, 1.0, 0.0],
        help="Energy calibration coefficients: a0 a1 a2 for E = a0 + a1*ch + a2*ch^2 (default: 0.0 1.0 0.0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading matrix from {input_path} ...")
    reader = CMATReader(input_path)
    mat = reader.to_numpy()
    proj = reader.get_projection()

    CMATWebHandler.reader = reader
    CMATWebHandler.matrix = mat
    CMATWebHandler.proj = proj
    CMATWebHandler.cal = args.cal

    server_address = ("", args.port)
    httpd = HTTPServer(server_address, CMATWebHandler)
    url = f"http://localhost:{args.port}"
    print(f"\n[+] Interactive 2D CMAT Web Viewer is ready!")
    print(f"[+] Access the viewer at: {url}")
    print(f"[+] Classic Binned 1D Histogram with real-time mouse inspector.")
    print(f"[+] Press Ctrl+C in terminal to stop server.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    main()
