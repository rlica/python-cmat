#!/usr/bin/env python3
"""
cmat3d_webviewer.py - High-Performance Web-based Interactive 3D Matrix Viewer.

Dedicated to 3-dimensional coincidence matrices in GASPware/gsort .cmat format.
Independent implementation safeguarding the mature 2D viewer.

Features:
  - Three Concurrent 1D Stepped-Staircase Histograms for Axis 1 (X), Axis 2 (Y), and Axis 3 (Z).
  - Selectable 2D Projection Plane:
      * Plane 0-1: Axis 1 (X) vs Axis 2 (Y) [summed over Axis 3 (Z)]
      * Plane 0-2: Axis 1 (X) vs Axis 3 (Z) [summed over Axis 2 (Y)]
      * Plane 1-2: Axis 2 (Y) vs Axis 3 (Z) [summed over Axis 1 (X)]
  - 3rd Axis Gating / Slicing: Sum over all channels by default, or gate on specific channels / regions.
  - Multi-Axis Coincidence Gating: Single-gate and double-gate with background subtraction.
  - Instant memory-mapped access and pre-computed 2D projection caches.
  - Full GASPware navigation, 2D box zoom, limits (L/R/D/U), peak search, multi-peak fitting, integration, and publication PDF exports.

Usage:
  python3 cmat3d_webviewer.py matrix3d.cmat --port 8081
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
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
try:
    from http.server import ThreadingHTTPServer as HTTPServerClass
except ImportError:
    from http.server import HTTPServer as HTTPServerClass
from http.server import BaseHTTPRequestHandler
import numpy as np

# Configure Matplotlib cache directory within workspace
cache_dir = Path(__file__).resolve().parent / ".matplotlib_cache"
try:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
except Exception:
    pass

from cmat3d import CMAT3DReader, compute_3d_gate, compute_2d_banana_gate, compute_2d_gamba_gate, parse_gate_ranges
from cmat_webviewer import (
    fit_gaussian_peak,
    fit_all_peaks_1d,
    find_peaks_1d,
    integrate_peak_1d,
    fit_2d_gaussian_peak,
    generate_pdf_1d,
    generate_pdf_2d,
    browse_filesystem,
    get_local_ip,
    is_ssh_session,
    launch_browser,
    print_fit_terminal_report,
    print_multi_fit_terminal_report,
    print_integrate_terminal_report,
    print_peaks_terminal_report,
    print_fit_2d_terminal_report,
    append_fit_1d_result_to_file,
    append_fit_2d_result_to_file,
    get_default_fit_log_filename,
)

CONFIG_FILENAME = "python-cmat3d-config.txt"


def generate_config_content(cfg: dict) -> str:
    """Generate clean, commented INI-style python-cmat3d-config.txt text."""
    cal_0_str = cfg.get("cal_0", "0.0, 1.0, 0.0")
    if isinstance(cal_0_str, (list, tuple)):
        cal_0_str = ", ".join(str(v) for v in cal_0_str)

    cal_1_str = cfg.get("cal_1", "0.0, 1.0, 0.0")
    if isinstance(cal_1_str, (list, tuple)):
        cal_1_str = ", ".join(str(v) for v in cal_1_str)

    cal_2_str = cfg.get("cal_2", "0.0, 1.0, 0.0")
    if isinstance(cal_2_str, (list, tuple)):
        cal_2_str = ", ".join(str(v) for v in cal_2_str)

    open_br = cfg.get("open_browser", True)
    open_br_str = "true" if open_br in (True, "true", "True", "1", 1) else "false"

    return f"""# ==============================================================================
# python-cmat3d configuration file
# Automatically generated when no config file is present in the working directory.
# You can edit these values directly or click "Save Config" in the Web Viewer.
# ==============================================================================

# Energy Calibration Polynomials: E(keV) = a0 + a1*ch + a2*ch^2
cal_0 = {cal_0_str}
cal_1 = {cal_1_str}
cal_2 = {cal_2_str}

# Default 2D Projection Plane ('0-1', '0-2', or '1-2')
default_plane = {cfg.get('default_plane', '0-1')}

# Peak Fitting Defaults
fit_type = {cfg.get('fit_type', 'gaussian')}
fwhm_mult_1d = {cfg.get('fwhm_mult_1d', 4.0)}
roi_half_width_2d = {cfg.get('roi_half_width_2d', 16)}
fit_verbosity = {cfg.get('fit_verbosity', 'compact')}
fit_log = {"true" if cfg.get("fit_log") in (True, "true", "True", "1", 1) else "false"}
fit_log_file = {cfg.get('fit_log_file', '')}

# 2D Heatmap & Color Defaults
colormap = {cfg.get('colormap', 'turbo')}
scale_mode = {cfg.get('scale_mode', 'log')}
vmax = {cfg.get('vmax', 500)}
vmin = {cfg.get('vmin', 1)}

# Navigation & Scrolling
scroll_sensitivity = {cfg.get('scroll_sensitivity', 4)}

# Automatic Peak Search Defaults
peak_search_method = {cfg.get('peak_search_method', 'cwt')}
peak_search_snr = {cfg.get('peak_search_snr', 9.0)}

# 1D Spectrum Display Defaults
proj_range = {cfg.get('proj_range', 'synced')}
proj_scale = {cfg.get('proj_scale', 'linear')}

# Server Network Settings
host = {cfg.get('host', '0.0.0.0')}
port = {cfg.get('port', 8081)}
open_browser = {open_br_str}
browser = {cfg.get('browser', 'default')}
"""


def save_config_file(config_path: Path, current_settings: dict) -> None:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(current_settings)
    config_path.write_text(generate_config_content(cfg), encoding="utf-8")


def print_gate_terminal_report_3d(gate_res: dict, matrix_name: str, target_axis: int, gate_specs: dict):
    axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]
    dst_det = axis_names[target_axis]
    gated_axes = sorted(gate_specs.keys())
    if len(gated_axes) == 1:
        ga = gated_axes[0]
        src_det = axis_names[ga]
        w_strs = [f"[{w[0]}..{w[1]}]" for w in gate_specs[ga].get("w", [])] or ["None"]
        b_strs = [f"[{b[0]}..{b[1]}]" for b in gate_specs[ga].get("b", [])] or ["None"]
        scale_text = f"{gate_res.get('scale', 0.0):.4f}"
        print(f"\n[1D Gate Cut] {matrix_name} -> Gated {src_det} => {dst_det} Coincidence Spectrum:")
        print(f"  • Gate W: {', '.join(w_strs)} (total width: {gate_res.get('w_width', 0)} ch)")
        print(f"  • Bg B:   {', '.join(b_strs)} (total width: {gate_res.get('b_width', 0)} ch, scale: {scale_text})")
        print(f"  • Counts: Net={sum(gate_res.get('net_spec', [])):,.1f} cts\n", flush=True)
    elif len(gated_axes) == 2:
        g1, g2 = gated_axes[0], gated_axes[1]
        src_det = f"{axis_names[g1]} & {axis_names[g2]}"
        print(f"\n[3D Double Gate Cut] {matrix_name} -> Gated {src_det} => {dst_det} Double-Coincidence Spectrum:")
        print(f"  • Gate 1: W={gate_specs[g1].get('w', [])}, B={gate_specs[g1].get('b', [])} (scale: {gate_res.get('scale_1', 0.0):.4f})")
        print(f"  • Gate 2: W={gate_specs[g2].get('w', [])}, B={gate_specs[g2].get('b', [])} (scale: {gate_res.get('scale_2', 0.0):.4f})")
        print(f"  • Counts: Net={sum(gate_res.get('net_spec', [])):,.1f} cts\n", flush=True)


def print_banana_gate_terminal_report_3d(res: dict, matrix_name: str):
    axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]
    dst_det = axis_names[res["target_axis"]]
    plane = res.get("plane", "0-1")
    has_bg = res.get("has_bg", False)

    print(f"\n[2D Banana Gate Cut] {matrix_name} -> Plane {plane} => {dst_det}:")

    px_peak = res.get("pixel_count_peak", res.get("pixel_count", 0))
    cts_peak = res.get("counts_peak", res.get("total_counts", 0))
    area_peak = res.get("area_peak", float(px_peak))
    poly_peak = res.get("polygon_peak", res.get("polygon", []))

    print(f"  • Peak Banana (W): {px_peak:,} px (area: {area_peak:,.1f} ch², {len(poly_peak)} vertices) | Counts: {cts_peak:,} cts")

    if has_bg:
        px_bg = res.get("pixel_count_bg", 0)
        cts_bg = res.get("counts_bg", 0)
        area_bg = res.get("area_bg", float(px_bg))
        poly_bg = res.get("polygon_bg", [])
        scale = res.get("scale", 0.0)
        print(f"  • Bg Banana (B):   {px_bg:,} px (area: {area_bg:,.1f} ch², {len(poly_bg)} vertices) | Counts: {cts_bg:,} cts (Scale factor: {scale:.4f})")
        print(f"  • Net Area Counts: {res.get('net_counts', 0):,.1f} counts\n", flush=True)
    else:
        print(f"  • Net Gated Counts:{res.get('total_gated_counts', 0):,} counts\n", flush=True)


def print_gamba_gate_terminal_report_3d(gamba_res: dict, matrix_name: str, plane: str, cal_z: tuple, is_cal_z: bool = False):
    axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]
    target_axis = gamba_res.get("target_axis", 2)
    dst_det = axis_names[target_axis]

    if plane == "0-1":
        ax_x, ax_y = 0, 1
    elif plane == "0-2":
        ax_x, ax_y = 0, 2
    else:
        ax_x, ax_y = 1, 2

    cx_ch = gamba_res.get("centroid_x_ch", 0.0)
    cy_ch = gamba_res.get("centroid_y_ch", 0.0)
    cx_e = gamba_res.get("centroid_x_e")
    cy_e = gamba_res.get("centroid_y_e")
    pos_x = f"{cx_e:.1f} keV (ch {cx_ch:.1f})" if cx_e is not None else f"ch {cx_ch:.1f}"
    pos_y = f"{cy_e:.1f} keV (ch {cy_ch:.1f})" if cy_e is not None else f"ch {cy_ch:.1f}"

    px_r = gamba_res.get("peak_x_range", [0, 0])
    py_r = gamba_res.get("peak_y_range", [0, 0])
    rx_r = gamba_res.get("roi_x_range", [0, 0])
    ry_r = gamba_res.get("roi_y_range", [0, 0])

    print(f"\n[2D Gamba 3D Coincidence Cut] {matrix_name} -> Plane {plane} ({axis_names[ax_x]} × {axis_names[ax_y]}) => {dst_det}:", flush=True)
    print(f"  • Fitted 2D Coincidence Peak: {axis_names[ax_x]} = {pos_x}, {axis_names[ax_y]} = {pos_y}", flush=True)
    print(f"  • Gamba 2D Gates (ROI [{rx_r[0]}..{rx_r[1]}] × [{ry_r[0]}..{ry_r[1]}]):", flush=True)
    print(f"      P|P:   [{px_r[0]}..{px_r[1]}] × [{py_r[0]}..{py_r[1]}] (Area: {gamba_res.get('area_pp', 0):,} px, Raw S_pp: {gamba_res.get('n_pp_m', 0):,.1f} cts)", flush=True)
    print(f"      P|BG:  Peak X × ROI BG Y (Area: {gamba_res.get('area_pbg', 0):,} px, Scale: {gamba_res.get('scale_pbg', 0.0):.4f}, Raw: {gamba_res.get('n_pbg_raw', 0):,.1f} cts)", flush=True)
    print(f"      BG|P:  ROI BG X × Peak Y (Area: {gamba_res.get('area_bgp', 0):,} px, Scale: {gamba_res.get('scale_bgp', 0.0):.4f}, Raw: {gamba_res.get('n_bgp_raw', 0):,.1f} cts)", flush=True)
    print(f"      BG|BG: ROI BG X × ROI BG Y (Area: {gamba_res.get('area_bgbg', 0):,} px, Scale: {gamba_res.get('scale_bgbg', 0.0):.4f}, Raw: {gamba_res.get('n_bgbg_raw', 0):,.1f} cts)", flush=True)
    print(f"  • {dst_det} Net Coincidence Spectrum: Integrated Area = {gamba_res.get('net_counts', 0):,.1f} cts (Π Ratio = {gamba_res.get('pi_ratio_percent', 0):.1f}%)\n", flush=True)


DEFAULT_CONFIG = {
    "cal": "0.0, 1.0, 0.0",
    "cal_0": "0.0, 1.0, 0.0",
    "cal_1": "0.0, 1.0, 0.0",
    "cal_2": "0.0, 1.0, 0.0",
    "default_plane": "0-1",
    "fit_type": "gaussian",
    "fwhm_mult_1d": 4.0,
    "roi_half_width_2d": 16,
    "fit_verbosity": "compact",
    "fit_log": False,
    "fit_log_file": "",
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
    "port": 8081,
    "open_browser": True,
    "browser": "default",
}


def parse_cal_coefficients(cal_str: Any) -> list:
    if isinstance(cal_str, (list, tuple)):
        return [float(x) for x in cal_str[:3]]
    if not cal_str or not str(cal_str).strip():
        return [0.0, 1.0, 0.0]
    parts = str(cal_str).replace(",", " ").split()
    coeffs = []
    for p in parts:
        try:
            coeffs.append(float(p))
        except ValueError:
            pass
    while len(coeffs) < 2:
        coeffs.append(1.0 if len(coeffs) == 1 else 0.0)
    if len(coeffs) < 3:
        coeffs.append(0.0)
    return coeffs[:3]


def load_or_create_config(path: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if not path.exists():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# python-cmat3d configuration file\n")
                for k, v in cfg.items():
                    f.write(f"{k} = {v}\n")
        except Exception:
            pass
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ("port", "vmax", "vmin", "scroll_sensitivity", "roi_half_width_2d"):
                        try:
                            cfg[k] = int(v)
                        except ValueError:
                            pass
                    elif k in ("fwhm_mult_1d", "peak_search_snr"):
                        try:
                            cfg[k] = float(v)
                        except ValueError:
                            pass
                    elif k in ("open_browser", "fit_log"):
                        cfg[k] = v.lower() in ("true", "1", "yes", "on")
                    else:
                        cfg[k] = v
        print(f"[*] Loaded configuration from {path.name}")
    except Exception as e:
        print(f"[!] Warning: Could not read config {path.name}: {e}. Using defaults.", file=sys.stderr)
    return cfg


HTML_FILE_PATH = Path(__file__).resolve().parent / "cmat3d_webviewer.html"


def get_html_content() -> str:
    if not HTML_FILE_PATH.exists():
        raise FileNotFoundError(f"Template '{HTML_FILE_PATH}' was not found.")
    return HTML_FILE_PATH.read_text(encoding="utf-8")


class MatrixSession3D:
    def __init__(self, config: dict):
        self.config = config
        self.matrices: List[dict] = []
        self.active_index: int = 0
        self.active_plane: str = config.get("default_plane", "0-1")
        self.gate_3rd: Optional[List[int]] = None
        self.gates_1d: Dict[int, Dict[str, List[List[int]]]] = {
            0: {"w": [], "b": []},
            1: {"w": [], "b": []},
            2: {"w": [], "b": []},
        }

        # Energy calibrations
        c_global = parse_cal_coefficients(config.get("cal", "0.0, 1.0, 0.0"))
        self.cal: Dict[int, list] = {
            0: parse_cal_coefficients(config.get("cal_0", c_global)),
            1: parse_cal_coefficients(config.get("cal_1", c_global)),
            2: parse_cal_coefficients(config.get("cal_2", c_global)),
        }
        self.integration_1d: Dict[int, Optional[dict]] = {0: None, 1: None, 2: None}
        self.fit_2d = None
        self.fit_log_enabled = bool(config.get("fit_log", False))
        self.fit_log_filename = str(config.get("fit_log_file", "") or "")
        if self.fit_log_enabled and not self.fit_log_filename:
            self.fit_log_filename = get_default_fit_log_filename()

    def set_fit_log(self, enabled: bool, filename: str = None) -> dict:
        self.fit_log_enabled = bool(enabled)
        if filename:
            self.fit_log_filename = str(filename).strip()
        elif self.fit_log_enabled and not self.fit_log_filename:
            self.fit_log_filename = get_default_fit_log_filename()
        return {
            "enabled": self.fit_log_enabled,
            "filename": self.fit_log_filename
        }

    def add_matrix_file(self, path: Path, name: str = None, cal: dict = None) -> int:
        path = Path(path).resolve()
        for idx, m in enumerate(self.matrices):
            if m["path"] == str(path):
                if name:
                    m["name"] = name
                return idx

        reader = CMAT3DReader(path, use_cache=True)
        matrix_name = name or path.name

        matrix_cal = {0: list(self.cal[0]), 1: list(self.cal[1]), 2: list(self.cal[2])}
        if cal:
            if isinstance(cal, dict):
                for k, v in cal.items():
                    if int(k) in (0, 1, 2):
                        matrix_cal[int(k)] = parse_cal_coefficients(v)

        entry = {
            "index": len(self.matrices),
            "name": matrix_name,
            "filename": path.name,
            "path": str(path),
            "reader": reader,
            "shape": list(reader.shape),  # [res1, res2, res3]
            "step": list(reader.step),
            "total_counts": reader._total_counts or int(np.sum(reader.get_projection(0))),
            "max_count": reader._max_count or int(np.max(reader.proj_2d.get("0-1", 0))),
            "nonzero_voxels": reader._nonzero_voxels or 0,
            "cal": matrix_cal,
        }
        self.matrices.append(entry)
        return len(self.matrices) - 1

    def get_active_matrix(self) -> Optional[dict]:
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
        except ValueError:
            pass

        ident_str = str(identifier).strip().lower()
        for idx, m in enumerate(self.matrices):
            if m["name"].lower() == ident_str or m["filename"].lower() == ident_str:
                self.active_index = idx
                return idx
        raise KeyError(f"Matrix '{identifier}' not found in loaded matrices.")

    def set_active_plane(self, plane: str) -> str:
        p = str(plane).strip()
        if p in ("0-1", "0-2", "1-2"):
            self.active_plane = p
        return self.active_plane

    def get_cal(self, axis: int) -> list:
        m = self.get_active_matrix()
        if m and "cal" in m and axis in m["cal"]:
            return m["cal"][axis]
        return self.cal.get(axis, [0.0, 1.0, 0.0])

    def set_cal(self, axis: int, coeffs: list):
        parsed = parse_cal_coefficients(coeffs)
        self.cal[axis] = parsed
        m = self.get_active_matrix()
        if m and "cal" in m:
            m["cal"][axis] = parsed

    def is_calibrated(self, axis: int) -> bool:
        c = self.get_cal(axis)
        return not (abs(c[0]) < 1e-9 and abs(c[1] - 1.0) < 1e-9 and abs(c[2]) < 1e-9)


class CMAT3DWebHandler(BaseHTTPRequestHandler):
    session: Optional[MatrixSession3D] = None

    def log_message(self, format, *args):
        # Silence default HTTP access logs to keep terminal clean
        return

    @classmethod
    def get_session(cls) -> MatrixSession3D:
        return cls.session

    def get_metadata_dict(self) -> dict:
        session = self.get_session()
        m = session.get_active_matrix()
        if not m:
            return {"loaded": False, "matrices": []}

        reader: CMAT3DReader = m["reader"]
        plane = session.active_plane

        # Dimensions mapping for active plane
        # 0-1: X=Axis 0 (Axis 1), Y=Axis 1 (Axis 2), 3rd=Axis 2 (Axis 3)
        # 0-2: X=Axis 0 (Axis 1), Y=Axis 2 (Axis 3), 3rd=Axis 1 (Axis 2)
        # 1-2: X=Axis 1 (Axis 2), Y=Axis 2 (Axis 3), 3rd=Axis 0 (Axis 1)
        axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]
        if plane == "0-1":
            ax_x, ax_y, ax_3rd = 0, 1, 2
            dim_x, dim_y = reader.res1, reader.res2
        elif plane == "0-2":
            ax_x, ax_y, ax_3rd = 0, 2, 1
            dim_x, dim_y = reader.res1, reader.res3
        else:
            ax_x, ax_y, ax_3rd = 1, 2, 0
            dim_x, dim_y = reader.res2, reader.res3

        return {
            "loaded": True,
            "matrix_index": session.active_index,
            "matrix_name": m["name"],
            "filename": m["filename"],
            "path": m["path"],
            "ndim": 3,
            "shape": [reader.res1, reader.res2, reader.res3],
            "step": [reader.step1, reader.step2, reader.step3],
            "blocks": [reader.ndiv1, reader.ndiv2, reader.ndiv3],
            "active_plane": plane,
            "plane_axis_indices": [ax_x, ax_y, ax_3rd],
            "plane_axis_names": [axis_names[ax_x], axis_names[ax_y], axis_names[ax_3rd]],
            "plane_shape": [dim_x, dim_y],
            "gate_3rd": session.gate_3rd,
            "cal_0": session.get_cal(0),
            "cal_1": session.get_cal(1),
            "cal_2": session.get_cal(2),
            "is_cal_0": session.is_calibrated(0),
            "is_cal_1": session.is_calibrated(1),
            "is_cal_2": session.is_calibrated(2),
            "total_counts": m["total_counts"],
            "max_count": m["max_count"],
            "nonzero_voxels": m["nonzero_voxels"],
            "matrices": [
                {
                    "index": i,
                    "name": mat["name"],
                    "filename": mat["filename"],
                    "shape": mat["shape"],
                    "total_counts": mat["total_counts"],
                }
                for i, mat in enumerate(session.matrices)
            ],
        }

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        session = self.get_session()
        m = session.get_active_matrix() if session else None
        reader: Optional[CMAT3DReader] = m["reader"] if m else None

        if path == "/" or path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_html_content().encode("utf-8"))

        elif path == "/api/quit":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "shutting_down"}).encode("utf-8"))
            print("\n[*] Quit request received. Shutting down server...\n", flush=True)
            threading.Timer(0.15, lambda: os._exit(0)).start()

        elif path == "/api/metadata":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.get_metadata_dict()).encode("utf-8"))

        elif path == "/api/set_plane":
            plane = query.get("plane", ["0-1"])[0]
            session.set_active_plane(plane)
            print(f"[*] 2D Display Plane switched to: {session.active_plane}", flush=True)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.get_metadata_dict()).encode("utf-8"))

        elif path == "/api/set_gate_3rd":
            gate_str = query.get("gate", [""])[0]
            if gate_str.strip():
                parsed_g = parse_gate_ranges(gate_str)
                session.gate_3rd = parsed_g[0] if parsed_g else None
            else:
                session.gate_3rd = None

            print(f"[*] 3rd Axis Gate set to: {session.gate_3rd}", flush=True)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.get_metadata_dict()).encode("utf-8"))

        elif path == "/api/tile":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            plane = query.get("plane", [session.active_plane])[0]
            gate_str = query.get("gate_3rd", [None])[0]
            gate_3rd = parse_gate_ranges(gate_str)[0] if (gate_str and gate_str.strip()) else session.gate_3rd

            # Dimensions of active 2D plane
            if plane == "0-1":
                max_w, max_h = reader.res1, reader.res2
            elif plane == "0-2":
                max_w, max_h = reader.res1, reader.res3
            else:
                max_w, max_h = reader.res2, reader.res3

            x0 = max(0, min(max_w - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(max_w, int(float(query.get("x1", [max_w])[0]))))
            y0 = max(0, min(max_h - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(max_h, int(float(query.get("y1", [max_h])[0]))))

            target_w = max(16, min(2048, int(query.get("w", [800])[0])))
            target_h = max(16, min(2048, int(query.get("h", [600])[0])))

            sub = reader.get_2d_plane(plane=plane, gate_3rd=gate_3rd, x0=x0, x1=x1, y0=y0, y1=y1)
            sh_y, sh_x = sub.shape

            # 2D max pooling downsampling
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
                out = sub_padded.reshape(new_h, step_y, new_w, step_x).max(axis=(1, 3))
                out = np.ascontiguousarray(out[:target_h, :target_w], dtype=np.int32)

            actual_h, actual_w = out.shape
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Access-Control-Expose-Headers", "X-Shape-W, X-Shape-H, X-Slice-X0, X-Slice-X1, X-Slice-Y0, X-Slice-Y1")
            self.send_header("X-Shape-W", str(actual_w))
            self.send_header("X-Shape-H", str(actual_h))
            self.send_header("X-Slice-X0", str(x0))
            self.send_header("X-Slice-X1", str(x1))
            self.send_header("X-Slice-Y0", str(y0))
            self.send_header("X-Slice-Y1", str(y1))
            self.end_headers()
            self.wfile.write(out.tobytes())

        elif path == "/api/projection_region":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            plane = query.get("plane", [session.active_plane])[0]
            if plane == "0-1":
                max_w, max_h = reader.res1, reader.res2
            elif plane == "0-2":
                max_w, max_h = reader.res1, reader.res3
            else:
                max_w, max_h = reader.res2, reader.res3

            x0 = max(0, min(max_w - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(max_w, int(float(query.get("x1", [max_w])[0]))))
            y0 = max(0, min(max_h - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(max_h, int(float(query.get("y1", [max_h])[0]))))

            gate_str = query.get("gate_3rd", [None])[0]
            gate_3rd = parse_gate_ranges(gate_str)[0] if (gate_str and gate_str.strip()) else session.gate_3rd

            spec0, spec1, spec2 = reader.get_projections_for_region(
                plane=plane, x0=x0, x1=x1, y0=y0, y1=y1, gate_3rd=gate_3rd
            )

            resp = {
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "plane": plane,
                "spec0": spec0.tolist(),
                "spec1": spec1.tolist(),
                "spec2": spec2.tolist(),
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif path == "/api/gate_1d":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            target_axis = int(query.get("target_axis", [query.get("axis", [0])[0]])[0])
            gate_specs = {}
            for ax in (0, 1, 2):
                if ax == target_axis:
                    continue
                w_str = query.get(f"w{ax}", [query.get(f"w_gates_{ax}", [""])[0]])[0]
                b_str = query.get(f"b{ax}", [query.get(f"b_gates_{ax}", [""])[0]])[0]
                w_g = parse_gate_ranges(w_str)
                b_g = parse_gate_ranges(b_str)
                if w_g or b_g:
                    gate_specs[ax] = {"w": w_g, "b": b_g}

            gate_res = compute_3d_gate(reader, target_axis, gate_specs)
            if gate_specs:
                print_gate_terminal_report_3d(gate_res, m["filename"], target_axis, gate_specs)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(gate_res).encode("utf-8"))

        elif path == "/api/banana_gate":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            plane = query.get("plane", [session.active_plane])[0]
            poly_peak_str = query.get("polygon_peak", query.get("polygon", ["[]"]))[0]
            poly_bg_str = query.get("polygon_bg", ["[]"])[0]
            try:
                polygon_peak = json.loads(poly_peak_str)
            except Exception:
                polygon_peak = []
            try:
                polygon_bg = json.loads(poly_bg_str)
            except Exception:
                polygon_bg = []

            res = compute_2d_banana_gate(reader, plane, polygon_peak=polygon_peak, polygon_bg=polygon_bg)
            if res.get("success") and (res.get("pixel_count_peak", 0) > 0 or res.get("pixel_count_bg", 0) > 0):
                print_banana_gate_terminal_report_3d(res, m["filename"])

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path.startswith("/api/search_peaks"):
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            axis = int(query.get("axis", [0])[0])
            method = query.get("method", ["cwt"])[0].lower()
            min_snr = float(query.get("min_snr", [9.0])[0])
            min_counts = float(query.get("min_counts", [10.0])[0])
            fwhm_est = float(query.get("fwhm_est", [4.0])[0])
            range_mode = query.get("range", ["visible"])[0].lower()
            plane = query.get("plane", [session.active_plane])[0]

            x0 = int(float(query.get("x0", [0])[0]))
            x1 = int(float(query.get("x1", [4096])[0]))
            y0 = int(float(query.get("y0", [0])[0]))
            y1 = int(float(query.get("y1", [4096])[0]))

            # Check if gated
            gate_specs = {}
            for ax in (0, 1, 2):
                if ax != axis:
                    w_g = parse_gate_ranges(query.get(f"w{ax}", [query.get(f"w_gates_{ax}", [""])[0]])[0])
                    b_g = parse_gate_ranges(query.get(f"b{ax}", [query.get(f"b_gates_{ax}", [""])[0]])[0])
                    if w_g or b_g:
                        gate_specs[ax] = {"w": w_g, "b": b_g}

            axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]
            if gate_specs:
                gate_res = compute_3d_gate(reader, axis, gate_specs)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
                det_name = f"{axis_names[axis]} (Gated Coincidence)"
            else:
                spec0, spec1, spec2 = reader.get_projections_for_region(plane=plane, x0=x0, x1=x1, y0=y0, y1=y1)
                spec = spec0 if axis == 0 else (spec1 if axis == 1 else spec2)
                spec = np.array(spec, dtype=np.float64)
                det_name = f"{axis_names[axis]} (Projection)"

            if range_mode == "visible":
                ch_min = max(0, min(len(spec) - 2, int(float(query.get("ch_min", [0])[0]))))
                ch_max = max(ch_min + 1, min(len(spec) - 1, int(float(query.get("ch_max", [len(spec) - 1])[0]))))
            else:
                ch_min, ch_max = 0, len(spec) - 1

            axis_cal = session.get_cal(axis)
            is_cal = session.is_calibrated(axis)
            try:
                res = find_peaks_1d(
                    spec, ch_min=ch_min, ch_max=ch_max, method=method,
                    min_snr=min_snr, min_counts=min_counts, fwhm_est=fwhm_est, cal=axis_cal
                )
                res["axis"] = axis
                print_peaks_terminal_report(res, det_name, m["filename"], is_cal)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis, "peaks": [], "count": 0}
                print(f"[!] Peak search error on axis {axis}: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/fit_peak":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            axis = int(query.get("axis", [0])[0])
            channel = float(query.get("channel", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0]
            fwhm_mult = float(query.get("fwhm_mult", [4.0])[0])
            plane = query.get("plane", [session.active_plane])[0]

            x0 = int(float(query.get("x0", [0])[0]))
            x1 = int(float(query.get("x1", [4096])[0]))
            y0 = int(float(query.get("y0", [0])[0]))
            y1 = int(float(query.get("y1", [4096])[0]))

            # Check if gated
            gate_specs = {}
            for ax in (0, 1, 2):
                if ax != axis:
                    w_g = parse_gate_ranges(query.get(f"w{ax}", [""])[0])
                    b_g = parse_gate_ranges(query.get(f"b{ax}", [""])[0])
                    if w_g or b_g:
                        gate_specs[ax] = {"w": w_g, "b": b_g}

            gate_3rd_str = query.get("gate_3rd", [None])[0]
            gate_3rd_parsed = parse_gate_ranges(gate_3rd_str) if gate_3rd_str else None
            gate_3rd = gate_3rd_parsed[0] if gate_3rd_parsed else None

            if gate_specs:
                gate_res = compute_3d_gate(reader, axis, gate_specs)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
                det_name = f"{axis_names[axis]} (Gated Coincidence)"
            else:
                spec0, spec1, spec2 = reader.get_projections_for_region(
                    plane=plane, x0=x0, x1=x1, y0=y0, y1=y1, gate_3rd=gate_3rd
                )
                spec = spec0 if axis == 0 else (spec1 if axis == 1 else spec2)
                spec = np.array(spec, dtype=np.float64)
                det_name = f"{axis_names[axis]} (Projection)"

            axis_cal = session.get_cal(axis)
            is_cal = session.is_calibrated(axis)
            try:
                res = fit_gaussian_peak(
                    np.arange(len(spec)) + 0.5, spec, channel, fit_type=fit_type, fwhm_mult=fwhm_mult, cal=axis_cal
                )
                res["axis"] = axis
                print_fit_terminal_report(res, det_name, m["filename"], is_cal, verbosity="compact")
                if res.get("success") and session.fit_log_enabled:
                    append_fit_1d_result_to_file(session.fit_log_filename, res, is_cal)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis}
                print(f"[!] Peak fit error at channel {channel:.1f} on axis {axis}: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/fit_all_peaks_1d":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            axis = int(query.get("axis", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0].lower()
            fwhm_est = float(query.get("fwhm_est", [4.0])[0])
            min_snr = float(query.get("min_snr", [9.0])[0])
            plane = query.get("plane", [session.active_plane])[0]

            x0 = int(float(query.get("x0", [0])[0]))
            x1 = int(float(query.get("x1", [4096])[0]))
            y0 = int(float(query.get("y0", [0])[0]))
            y1 = int(float(query.get("y1", [4096])[0]))

            gate_specs = {}
            for ax in (0, 1, 2):
                if ax != axis:
                    w_g = parse_gate_ranges(query.get(f"w{ax}", [""])[0])
                    b_g = parse_gate_ranges(query.get(f"b{ax}", [""])[0])
                    if w_g or b_g:
                        gate_specs[ax] = {"w": w_g, "b": b_g}

            gate_3rd_str = query.get("gate_3rd", [None])[0]
            gate_3rd_parsed = parse_gate_ranges(gate_3rd_str) if gate_3rd_str else None
            gate_3rd = gate_3rd_parsed[0] if gate_3rd_parsed else None

            if gate_specs:
                gate_res = compute_3d_gate(reader, axis, gate_specs)
                spec = np.array(gate_res["net_spec"], dtype=np.float64)
            else:
                spec0, spec1, spec2 = reader.get_projections_for_region(
                    plane=plane, x0=x0, x1=x1, y0=y0, y1=y1, gate_3rd=gate_3rd
                )
                spec = spec0 if axis == 0 else (spec1 if axis == 1 else spec2)
                spec = np.array(spec, dtype=np.float64)

            ch_min = int(float(query.get("ch_min", [0])[0]))
            ch_max = int(float(query.get("ch_max", [len(spec) - 1])[0]))
            ch_min = max(0, min(len(spec) - 2, ch_min))
            ch_max = max(ch_min + 1, min(len(spec) - 1, ch_max))

            axis_cal = session.get_cal(axis)
            is_cal = session.is_calibrated(axis)

            peaks_str = query.get("peaks", [""])[0]
            peak_channels = []
            if peaks_str.strip():
                try:
                    peak_channels = [float(c.strip()) for c in peaks_str.split(",") if c.strip()]
                except Exception:
                    pass

            if not peak_channels:
                search_method = query.get("search_method", ["cwt"])[0].lower()
                search_res = find_peaks_1d(
                    spec, ch_min=ch_min, ch_max=ch_max, method=search_method, min_snr=min_snr, fwhm_est=fwhm_est, cal=axis_cal
                )
                peak_channels = [p["channel"] for p in search_res.get("peaks", [])]

            fwhm_mult = float(query.get("fwhm_mult", [4.0])[0])
            bg_method = query.get("bg_method", ["peak_aware"])[0].lower()

            t0 = time.time()
            try:
                res = fit_all_peaks_1d(
                    spec,
                    ch_min=ch_min,
                    ch_max=ch_max,
                    peak_channels=peak_channels,
                    fit_type=fit_type,
                    fwhm_est=fwhm_est,
                    cal=axis_cal,
                    fwhm_mult=fwhm_mult,
                    bg_method=bg_method,
                )
                res["axis"] = axis
                res["elapsed_ms"] = round((time.time() - t0) * 1000.0, 1)
                if res.get("success"):
                    det_name = axis_names[axis]
                    print_multi_fit_terminal_report(res, det_name, m["filename"], is_cal)
                    if session.fit_log_enabled:
                        for pk in res.get("peaks", []):
                            append_fit_1d_result_to_file(session.fit_log_filename, pk, is_cal)
            except Exception as e:
                res = {"success": False, "error": str(e), "axis": axis, "peaks": []}
                print(f"[!] Multi-peak fit error on axis {axis}: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/integrate_1d":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            axis = int(query.get("axis", [0])[0])
            reg_str = query.get("region", [""])[0]
            region = [float(v.strip()) for v in reg_str.split(",") if v.strip()] if reg_str.strip() else None

            if not region or len(region) < 2:
                res = {"success": False, "error": "Invalid region."}
            else:
                plane = query.get("plane", [session.active_plane])[0]
                x0 = int(float(query.get("x0", [0])[0]))
                x1 = int(float(query.get("x1", [4096])[0]))
                y0 = int(float(query.get("y0", [0])[0]))
                y1 = int(float(query.get("y1", [4096])[0]))

                gate_specs = {}
                for ax in (0, 1, 2):
                    if ax != axis:
                        w_g = parse_gate_ranges(query.get(f"w{ax}", [""])[0])
                        b_g = parse_gate_ranges(query.get(f"b{ax}", [""])[0])
                        if w_g or b_g:
                            gate_specs[ax] = {"w": w_g, "b": b_g}

                gate_3rd_str = query.get("gate_3rd", [None])[0]
                gate_3rd_parsed = parse_gate_ranges(gate_3rd_str) if gate_3rd_str else None
                gate_3rd = gate_3rd_parsed[0] if gate_3rd_parsed else None

                if gate_specs:
                    gate_res = compute_3d_gate(reader, axis, gate_specs)
                    spec = np.array(gate_res["net_spec"], dtype=np.float64)
                else:
                    spec0, spec1, spec2 = reader.get_projections_for_region(
                        plane=plane, x0=x0, x1=x1, y0=y0, y1=y1, gate_3rd=gate_3rd
                    )
                    spec = spec0 if axis == 0 else (spec1 if axis == 1 else spec2)
                    spec = np.array(spec, dtype=np.float64)

                bg_str = query.get("bg_regions", [""])[0]
                bg_regions = parse_gate_ranges(bg_str) if bg_str.strip() else None
                poly_str = query.get("poly_order", [""])[0]
                poly_order = int(poly_str) if (poly_str.isdigit() or (poly_str.startswith("-") and poly_str[1:].isdigit())) else None

                axis_cal = session.get_cal(axis)
                is_cal = session.is_calibrated(axis)

                res = integrate_peak_1d(spec, region, bg_regions=bg_regions, poly_order=poly_order, cal=axis_cal)
                res["axis"] = axis
                if res.get("success"):
                    session.integration_1d[axis] = res
                    det_name = axis_names[axis]
                    print_integrate_terminal_report(res, det_name, m["filename"], is_cal)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/fit_peak_2d":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            plane = query.get("plane", [session.active_plane])[0]
            x = float(query.get("x", [0])[0])
            y = float(query.get("y", [0])[0])
            fit_type = query.get("fit_type", ["gaussian"])[0]
            roi_half_width = int(float(query.get("roi_half_width", [16])[0]))

            if plane == "0-1":
                ax_x, ax_y = 0, 1
            elif plane == "0-2":
                ax_x, ax_y = 0, 2
            else:
                ax_x, ax_y = 1, 2

            cal_x = session.get_cal(ax_x)
            cal_y = session.get_cal(ax_y)
            is_cal_x = session.is_calibrated(ax_x)
            is_cal_y = session.is_calibrated(ax_y)

            gate_str = query.get("gate_3rd", [None])[0]
            gate_3rd = parse_gate_ranges(gate_str)[0] if (gate_str and gate_str.strip()) else session.gate_3rd

            mat_2d = reader.get_2d_plane(plane=plane, gate_3rd=gate_3rd)
            proj_x = np.sum(mat_2d, axis=0, dtype=np.float64)
            proj_y = np.sum(mat_2d, axis=1, dtype=np.float64)
            tot_counts = float(np.sum(mat_2d))

            try:
                res = fit_2d_gaussian_peak(
                    mat_2d,
                    x,
                    y,
                    fit_type=fit_type,
                    cal_x=cal_x,
                    cal_y=cal_y,
                    roi_half_width=roi_half_width,
                    proj_x=proj_x,
                    proj_y=proj_y,
                    total_counts=tot_counts,
                )
                print_fit_2d_terminal_report(res, m["filename"], (is_cal_x, is_cal_y), verbosity="compact")
                if res.get("success"):
                    target_axis = 2 if plane == "0-1" else (1 if plane == "0-2" else 0)
                    cal_z = session.get_cal(target_axis)
                    is_cal_z = session.is_calibrated(target_axis)
                    gamba_3rd = compute_2d_gamba_gate(reader, plane, res)
                    res["gamba_3rd_cut"] = gamba_3rd
                    if gamba_3rd.get("success"):
                        print_gamba_gate_terminal_report_3d(gamba_3rd, m["filename"], plane, cal_z, is_cal_z)
                    if session.fit_log_enabled:
                        append_fit_2d_result_to_file(session.fit_log_filename, res, (is_cal_x, is_cal_y))
            except Exception as e:
                res = {"success": False, "error": str(e), "is_2d": True}
                print(f"[!] 2D peak fit error: {e}", file=sys.stderr)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/export_pdf_1d":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            axis = int(query.get("axis", [0])[0])
            plane = query.get("plane", [session.active_plane])[0]
            x0 = int(float(query.get("x0", [0])[0]))
            x1 = int(float(query.get("x1", [4096])[0]))
            y0 = int(float(query.get("y0", [0])[0]))
            y1 = int(float(query.get("y1", [4096])[0]))
            is_log = int(query.get("is_log", [0])[0]) == 1

            spec0, spec1, spec2 = reader.get_projections_for_region(plane=plane, x0=x0, x1=x1, y0=y0, y1=y1)
            spec = spec0 if axis == 0 else (spec1 if axis == 1 else spec2)
            spec = np.array(spec, dtype=np.float64)

            axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]
            det_name = axis_names[axis]
            axis_cal = session.get_cal(axis)

            pdf_bytes = generate_pdf_1d(
                spec,
                0,
                len(spec) - 1,
                is_log=is_log,
                cal=axis_cal,
                axis_label=f"{det_name} Energy (keV)" if session.is_calibrated(axis) else f"{det_name} Channel",
                title=f"{m['filename']} - {det_name} Projection",
            )
            self.send_response(200)
            self.send_header("Content-type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{m["filename"]}_axis{axis}_spectrum.pdf"')
            self.end_headers()
            self.wfile.write(pdf_bytes)

        elif path == "/api/export_pdf_2d":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            plane = query.get("plane", [session.active_plane])[0]
            cmap_name = query.get("cmap", ["turbo"])[0]
            scale_mode = query.get("scale", ["log"])[0]
            vmin = float(query.get("vmin", [1.0])[0])
            vmax = float(query.get("vmax", [500.0])[0])

            if plane == "0-1":
                ax_x, ax_y = 0, 1
                max_w, max_h = reader.res1, reader.res2
            elif plane == "0-2":
                ax_x, ax_y = 0, 2
                max_w, max_h = reader.res1, reader.res3
            else:
                ax_x, ax_y = 1, 2
                max_w, max_h = reader.res2, reader.res3

            x0 = max(0, min(max_w - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(max_w, int(float(query.get("x1", [max_w])[0]))))
            y0 = max(0, min(max_h - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(max_h, int(float(query.get("y1", [max_h])[0]))))

            mat_2d = reader.get_2d_plane(plane=plane, gate_3rd=session.gate_3rd)
            axis_names = ["Axis 1 (X)", "Axis 2 (Y)", "Axis 3 (Z)"]

            pdf_bytes = generate_pdf_2d(
                mat_2d,
                x0,
                x1,
                y0,
                y1,
                cmap_name=cmap_name,
                scale_mode=scale_mode,
                vmin=vmin,
                vmax=vmax,
                cal_x=session.get_cal(ax_x),
                cal_y=session.get_cal(ax_y),
                x_label=f"{axis_names[ax_x]} (keV)" if session.is_calibrated(ax_x) else f"{axis_names[ax_x]} (Channel)",
                y_label=f"{axis_names[ax_y]} (keV)" if session.is_calibrated(ax_y) else f"{axis_names[ax_y]} (Channel)",
                title=f"{m['filename']} - 2D Plane {plane}",
            )
            self.send_response(200)
            self.send_header("Content-type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{m["filename"]}_plane_{plane}.pdf"')
            self.end_headers()
            self.wfile.write(pdf_bytes)

        elif path == "/api/export_dat":
            if not reader:
                self.send_error(404, "No matrix loaded")
                return

            axis = int(query.get("axis", [0])[0])
            plane = query.get("plane", [session.active_plane])[0]
            spec0, spec1, spec2 = reader.get_projections_for_region(plane=plane, x0=0, x1=4096, y0=0, y1=4096)
            spec = spec0 if axis == 0 else (spec1 if axis == 1 else spec2)

            cal = session.get_cal(axis)
            lines = [f"# Exported spectrum for {m['filename']} - Axis {axis}\n# Ch\tEnergy(keV)\tCounts\n"]
            for ch, val in enumerate(spec):
                en = cal[0] + cal[1] * (ch + 0.5) + cal[2] * ((ch + 0.5) ** 2)
                lines.append(f"{ch}\t{en:.3f}\t{val}\n")

            content = "".join(lines).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{m["filename"]}_axis{axis}.dat"')
            self.end_headers()
            self.wfile.write(content)

        elif path.startswith("/api/browse_fs"):
            req_path = query.get("path", [""])[0]
            res_data = browse_filesystem(req_path)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_data).encode("utf-8"))

        elif path.startswith("/api/select_matrix"):
            idx_str = query.get("index", ["0"])[0]
            try:
                idx = int(idx_str)
                session.select_matrix(idx)
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.get_metadata_dict()).encode("utf-8"))

        elif path.startswith("/api/open_file"):
            target = query.get("path", [""])[0]
            p = Path(target).resolve()
            if p.exists() and p.is_file():
                try:
                    new_idx = session.add_matrix_file(p)
                    session.select_matrix(new_idx)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(self.get_metadata_dict()).encode("utf-8"))
                    return
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
                    return
            self.send_error(400, "File not found or invalid")

        elif path == "/api/update_cal":
            axis = int(query.get("axis", [0])[0])
            coeffs_str = query.get("coeffs", [""])[0]
            if coeffs_str:
                session.set_cal(axis, coeffs_str)
                print(f"[*] Calibration updated for Axis {axis}: {session.get_cal(axis)}", flush=True)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "cal": session.get_cal(axis)}).encode("utf-8"))

        elif path == "/api/fit_log_status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "enabled": session.fit_log_enabled,
                "filename": session.fit_log_filename
            }).encode("utf-8"))

        elif path == "/api/set_fit_log":
            enabled_str = query.get("enabled", [""])[0].lower()
            filename = query.get("filename", [""])[0].strip()
            enabled = enabled_str in ("1", "true", "on", "yes")
            if not enabled_str and filename:
                enabled = session.fit_log_enabled
            status = session.set_fit_log(enabled, filename=filename if filename else None)
            print(f"[*] Fit logging {'ENABLED' if status['enabled'] else 'DISABLED'}: {status['filename']}", flush=True)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode("utf-8"))

        elif path == "/isotope_search" or path == "/isotope_id" or path.startswith("/isotope_search") or path.startswith("/isotope_id"):
            popup_path = Path(__file__).resolve().parent / "ensdf_popup.html"
            if not popup_path.exists():
                self.send_error(404, "ensdf_popup.html not found")
                return
            with open(popup_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

        elif path.startswith("/api/ensdf/status"):
            from ensdf_search import ENSDFSearchEngine
            engine = ENSDFSearchEngine()
            stats = engine.get_stats()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))

        elif path.startswith("/api/ensdf/files"):
            import datetime
            files = []
            for p in sorted(Path(".").glob("fit_results*.txt"), key=lambda x: x.stat().st_mtime, reverse=True):
                st = p.stat()
                mtime_str = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                files.append({
                    "name": p.name,
                    "path": str(p.resolve()),
                    "size_bytes": st.st_size,
                    "modified": mtime_str
                })
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "files": files}).encode("utf-8"))

        elif path.startswith("/api/ensdf/identify"):
            from ensdf_search import ENSDFSearchEngine, parse_human_duration
            engine = ENSDFSearchEngine()

            file_param = query.get("file", [""])[0].strip()
            gamma_str = (query.get("gamma") or query.get("energy") or [""])[0].strip()
            g1_str = (query.get("e1") or query.get("gamma1") or [""])[0].strip()
            g2_str = (query.get("e2") or query.get("gamma2") or [""])[0].strip()
            a_min_str = query.get("a_min", [""])[0].strip()
            a_max_str = query.get("a_max", [""])[0].strip()
            elems_str = (query.get("elements") or query.get("element") or [""])[0].strip()
            min_t12_str = query.get("min_t12", [""])[0].strip()
            max_t12_str = query.get("max_t12", [""])[0].strip()
            tol_str = query.get("tol", ["1.5"])[0].strip()
            ds_type = query.get("ds_type", ["all"])[0].strip()

            a_min = int(a_min_str) if (a_min_str.isdigit() or (a_min_str.startswith("-") and a_min_str[1:].isdigit())) else None
            a_max = int(a_max_str) if (a_max_str.isdigit() or (a_max_str.startswith("-") and a_max_str[1:].isdigit())) else None
            elements = [e.strip() for e in re.split(r"[\s,]+", elems_str) if e.strip()] if elems_str else None
            min_t12_s = parse_human_duration(min_t12_str)
            max_t12_s = parse_human_duration(max_t12_str)
            try:
                tol = float(tol_str)
            except ValueError:
                tol = 1.5

            res_data = {"success": True}
            try:
                if file_param:
                    res_data["type"] = "file"
                    rep = engine.identify_fit_results_file(
                        file_param, tol=tol, a_min=a_min, a_max=a_max,
                        elements=elements, min_t12_s=min_t12_s, max_t12_s=max_t12_s,
                        dataset_type=ds_type
                    )
                    res_data["report"] = rep
                    res_data["dominant_mass"] = rep.get("dominant_mass")
                    res_data["parsimonious_isotopes"] = rep.get("parsimonious_isotopes", [])
                    res_data["results_1d"] = rep.get("results_1d", [])
                    res_data["results_2d"] = rep.get("results_2d", [])
                elif g1_str and g2_str:
                    e1 = float(g1_str)
                    e2 = float(g2_str)
                    res_data["type"] = "2d"
                    res_data["e1"] = e1
                    res_data["e2"] = e2
                    candidates = engine.search_2d(
                        e1, e2, tol1=tol, tol2=tol, a_min=a_min, a_max=a_max,
                        elements=elements, min_t12_s=min_t12_s, max_t12_s=max_t12_s,
                        dataset_type=ds_type
                    )
                    res_data["candidates"] = candidates
                    res_data["results_2d"] = [{
                        "fit": {"energy1": e1, "energy2": e2, "energy1_err": 0.1, "energy2_err": 0.1},
                        "candidates": candidates,
                        "best_match": candidates[0] if candidates else None
                    }]
                    res_data["results_1d"] = []
                elif gamma_str:
                    eg = float(gamma_str)
                    res_data["type"] = "1d"
                    res_data["energy"] = eg
                    candidates = engine.search_1d(
                        eg, tol=tol, a_min=a_min, a_max=a_max,
                        elements=elements, min_t12_s=min_t12_s, max_t12_s=max_t12_s,
                        dataset_type=ds_type
                    )
                    res_data["candidates"] = candidates
                    res_data["results_1d"] = [{
                        "fit": {"energy": eg, "area": 0, "energy_err": 0.1},
                        "candidates": candidates,
                        "best_match": candidates[0] if candidates else None
                    }]
                    res_data["results_2d"] = []
                else:
                    res_data["success"] = False
                    res_data["error"] = "No search query provided. Specify 'file', 'gamma', or 'e1' & 'e2'."
            except Exception as e:
                res_data["success"] = False
                res_data["error"] = str(e)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_data).encode("utf-8"))

        elif path == "/halflife" or path == "/halflife_popup.html" or path == "/lifetime" or path.startswith("/halflife") or path.startswith("/lifetime"):
            popup_path = Path(__file__).resolve().parent / "halflife_popup.html"
            if not popup_path.exists():
                self.send_error(404, "halflife_popup.html not found")
                return
            with open(popup_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

        elif path.startswith("/api/halflife/files"):
            import datetime
            files = []
            patterns = ["*.dat", "*.txt", "*.fit"]
            seen = set()
            for pat in patterns:
                for p in sorted(Path(".").glob(pat), key=lambda x: x.stat().st_mtime, reverse=True):
                    if p.name in seen or p.name.startswith("."):
                        continue
                    seen.add(p.name)
                    st = p.stat()
                    mtime_str = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    files.append({
                        "name": p.name,
                        "path": str(p.resolve()),
                        "size_bytes": st.st_size,
                        "modified": mtime_str
                    })
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "files": files}).encode("utf-8"))

        elif path.startswith("/api/halflife/load"):
            from halflife import load_ascii_spectrum
            fn = query.get("file", [""])[0].strip()
            use_energy = int(query.get("use_energy", [0])[0]) == 1
            if not fn:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No file parameter provided"}).encode("utf-8"))
                return

            filepath = Path(fn)
            if not filepath.exists():
                alt = Path(__file__).resolve().parent / fn
                if alt.exists():
                    filepath = alt
                else:
                    self.send_response(404)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"File '{fn}' not found"}).encode("utf-8"))
                    return

            try:
                spec_obj = load_ascii_spectrum(filepath, use_energy=use_energy)
                resp = {
                    "success": True,
                    "filename": spec_obj.filename,
                    "x": spec_obj.x.tolist(),
                    "y": spec_obj.y.tolist(),
                    "dy": spec_obj.dy.tolist(),
                    "x_energy": spec_obj.x_energy.tolist() if spec_obj.x_energy is not None else None,
                    "header_lines": spec_obj.header_lines,
                    "is_gated": spec_obj.is_gated,
                    "bg_scale_factor": spec_obj.bg_scale_factor,
                    "x_label": spec_obj.x_label
                }
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path.startswith("/api/halflife/active_spectrum"):
            axis = int(query.get("axis", [0])[0])
            session = self.get_session()
            spec = session.get_1d_spectrum(axis)
            if spec is None or len(spec) == 0:
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"No spectrum available on Axis {axis}"}).encode("utf-8"))
                return

            m = session.get_active_matrix()
            mname = m['name'] if m else "matrix3d"
            cal = session.get_cal(axis)
            dy_arr = np.sqrt(np.maximum(spec, 1.0))
            x_arr = np.arange(len(spec), dtype=np.float64)
            x_energy = np.array([cal[0] + cal[1]*ch + (cal[2]*(ch**2) if len(cal)>2 else 0) for ch in x_arr], dtype=np.float64) if (cal[1] != 1.0 or cal[0] != 0.0) else None

            resp = {
                "success": True,
                "filename": f"{mname}_Axis{axis+1}.dat",
                "x": x_arr.tolist(),
                "y": spec.tolist(),
                "dy": dy_arr.tolist(),
                "x_energy": x_energy.tolist() if x_energy is not None else None,
                "header_lines": [f"# Active 1D Spectrum from 3D Matrix {mname} - Axis {axis+1}"],
                "is_gated": False,
                "bg_scale_factor": 0.0,
                "x_label": "Channel"
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif path.startswith("/api/halflife/export_pdf"):
            from halflife import HalfLifeFitter
            fn = query.get("file", ["spectrum.dat"])[0].strip()
            t12 = float(query.get("t12", [20.0])[0])
            fwhm = float(query.get("fwhm", [15.0])[0])
            centroid = float(query.get("centroid", [0.0])[0])
            scale = float(query.get("scale", [1000.0])[0])
            bg = float(query.get("bg", [0.0])[0])
            r0 = float(query.get("r0", [0.0])[0])
            r1 = float(query.get("r1", [0.0])[0])
            is_log = int(query.get("log", [0])[0]) == 1

            fitter = HalfLifeFitter()
            filepath = Path(fn)
            if not filepath.exists():
                filepath = Path(__file__).resolve().parent / fn

            try:
                if filepath.exists():
                    fitter.load_data(filepath)
                else:
                    session = self.get_session()
                    spec = session.get_1d_spectrum(0)
                    fitter.set_data(np.arange(len(spec)), spec)

                res = fitter.fit(
                    t12=t12, fwhm=fwhm, centroid=centroid, scale=scale, bg=bg,
                    freepars=[False, False, False, False, False],
                    fit_range=(r0, r1) if (r1 > r0) else None
                )

                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_pdf_path = Path(tmp.name)

                fitter.export_plot(tmp_pdf_path, log_scale=is_log)
                pdf_bytes = tmp_pdf_path.read_bytes()
                try:
                    tmp_pdf_path.unlink()
                except Exception:
                    pass

                self.send_response(200)
                self.send_header("Content-type", "application/pdf")
                self.send_header("Content-Disposition", f'attachment; filename="{Path(fn).stem}_halflife_fit.pdf"')
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
        session = self.get_session()
        m = session.get_active_matrix()

        if self.path == "/api/save_config":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode("utf-8"))

                if session.config is None:
                    session.config = DEFAULT_CONFIG.copy()

                session.config.update(data)
                target_path = Path.cwd() / CONFIG_FILENAME
                save_config_file(target_path, session.config)
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

                new_idx = session.add_matrix_file(target_path)
                session.select_matrix(new_idx)
                print(f"\n[+] Loaded 3D matrix file: {target_path.name}", flush=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                info = self.get_metadata_dict()
                self.wfile.write(json.dumps(info).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

        elif self.path == "/api/halflife/fit":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                from halflife import HalfLifeFitter
                fitter = HalfLifeFitter()
                x_arr = np.array(data["x"], dtype=np.float64)
                y_arr = np.array(data["y"], dtype=np.float64)
                dy_arr = np.array(data["dy"], dtype=np.float64) if "dy" in data else None
                fitter.set_data(x_arr, y_arr, dy=dy_arr)

                fit_range = tuple(data["fit_range"]) if ("fit_range" in data and data["fit_range"]) else None
                freepars = data.get("freepars", [True, True, True, True, False])

                res = fitter.fit(
                    t12=data.get("t12"),
                    fwhm=data.get("fwhm"),
                    centroid=data.get("centroid"),
                    scale=data.get("scale"),
                    bg=data.get("bg"),
                    freepars=freepars,
                    fit_range=fit_range
                )

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif self.path == "/api/halflife/scan_bg":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                from halflife import HalfLifeFitter
                fitter = HalfLifeFitter()
                x_arr = np.array(data["x"], dtype=np.float64)
                y_arr = np.array(data["y"], dtype=np.float64)
                dy_arr = np.array(data["dy"], dtype=np.float64) if "dy" in data else None
                fitter.set_data(x_arr, y_arr, dy=dy_arr)

                fit_range = tuple(data["fit_range"]) if ("fit_range" in data and data["fit_range"]) else None
                fitter.active_range = fit_range
                fitter.pars = [
                    float(data.get("t12", 20.0)),
                    float(data.get("fwhm", 15.0)),
                    float(data.get("centroid", 0.0)),
                    float(data.get("scale", 1000.0)),
                    float(data.get("bg", 0.0))
                ]
                fitter.freepars = data.get("freepars", [True, True, True, True, False])

                b_min = float(data["b_min"]) if "b_min" in data else None
                b_max = float(data["b_max"]) if "b_max" in data else None
                steps = int(data.get("steps", 21))

                scan_res = fitter.scan_background(b_min=b_min, b_max=b_max, steps=steps, apply_best=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(scan_res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif self.path == "/api/halflife/compress":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                from halflife import HalfLifeFitter
                fitter = HalfLifeFitter()
                x_arr = np.array(data["x"], dtype=np.float64)
                y_arr = np.array(data["y"], dtype=np.float64)
                dy_arr = np.array(data["dy"], dtype=np.float64) if "dy" in data else None
                fitter.set_data(x_arr, y_arr, dy=dy_arr)
                fitter.spec.filename = data.get("filename", "compressed")

                factor = int(data.get("factor", 2))
                new_spec = fitter.compress(factor)

                resp = {
                    "success": True,
                    "filename": new_spec.filename,
                    "x": new_spec.x.tolist(),
                    "y": new_spec.y.tolist(),
                    "dy": new_spec.dy.tolist(),
                    "x_energy": None,
                    "is_gated": False,
                    "bg_scale_factor": 0.0,
                    "x_label": new_spec.x_label
                }
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")


def main():
    config_path = Path.cwd() / CONFIG_FILENAME
    config = load_or_create_config(config_path)

    parser = argparse.ArgumentParser(
        description="Launch modern Web-based interactive 3D matrix viewer for GASPware/gsort .cmat files."
    )
    parser.add_argument(
        "input",
        nargs="*",
        type=str,
        default=[],
        help="Path to one or more input 3D .cmat file(s) (e.g. GeE-Rings3D.cmat)",
    )
    parser.add_argument(
        "-H", "--host",
        type=str,
        default=None,
        help=f"Web server host/interface (default: {config.get('host', '0.0.0.0')})",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help=f"Web server port (default: {config.get('port', 8081)})",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default=None,
        help="Default 2D plane to display: '0-1' (Axis 1 vs Axis 2), '0-2' (Axis 1 vs Axis 3), or '1-2' (Axis 2 vs Axis 3)",
    )
    parser.add_argument(
        "--cal",
        nargs="+",
        type=float,
        metavar="COEFF",
        default=None,
        help="Global calibration coefficients applied to all axes: a0 a1 [a2]",
    )
    parser.add_argument(
        "--cal-0", "--cal-x", "--cal-1",
        nargs="+",
        type=float,
        dest="cal_0",
        metavar="COEFF",
        default=None,
        help="Axis 1 (X) calibration coefficients: a0 a1 [a2]",
    )
    parser.add_argument(
        "--cal-1-axis", "--cal-y", "--cal-2",
        nargs="+",
        type=float,
        dest="cal_1",
        metavar="COEFF",
        default=None,
        help="Axis 2 (Y) calibration coefficients: a0 a1 [a2]",
    )
    parser.add_argument(
        "--cal-2-axis", "--cal-z", "--cal-3",
        nargs="+",
        type=float,
        dest="cal_2",
        metavar="COEFF",
        default=None,
        help="Axis 3 (Z) calibration coefficients: a0 a1 [a2]",
    )
    parser.add_argument(
        "--fit-log", "--log-fits",
        nargs="?",
        const=True,
        default=None,
        metavar="FILENAME",
        help="Enable writing 1D/2D Gaussian fit results to a text file (default: fit_results_<timestamp>.txt)",
    )
    parser.add_argument(
        "-b", "--browser",
        type=str,
        default=None,
        help="Specify browser to open (e.g. 'chrome', 'firefox', 'safari', or 'none')",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Do not automatically open web browser",
    )

    args = parser.parse_args()

    session = MatrixSession3D(config)
    CMAT3DWebHandler.session = session

    if args.fit_log is not None:
        if isinstance(args.fit_log, str) and args.fit_log.strip():
            session.set_fit_log(True, filename=args.fit_log.strip())
        else:
            session.set_fit_log(True)

    if args.cal is not None:
        c = parse_cal_coefficients(args.cal)
        session.set_cal(0, c)
        session.set_cal(1, c)
        session.set_cal(2, c)
    if args.cal_0 is not None:
        session.set_cal(0, parse_cal_coefficients(args.cal_0))
    if args.cal_1 is not None:
        session.set_cal(1, parse_cal_coefficients(args.cal_1))
    if args.cal_2 is not None:
        session.set_cal(2, parse_cal_coefficients(args.cal_2))

    if args.plane:
        session.set_active_plane(args.plane)

    # Load input files
    input_files = []
    for item in args.input:
        matches = glob.glob(item)
        if matches:
            input_files.extend(matches)
        else:
            input_files.append(item)

    if not input_files:
        # Check if default GeE-Rings3D.cmat exists in cwd
        default_3d = Path.cwd() / "GeE-Rings3D.cmat"
        if default_3d.exists():
            input_files.append(str(default_3d))

    for f_path in input_files:
        p = Path(f_path).resolve()
        if p.exists() and p.is_file():
            try:
                session.add_matrix_file(p)
            except Exception as e:
                print(f"[!] Error loading matrix {p.name}: {e}", file=sys.stderr)

    if not session.matrices:
        print("[!] No valid 3D .cmat matrices specified or found.", file=sys.stderr)
        sys.exit(1)

    host = args.host if args.host is not None else str(config.get("host", "0.0.0.0")).strip()
    port = args.port if args.port is not None else int(config.get("port", 8081))
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

    print(f"[*] Loaded {len(session.matrices)} matrix file{'s' if len(session.matrices) > 1 else ''}:")
    for idx, m in enumerate(session.matrices):
        shape_str = "×".join(str(s) for s in m["shape"])
        print(f"    [{idx + 1}/{len(session.matrices)}] '{m['name']}' ({shape_str}, {m['total_counts']:,} counts)")

    bind_host = "" if host in ("0.0.0.0", "", "::") else host
    server_address = (bind_host, port)
    try:
        httpd = HTTPServerClass(server_address, CMAT3DWebHandler)
    except OSError as e:
        if "Address already in use" in str(e):
            port += 1
            server_address = (bind_host, port)
            httpd = HTTPServerClass(server_address, CMAT3DWebHandler)
        else:
            raise

    local_ip = get_local_ip()
    if host in ("0.0.0.0", "", "::"):
        target_url = f"http://{local_ip}:{port}"
    elif host in ("127.0.0.1", "localhost"):
        target_url = f"http://127.0.0.1:{port}"
    else:
        target_url = f"http://{host}:{port}"

    print(f"\n[+] Interactive 3D CMAT Web Viewer is ready!")
    print(f"[+] Access URL:                {target_url}  (Ctrl+Click to open)")
    print(f"[+] Triple Binned 1D Histograms with real-time mouse inspector and 2D projection planes.")

    if in_ssh:
        print(f"\n[*] SSH session detected:")
        if ssh_browser_skipped:
            print(f"    - Remote browser auto-launch skipped to keep your SSH session fast.")
        print(f"    - Ctrl+Click (or Cmd+Click on macOS) on the URL above to open in your local browser.")
        print(f"    - (Or use SSH port forwarding: ssh -L {port}:localhost:{port} user@server)")

    print(f"\n[+] Press Ctrl+C in terminal to stop server.\n")

    if open_browser:
        threading.Timer(0.6, lambda: launch_browser(target_url, browser_choice)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
