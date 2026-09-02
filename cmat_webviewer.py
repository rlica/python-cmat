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
      * CTRL + Left Drag: Box zoom into rectangle
      * Left / Right Arrow: Set Left (Xmin) and Right (Xmax) limit markers at cursor
      * Down / Up Arrow: Set Down (Ymin) and Up (Ymax) limit markers at cursor
      * E / e: Expand / Zoom into set limit markers
      * F / f: Full matrix view (reset zoom)
      * CTRL + Arrows: Shift / Pan viewport
      * CTRL + Click / M: Center viewport on cursor
      * P / X / Y: Toggle 1D Projection Axis (Det 1 vs Det 2)
      * 1 / 2 / 4 (or L): Switch Linear, Sqrt, Log color scale
      * C / c: Cycle color gradients (Turbo, Viridis, Plasma, Inferno, Hot, Jet, Gray)
      * H / ?: Toggle keyboard shortcuts modal

Usage:
  python3 cmat_webviewer.py GeE-symm.cmat --port 8080
"""

import sys
import json
import argparse
import webbrowser
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from cmat import CMATReader


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

        elif self.path.startswith("/api/projection_region"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            x0 = max(0, min(self.matrix.shape[1] - 1, int(float(query.get("x0", [0])[0]))))
            x1 = max(x0 + 1, min(self.matrix.shape[1], int(float(query.get("x1", [self.matrix.shape[1]])[0]))))
            y0 = max(0, min(self.matrix.shape[0] - 1, int(float(query.get("y0", [0])[0]))))
            y1 = max(y0 + 1, min(self.matrix.shape[0], int(float(query.get("y1", [self.matrix.shape[0]])[0]))))
            axis = int(query.get("axis", [0])[0])

            if axis == 0:
                # X Projection (Det 1): Sum along Y axis between y0 and y1
                if y0 == 0 and y1 >= self.matrix.shape[0]:
                    spec = self.proj
                else:
                    spec = np.sum(self.matrix[y0:y1, :], axis=0, dtype=np.int64)
            else:
                # Y Projection (Det 2): Sum along X axis between x0 and x1
                if x0 == 0 and x1 >= self.matrix.shape[1]:
                    spec = np.sum(self.matrix, axis=1, dtype=np.int64)
                else:
                    spec = np.sum(self.matrix[:, x0:x1], axis=1, dtype=np.int64)

            resp = {
                "axis": axis,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "spectrum": spec.tolist(),
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
    overflow: hidden;
  }
  #canvas2d { width: 100%; height: 100%; display: block; cursor: crosshair; }
  .panel-1d {
    flex: 4.5;
    background: #181818;
    border-radius: 6px;
    border: 1px solid #2a2a2a;
    display: flex;
    flex-direction: column;
    padding: 8px;
  }
  #canvas1d { width: 100%; flex: 1; background: #111; border-radius: 4px; cursor: crosshair; }
  .panel-1d-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .panel-1d-title { font-size: 0.82rem; font-weight: bold; color: #ff9800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .overlay-status {
    position: absolute;
    bottom: 8px;
    left: 60px;
    background: rgba(18,18,18,0.85);
    border: 1px solid #00e5ff;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.78rem;
    color: #00e5ff;
    pointer-events: none;
    font-family: monospace;
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
    <div class="control-group">
      <h3>1D Histogram Projection [P]</h3>
      <label>Projected Spectrum Axis</label>
      <select id="projAxisSelect">
        <option value="0" selected>Det 1 (X Projection, Slice Y)</option>
        <option value="1">Det 2 (Y Projection, Slice X)</option>
      </select>

      <label style="margin-top: 6px;">1D Energy / Channel Range</label>
      <select id="projRangeSelect">
        <option value="synced" selected>Synced with 2D Window Zoom</option>
        <option value="full">Full 0..4095 Range</option>
      </select>

      <label style="margin-top: 6px;">1D Vertical Scale</label>
      <select id="projScaleSelect">
        <option value="log" selected>Logarithmic</option>
        <option value="linear">Linear</option>
      </select>
    </div>

    <div class="control-group">
      <h3>Zoom & Limits (cmat)</h3>
      <button id="btnFull" class="secondary">Full Matrix View [F]</button>
      <button id="btnExpand" style="background:#00838f;">Expand Limits [E]</button>
      <button id="btnResetMarkers" class="secondary" style="margin-top:4px;">Clear Limit Markers</button>
    </div>

    <div class="control-group">
      <h3>Color Gradient (2D)</h3>
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

      <label style="margin-top: 6px;">Scale Mode [1, 2, 4]</label>
      <select id="scaleSelect">
        <option value="log" selected>Logarithmic (LogNorm)</option>
        <option value="sqrt">Power / Sqrt</option>
        <option value="linear">Linear</option>
      </select>
    </div>

    <div class="control-group">
      <h3>Contrast / Cutoffs</h3>
      <label>Max Contrast <span class="range-val" id="vmaxLabel">1000</span></label>
      <input type="range" id="vmaxSlider" min="1" max="10000" value="500">

      <label style="margin-top: 6px;">Min Threshold <span class="range-val" id="vminLabel">1</span></label>
      <input type="range" id="vminSlider" min="0" max="50" value="1">
      <button id="btnHelp" class="secondary" style="background:#4a148c; margin-top:8px;">Help / Shortcuts [?]</button>
    </div>
  </aside>

  <div class="workspace">
    <div class="panel-2d">
      <canvas id="canvas2d" tabindex="0"></canvas>
      <div class="overlay-status" id="overlayStatus">Limits: [L: - | R: - | D: - | U: -] -> Press 'E' to Expand, 'F' Full</div>
    </div>
    <div class="panel-1d">
      <div class="panel-1d-header">
        <span class="panel-1d-title" id="specTitle">1D Auto-Projection Histogram</span>
        <button id="btnExport1D" style="width: auto; padding: 3px 8px; font-size: 0.75rem;">Export .dat</button>
      </div>
      <canvas id="canvas1d"></canvas>
    </div>
  </div>
</main>

<div class="help-modal" id="helpModal">
  <h2>GASPware cmat Navigation Shortcuts</h2>
  <table>
    <tr><td>Ctrl + Drag</td><td>Box Zoom into rectangle</td></tr>
    <tr><td>Left Arrow (←)</td><td>Set Left limit (Xmin) at cursor</td></tr>
    <tr><td>Right Arrow (→)</td><td>Set Right limit (Xmax) at cursor</td></tr>
    <tr><td>Down Arrow (↓)</td><td>Set Down limit (Ymin) at cursor</td></tr>
    <tr><td>Up Arrow (↑)</td><td>Set Up limit (Ymax) at cursor</td></tr>
    <tr><td>E / e</td><td>Expand / Zoom into set limits</td></tr>
    <tr><td>F / f</td><td>Full matrix view (zoom out)</td></tr>
    <tr><td>P / X / Y</td><td>Toggle 1D Projection Axis (Det 1 vs Det 2)</td></tr>
    <tr><td>Ctrl + Arrows</td><td>Shift / Pan viewport</td></tr>
    <tr><td>Ctrl + Click / M</td><td>Center view on cursor point</td></tr>
    <tr><td>1 / 2 / 4 (or L)</td><td>Switch Linear, Sqrt, Log color scale</td></tr>
    <tr><td>C / c</td><td>Cycle colormaps</td></tr>
    <tr><td>Esc / ?</td><td>Close Help</td></tr>
  </table>
  <button onclick="document.getElementById('helpModal').style.display='none'" style="margin-top: 15px;">Close</button>
</div>

<script>
  let metadata = null;
  let currentProjSpec = null;
  let currentProjSlice = { x0: 0, x1: 4096, y0: 0, y1: 4096, axis: 0 };

  // Viewport in channel coordinates
  let view = { x0: 0, x1: 4096, y0: 0, y1: 4096 };

  // GASPware Markers [Left, Right, Down, Up]
  let markers = { left: null, right: null, down: null, up: null };

  // Mouse / Drag State
  let isBoxZooming = false;
  let isPanning = false;
  let dragStartPos = { x: 0, y: 0 };
  let mouseCurrentPos = { x: 0, y: 0 };
  let cursorChannel = { x: 0, y: 0 };
  let cursor1DChannel = null;

  const canvas2d = document.getElementById('canvas2d');
  const ctx2d = canvas2d.getContext('2d');
  const canvas1d = document.getElementById('canvas1d');
  const ctx1d = canvas1d.getContext('2d');

  // Tile Data & Shape
  let currentTileData = null;
  let tileW = 0, tileH = 0;

  // Plot Margins
  const margin2D = { left: 55, bottom: 40, top: 12, right: 15 };
  const margin1D = { left: 58, bottom: 38, top: 14, right: 18 };

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

  function getPlotRect1D() {
    return {
      x: margin1D.left,
      y: margin1D.top,
      w: Math.max(10, canvas1d.width - margin1D.left - margin1D.right),
      h: Math.max(10, canvas1d.height - margin1D.top - margin1D.bottom)
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
        r = Math.min(255, t*280);
        g = Math.min(255, Math.max(0, (t-0.3)*300));
        b = Math.min(255, Math.max(0, (t-0.7)*500));
      } else if (cmap === 'hot') {
        r = Math.min(255, t*3*255);
        g = Math.min(255, Math.max(0, (t-0.33)*3*255));
        b = Math.min(255, Math.max(0, (t-0.66)*3*255));
      } else if (cmap === 'gray') {
        r = g = b = (t * 255);
      } else {
        r = Math.max(0, Math.min(255, (1.5 - Math.abs(t * 4 - 3)) * 255));
        g = Math.max(0, Math.min(255, (1.5 - Math.abs(t * 4 - 2)) * 255));
        b = Math.max(0, Math.min(255, (1.5 - Math.abs(t * 4 - 1)) * 255));
      }
      colorLUT[i] = rgba32(r | 0, g | 0, b | 0, 255);
    }
  }

  async function init() {
    const metaRes = await fetch('/api/metadata');
    metadata = await metaRes.json();
    document.getElementById('matInfo').innerText = `${metadata.filename} (${metadata.shape[0]}x${metadata.shape[1]})`;

    view.x0 = 0; view.x1 = metadata.shape[0];
    view.y0 = 0; view.y1 = metadata.shape[1];

    document.getElementById('vmaxSlider').max = Math.max(100, Math.min(50000, metadata.max_count));
    document.getElementById('vmaxSlider').value = Math.min(800, metadata.max_count);
    document.getElementById('vmaxLabel').innerText = document.getElementById('vmaxSlider').value;

    updateColorLUT();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    setupEvents();
    fetchTileAndRender();
    fetch1DProjection();
  }

  function resizeCanvas() {
    canvas2d.width = canvas2d.clientWidth;
    canvas2d.height = canvas2d.clientHeight;
    canvas1d.width = canvas1d.clientWidth;
    canvas1d.height = canvas1d.clientHeight;
    fetchTileAndRender();
    render1DSpectrum();
  }

  let tileFetchDebounce = null;
  let projFetchDebounce = null;

  async function fetchTileAndRender() {
    const pr = getPlotRect2D();
    const x0 = Math.floor(Math.max(0, view.x0));
    const x1 = Math.ceil(Math.min(metadata.shape[0], view.x1));
    const y0 = Math.floor(Math.max(0, view.y0));
    const y1 = Math.ceil(Math.min(metadata.shape[1], view.y1));

    const targetW = Math.round(pr.w);
    const targetH = Math.round(pr.h);

    const url = `/api/tile?x0=${x0}&x1=${x1}&y0=${y0}&y1=${y1}&w=${targetW}&h=${targetH}`;
    const res = await fetch(url);
    tileW = parseInt(res.headers.get('X-Shape-W')) || 1;
    tileH = parseInt(res.headers.get('X-Shape-H')) || 1;

    const buf = await res.arrayBuffer();
    currentTileData = new Int32Array(buf);
    render2D();
  }

  async function fetch1DProjection() {
    const axis = parseInt(document.getElementById('projAxisSelect').value);
    const x0 = Math.floor(Math.max(0, view.x0));
    const x1 = Math.ceil(Math.min(metadata.shape[0], view.x1));
    const y0 = Math.floor(Math.max(0, view.y0));
    const y1 = Math.ceil(Math.min(metadata.shape[1], view.y1));

    const url = `/api/projection_region?x0=${x0}&x1=${x1}&y0=${y0}&y1=${y1}&axis=${axis}`;
    const res = await fetch(url);
    const data = await res.json();
    currentProjSpec = data.spectrum;
    currentProjSlice = data;

    // Update Title
    const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
    if (axis === 0) {
      if (y0 === 0 && y1 >= metadata.shape[1]) {
        document.getElementById('specTitle').innerText = `1D Total Projection: Det 1 (X Axis)`;
      } else {
        const e0 = chToEnergy(y0).toFixed(1);
        const e1 = chToEnergy(y1).toFixed(1);
        document.getElementById('specTitle').innerText = isCal
          ? `Det 1 (X Proj) | Sliced Det 2 (Y): ${e0} - ${e1} keV (ch ${y0}-${y1})`
          : `Det 1 (X Proj) | Sliced Det 2 (Y): ch ${y0} - ${y1}`;
      }
    } else {
      if (x0 === 0 && x1 >= metadata.shape[0]) {
        document.getElementById('specTitle').innerText = `1D Total Projection: Det 2 (Y Axis)`;
      } else {
        const e0 = chToEnergy(x0).toFixed(1);
        const e1 = chToEnergy(x1).toFixed(1);
        document.getElementById('specTitle').innerText = isCal
          ? `Det 2 (Y Proj) | Sliced Det 1 (X): ${e0} - ${e1} keV (ch ${x0}-${x1})`
          : `Det 2 (Y Proj) | Sliced Det 1 (X): ch ${x0} - ${x1}`;
      }
    }

    render1DSpectrum();
  }

  function render2D() {
    const cw = canvas2d.width, ch = canvas2d.height;
    ctx2d.clearRect(0, 0, cw, ch);

    const pr = getPlotRect2D();

    // 1. Direct 32-bit Uint32 buffer rendering
    if (currentTileData && tileW > 0 && tileH > 0) {
      const imgData = ctx2d.createImageData(tileW, tileH);
      const buf32 = new Uint32Array(imgData.data.buffer);

      const vmin = parseInt(document.getElementById('vminSlider').value);
      const vmax = parseInt(document.getElementById('vmaxSlider').value);
      const scaleMode = document.getElementById('scaleSelect').value;

      const isLog = (scaleMode === 'log');
      const isSqrt = (scaleMode === 'sqrt');
      const logMin = Math.log(Math.max(1, vmin));
      const logScale = 255.0 / (Math.log(Math.max(2, vmax)) - logMin);
      const sqrtScale = 255.0 / Math.sqrt(Math.max(1, vmax));
      const linScale = 255.0 / Math.max(1, vmax - vmin);

      const len = currentTileData.length;
      for (let i = 0; i < len; i++) {
        let val = currentTileData[i];
        if (val <= vmin) {
          buf32[i] = 0xFF000000;
        } else {
          let idx;
          if (isLog) {
            idx = (Math.log(val) - logMin) * logScale;
          } else if (isSqrt) {
            idx = Math.sqrt(val) * sqrtScale;
          } else {
            idx = (val - vmin) * linScale;
          }
          if (idx < 0) idx = 0;
          else if (idx > 255) idx = 255;
          buf32[i] = colorLUT[idx | 0];
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

    const numTicksX = 6;
    for (let i = 0; i <= numTicksX; i++) {
      const frac = i / numTicksX;
      const chVal = Math.round(view.x0 + frac * (view.x1 - view.x0));
      const px = pr.x + frac * pr.w;
      ctx2d.beginPath();
      ctx2d.moveTo(px, pr.y + pr.h);
      ctx2d.lineTo(px, pr.y + pr.h + 5);
      ctx2d.stroke();
      ctx2d.fillText(chVal.toString(), px, pr.y + pr.h + 16);
    }
    ctx2d.fillText('Det 1 (Channel)', pr.x + pr.w / 2, pr.y + pr.h + 30);

    ctx2d.textAlign = 'right';
    const numTicksY = 6;
    for (let i = 0; i <= numTicksY; i++) {
      const frac = i / numTicksY;
      const chVal = Math.round(view.y0 + frac * (view.y1 - view.y0));
      const py = pr.y + pr.h - frac * pr.h;
      ctx2d.beginPath();
      ctx2d.moveTo(pr.x, py);
      ctx2d.lineTo(pr.x - 5, py);
      ctx2d.stroke();
      ctx2d.fillText(chVal.toString(), pr.x - 8, py + 3);
    }

    // 3. Limit Markers (Left, Right, Down, Up)
    ctx2d.save();
    ctx2d.beginPath();
    ctx2d.rect(pr.x, pr.y, pr.w, pr.h);
    ctx2d.clip();

    ctx2d.lineWidth = 1.5;
    if (markers.left !== null) {
      let pt = chToPx2D(markers.left, 0);
      ctx2d.strokeStyle = '#00e5ff';
      ctx2d.setLineDash([5, 5]);
      ctx2d.beginPath(); ctx2d.moveTo(pt.x, pr.y); ctx2d.lineTo(pt.x, pr.y + pr.h); ctx2d.stroke();
    }
    if (markers.right !== null) {
      let pt = chToPx2D(markers.right, 0);
      ctx2d.strokeStyle = '#00e5ff';
      ctx2d.setLineDash([2, 4]);
      ctx2d.beginPath(); ctx2d.moveTo(pt.x, pr.y); ctx2d.lineTo(pt.x, pr.y + pr.h); ctx2d.stroke();
    }
    if (markers.down !== null) {
      let pt = chToPx2D(0, markers.down);
      ctx2d.strokeStyle = '#ff4081';
      ctx2d.setLineDash([5, 5]);
      ctx2d.beginPath(); ctx2d.moveTo(pr.x, pt.y); ctx2d.lineTo(pr.x + pr.w, pt.y); ctx2d.stroke();
    }
    if (markers.up !== null) {
      let pt = chToPx2D(0, markers.up);
      ctx2d.strokeStyle = '#ff4081';
      ctx2d.setLineDash([2, 4]);
      ctx2d.beginPath(); ctx2d.moveTo(pr.x, pt.y); ctx2d.lineTo(pr.x + pr.w, pt.y); ctx2d.stroke();
    }
    ctx2d.setLineDash([]);

    // 4. Box Zoom Rectangle
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
    ctx2d.restore();
  }

  function render1DSpectrum() {
    const cw = canvas1d.width, ch = canvas1d.height;
    ctx1d.clearRect(0, 0, cw, ch);
    const spec = currentProjSpec;
    if (!spec || spec.length === 0) return;

    const pr = getPlotRect1D();
    const axis = parseInt(document.getElementById('projAxisSelect').value);
    const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
    const isLog = (document.getElementById('projScaleSelect').value === 'log');

    // Determine 1D channel bounds
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

    // Find max value in visible range
    let maxVal = 0;
    for (let i = chStart; i <= chEnd; i++) {
      if (spec[i] > maxVal) maxVal = spec[i];
    }
    if (maxVal === 0) maxVal = 1;

    const logMax = Math.log10(Math.max(10, maxVal));

    // Draw Plot Frame
    ctx1d.strokeStyle = '#444';
    ctx1d.lineWidth = 1;
    ctx1d.strokeRect(pr.x, pr.y, pr.w, pr.h);

    // Spectrum Binned Histogram (Staircase / Step-plot)
    ctx1d.save();
    ctx1d.beginPath();
    ctx1d.rect(pr.x, pr.y, pr.w, pr.h);
    ctx1d.clip();

    const span = chEnd - chStart + 1; // total bins in view
    const strokeColor = (axis === 0) ? '#00e5ff' : '#ff9800';
    const fillColor = (axis === 0) ? 'rgba(0, 229, 255, 0.12)' : 'rgba(255, 152, 0, 0.12)';

    // 1. Draw Filled Stepped Histogram
    ctx1d.beginPath();
    ctx1d.moveTo(pr.x, pr.y + pr.h);

    for (let i = chStart; i <= chEnd; i++) {
      let xL = pr.x + ((i - chStart) / span) * pr.w;
      let xR = pr.x + ((i + 1 - chStart) / span) * pr.w;
      let val = spec[i];
      let yNorm = isLog ? (val > 0 ? Math.log10(val) / logMax : 0) : val / maxVal;
      yNorm = Math.max(0, Math.min(1, yNorm));
      let y = pr.y + pr.h * (1 - yNorm);

      ctx1d.lineTo(xL, y);
      ctx1d.lineTo(xR, y);
    }
    ctx1d.lineTo(pr.x + pr.w, pr.y + pr.h);
    ctx1d.closePath();
    ctx1d.fillStyle = fillColor;
    ctx1d.fill();

    // 2. Draw Stepped Histogram Outline
    ctx1d.beginPath();
    ctx1d.moveTo(pr.x, pr.y + pr.h);
    for (let i = chStart; i <= chEnd; i++) {
      let xL = pr.x + ((i - chStart) / span) * pr.w;
      let xR = pr.x + ((i + 1 - chStart) / span) * pr.w;
      let val = spec[i];
      let yNorm = isLog ? (val > 0 ? Math.log10(val) / logMax : 0) : val / maxVal;
      yNorm = Math.max(0, Math.min(1, yNorm));
      let y = pr.y + pr.h * (1 - yNorm);

      ctx1d.lineTo(xL, y);
      ctx1d.lineTo(xR, y);
    }
    ctx1d.lineTo(pr.x + pr.w, pr.y + pr.h);
    ctx1d.strokeStyle = strokeColor;
    ctx1d.lineWidth = 1.2;
    ctx1d.stroke();

    // 3. Highlight Hovered Bin & Crosshair Tracker
    if (cursor1DChannel !== null && cursor1DChannel >= chStart && cursor1DChannel <= chEnd) {
      let xL = pr.x + ((cursor1DChannel - chStart) / span) * pr.w;
      let xR = pr.x + ((cursor1DChannel + 1 - chStart) / span) * pr.w;
      let val = spec[cursor1DChannel] || 0;
      let yNorm = isLog ? (val > 0 ? Math.log10(val) / logMax : 0) : val / maxVal;
      yNorm = Math.max(0, Math.min(1, yNorm));
      let y = pr.y + pr.h * (1 - yNorm);

      // Highlight active bin bar
      ctx1d.fillStyle = 'rgba(255, 214, 0, 0.35)';
      ctx1d.fillRect(xL, y, Math.max(1.5, xR - xL), (pr.y + pr.h) - y);

      ctx1d.strokeStyle = '#ffd600';
      ctx1d.lineWidth = 1.5;
      ctx1d.strokeRect(xL, y, Math.max(1.5, xR - xL), (pr.y + pr.h) - y);

      // Vertical tracker hairline
      let xMid = (xL + xR) / 2;
      ctx1d.strokeStyle = 'rgba(255, 214, 0, 0.75)';
      ctx1d.lineWidth = 1;
      ctx1d.setLineDash([3, 3]);
      ctx1d.beginPath();
      ctx1d.moveTo(xMid, pr.y);
      ctx1d.lineTo(xMid, pr.y + pr.h);
      ctx1d.stroke();
      ctx1d.setLineDash([]);
    }
    ctx1d.restore();

    // 4. On-Canvas HUD Badge (Real-time readout)
    const isCal = metadata.cal && (metadata.cal[0] !== 0 || metadata.cal[1] !== 1.0);
    let hudText;
    if (cursor1DChannel !== null && cursor1DChannel >= 0 && cursor1DChannel < spec.length) {
      const eVal = chToEnergy(cursor1DChannel).toFixed(1);
      const cVal = spec[cursor1DChannel] || 0;
      hudText = isCal
        ? `Ch: ${cursor1DChannel} | Energy: ${eVal} keV | Counts: ${cVal}`
        : `Ch: ${cursor1DChannel} | Counts: ${cVal}`;
    } else {
      hudText = `Hover over histogram to inspect bins`;
    }

    ctx1d.save();
    ctx1d.font = 'bold 10px monospace';
    const textW = ctx1d.measureText(hudText).width;
    const badgeX = pr.x + pr.w - textW - 16;
    const badgeY = pr.y + 6;
    ctx1d.fillStyle = 'rgba(20, 20, 20, 0.88)';
    ctx1d.strokeStyle = (cursor1DChannel !== null) ? '#ffd600' : '#555';
    ctx1d.lineWidth = 1;
    ctx1d.fillRect(badgeX, badgeY, textW + 12, 19);
    ctx1d.strokeRect(badgeX, badgeY, textW + 12, 19);
    ctx1d.fillStyle = (cursor1DChannel !== null) ? '#ffd600' : '#888';
    ctx1d.textAlign = 'left';
    ctx1d.fillText(hudText, badgeX + 6, badgeY + 13);
    ctx1d.restore();

    // 5. Axes Ticks on X (Channels & calibrated keV!)
    ctx1d.fillStyle = '#999';
    ctx1d.font = '10px monospace';
    ctx1d.textAlign = 'center';
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
      const frac = i / numTicks;
      const c = Math.round(chStart + frac * (span - 1));
      const x = pr.x + frac * pr.w;
      ctx1d.beginPath();
      ctx1d.moveTo(x, pr.y + pr.h);
      ctx1d.lineTo(x, pr.y + pr.h + 4);
      ctx1d.stroke();
      if (isCal) {
        const eStr = chToEnergy(c).toFixed(0);
        ctx1d.fillText(`${eStr}k (${c})`, x, pr.y + pr.h + 15);
      } else {
        ctx1d.fillText(c.toString(), x, pr.y + pr.h + 15);
      }
    }
    const xLabel = (axis === 0)
      ? (isCal ? 'Det 1 Energy (keV) / Channel' : 'Det 1 Channel')
      : (isCal ? 'Det 2 Energy (keV) / Channel' : 'Det 2 Channel');
    ctx1d.fillText(xLabel, pr.x + pr.w / 2, pr.y + pr.h + 28);

    // Ticks on Y
    ctx1d.textAlign = 'right';
    ctx1d.fillText(isLog ? 'log(N)' : maxVal.toString(), pr.x - 8, pr.y + 10);
    ctx1d.fillText('0', pr.x - 8, pr.y + pr.h);
  }

  function updateMarkerStatus() {
    const l = markers.left !== null ? markers.left : '-';
    const r = markers.right !== null ? markers.right : '-';
    const d = markers.down !== null ? markers.down : '-';
    const u = markers.up !== null ? markers.up : '-';
    document.getElementById('overlayStatus').innerText = `Limits: [L: ${l} | R: ${r} | D: ${d} | U: ${u}] -> Press 'E' to Expand, 'F' Full`;
  }

  function expandMarkers() {
    let xmin = markers.left !== null ? markers.left : view.x0;
    let xmax = markers.right !== null ? markers.right : view.x1;
    let ymin = markers.down !== null ? markers.down : view.y0;
    let ymax = markers.up !== null ? markers.up : view.y1;

    if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
    if (ymin > ymax) [ymin, ymax] = [ymax, ymin];

    if (xmax - xmin >= 1 || ymax - ymin >= 1) {
      view.x0 = Math.max(0, xmin);
      view.x1 = Math.min(metadata.shape[0], xmax);
      view.y0 = Math.max(0, ymin);
      view.y1 = Math.min(metadata.shape[1], ymax);

      markers.left = null; markers.right = null;
      markers.down = null; markers.up = null;
      updateMarkerStatus();
      fetchTileAndRender();
      fetch1DProjection();
    }
  }

  function zoomFull() {
    view.x0 = 0; view.x1 = metadata.shape[0];
    view.y0 = 0; view.y1 = metadata.shape[1];
    markers.left = null; markers.right = null;
    markers.down = null; markers.up = null;
    updateMarkerStatus();
    fetchTileAndRender();
    fetch1DProjection();
  }

  function shiftView(dir) {
    const spanX = view.x1 - view.x0;
    const spanY = view.y1 - view.y0;
    const shiftX = Math.round(spanX * 0.4);
    const shiftY = Math.round(spanY * 0.4);

    if (dir === 'left') {
      view.x0 = Math.max(0, view.x0 - shiftX);
      view.x1 = view.x0 + spanX;
    } else if (dir === 'right') {
      view.x1 = Math.min(metadata.shape[0], view.x1 + shiftX);
      view.x0 = view.x1 - spanX;
    } else if (dir === 'down') {
      view.y0 = Math.max(0, view.y0 - shiftY);
      view.y1 = view.y0 + spanY;
    } else if (dir === 'up') {
      view.y1 = Math.min(metadata.shape[1], view.y1 + shiftY);
      view.y0 = view.y1 - spanY;
    }
    fetchTileAndRender();
    fetch1DProjection();
  }

  function centerOnCursor() {
    const spanX = (view.x1 - view.x0) / 2;
    const spanY = (view.y1 - view.y0) / 2;
    view.x0 = Math.max(0, cursorChannel.x - spanX);
    view.x1 = Math.min(metadata.shape[0], view.x0 + 2 * spanX);
    view.y0 = Math.max(0, cursorChannel.y - spanY);
    view.y1 = Math.min(metadata.shape[1], view.y0 + 2 * spanY);
    fetchTileAndRender();
    fetch1DProjection();
  }

  async function updateHoverHUD() {
    const res = await fetch(`/api/value?x=${cursorChannel.x}&y=${cursorChannel.y}`);
    const data = await res.json();
    document.getElementById('hudCoords').innerText = `Det 1 (X): ch ${data.x} | Det 2 (Y): ch ${data.y} | Counts: ${data.value}`;
  }

  let hoverTimer = null;

  function setupEvents() {
    document.getElementById('cmapSelect').addEventListener('change', () => {
      updateColorLUT();
      render2D();
    });
    document.getElementById('scaleSelect').addEventListener('change', render2D);
    document.getElementById('vmaxSlider').addEventListener('input', (e) => {
      document.getElementById('vmaxLabel').innerText = e.target.value;
      render2D();
    });
    document.getElementById('vminSlider').addEventListener('input', (e) => {
      document.getElementById('vminLabel').innerText = e.target.value;
      render2D();
    });

    document.getElementById('projAxisSelect').addEventListener('change', fetch1DProjection);
    document.getElementById('projRangeSelect').addEventListener('change', render1DSpectrum);
    document.getElementById('projScaleSelect').addEventListener('change', render1DSpectrum);

    document.getElementById('btnFull').addEventListener('click', zoomFull);
    document.getElementById('btnExpand').addEventListener('click', expandMarkers);
    document.getElementById('btnResetMarkers').addEventListener('click', () => {
      markers.left = null; markers.right = null;
      markers.down = null; markers.up = null;
      updateMarkerStatus();
      render2D();
    });

    document.getElementById('btnExport1D').addEventListener('click', () => {
      if (!currentProjSpec) return;
      const axis = parseInt(document.getElementById('projAxisSelect').value);
      const isSynced = (document.getElementById('projRangeSelect').value === 'synced');
      let chStart = 0, chEnd = currentProjSpec.length - 1;
      if (isSynced) {
        if (axis === 0) {
          chStart = Math.max(0, Math.floor(view.x0));
          chEnd = Math.min(currentProjSpec.length - 1, Math.ceil(view.x1));
        } else {
          chStart = Math.max(0, Math.floor(view.y0));
          chEnd = Math.min(currentProjSpec.length - 1, Math.ceil(view.y1));
        }
      }

      let content = `# 1D Binned Histogram Export from ${metadata.filename}\\n`;
      content += `# Projection Axis: ${axis === 0 ? 'Det 1 (X)' : 'Det 2 (Y)'}\\n`;
      content += `# Sliced 2D Window: X=[${currentProjSlice.x0}..${currentProjSlice.x1}], Y=[${currentProjSlice.y0}..${currentProjSlice.y1}]\\n`;
      content += `# Channel Energy_keV Counts\\n`;

      for (let i = chStart; i <= chEnd; i++) {
        let e = chToEnergy(i).toFixed(3);
        content += `${i} ${e} ${currentProjSpec[i]}\\n`;
      }

      const blob = new Blob([content], { type: 'text/plain' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${metadata.filename}_hist_axis${axis}_${chStart}_${chEnd}.dat`;
      a.click();
    });

    document.getElementById('btnHelp').addEventListener('click', () => {
      document.getElementById('helpModal').style.display = 'block';
    });

    // Keyboard Shortcuts (GASPware cmat controls)
    window.addEventListener('keydown', (e) => {
      const isCtrl = e.ctrlKey || e.metaKey;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (isCtrl) shiftView('left');
        else { markers.left = cursorChannel.x; updateMarkerStatus(); render2D(); }
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (isCtrl) shiftView('right');
        else { markers.right = cursorChannel.x; updateMarkerStatus(); render2D(); }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (isCtrl) shiftView('down');
        else { markers.down = cursorChannel.y; updateMarkerStatus(); render2D(); }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (isCtrl) shiftView('up');
        else { markers.up = cursorChannel.y; updateMarkerStatus(); render2D(); }
      } else if (e.key === 'e' || e.key === 'E') {
        expandMarkers();
      } else if (e.key === 'f' || e.key === 'F') {
        zoomFull();
      } else if (e.key === 'p' || e.key === 'P') {
        const sel = document.getElementById('projAxisSelect');
        sel.value = (sel.value === '0') ? '1' : '0';
        fetch1DProjection();
      } else if (e.key === 'x' || e.key === 'X') {
        document.getElementById('projAxisSelect').value = '0';
        fetch1DProjection();
      } else if (e.key === 'y' || e.key === 'Y') {
        document.getElementById('projAxisSelect').value = '1';
        fetch1DProjection();
      } else if (e.key === 'c' || e.key === 'C') {
        const sel = document.getElementById('cmapSelect');
        sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
        updateColorLUT();
        render2D();
      } else if (e.key === '1') {
        document.getElementById('scaleSelect').value = 'linear';
        render2D();
      } else if (e.key === '2') {
        document.getElementById('scaleSelect').value = 'sqrt';
        render2D();
      } else if (e.key === '4' || e.key === 'l' || e.key === 'L') {
        document.getElementById('scaleSelect').value = 'log';
        render2D();
      } else if (e.key === 'm' || e.key === 'M') {
        centerOnCursor();
      } else if (e.key === '?' || e.key === 'h' || e.key === 'H') {
        const hm = document.getElementById('helpModal');
        hm.style.display = hm.style.display === 'block' ? 'none' : 'block';
      } else if (e.key === 'Escape') {
        document.getElementById('helpModal').style.display = 'none';
      }
    });

    // Mouse Controls on 2D Matrix
    canvas2d.addEventListener('mousedown', (e) => {
      const rect = canvas2d.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      dragStartPos = { x: mouseX, y: mouseY };
      mouseCurrentPos = { x: mouseX, y: mouseY };

      if (e.ctrlKey || e.metaKey) {
        isBoxZooming = true;
      } else {
        isPanning = true;
      }
    });

    window.addEventListener('mousemove', (e) => {
      const rect = canvas2d.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      mouseCurrentPos = { x: mouseX, y: mouseY };

      const ch = pxToCh2D(mouseX, mouseY);
      cursorChannel = {
        x: Math.max(0, Math.min(metadata.shape[0] - 1, Math.floor(ch.x))),
        y: Math.max(0, Math.min(metadata.shape[1] - 1, Math.floor(ch.y)))
      };

      if (isPanning) {
        const pr = getPlotRect2D();
        const dx = (mouseX - dragStartPos.x) * ((view.x1 - view.x0) / pr.w);
        const dy = (mouseY - dragStartPos.y) * ((view.y1 - view.y0) / pr.h);
        view.x0 -= dx; view.x1 -= dx;
        view.y0 += dy; view.y1 += dy;
        dragStartPos = { x: mouseX, y: mouseY };
        render2D();
      } else if (isBoxZooming) {
        render2D();
      }

      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(updateHoverHUD, 20);
    });

    window.addEventListener('mouseup', async (e) => {
      if (isBoxZooming) {
        isBoxZooming = false;
        const xStart = Math.min(dragStartPos.x, mouseCurrentPos.x);
        const xEnd = Math.max(dragStartPos.x, mouseCurrentPos.x);
        const yStart = Math.min(dragStartPos.y, mouseCurrentPos.y);
        const yEnd = Math.max(dragStartPos.y, mouseCurrentPos.y);

        if (xEnd - xStart > 5 && yEnd - yStart > 5) {
          const ch0 = pxToCh2D(xStart, yEnd);
          const ch1 = pxToCh2D(xEnd, yStart);

          view.x0 = Math.max(0, Math.floor(ch0.x));
          view.x1 = Math.min(metadata.shape[0], Math.ceil(ch1.x));
          view.y0 = Math.max(0, Math.floor(ch0.y));
          view.y1 = Math.min(metadata.shape[1], Math.ceil(ch1.y));
          fetchTileAndRender();
          fetch1DProjection();
        } else {
          render2D();
        }
      } else if (isPanning) {
        isPanning = false;
        fetchTileAndRender();
        fetch1DProjection();
      }
    });

    canvas2d.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.25 : 0.8;
      const rect = canvas2d.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const ch = pxToCh2D(mouseX, mouseY);
      const spanX = (view.x1 - view.x0) * zoomFactor;
      const spanY = (view.y1 - view.y0) * zoomFactor;

      view.x0 = Math.max(0, ch.x - (spanX * 0.5));
      view.x1 = Math.min(metadata.shape[0], view.x0 + spanX);
      view.y0 = Math.max(0, ch.y - (spanY * 0.5));
      view.y1 = Math.min(metadata.shape[1], view.y0 + spanY);

      clearTimeout(tileFetchDebounce);
      clearTimeout(projFetchDebounce);
      render2D();
      tileFetchDebounce = setTimeout(fetchTileAndRender, 40);
      projFetchDebounce = setTimeout(fetch1DProjection, 60);
    });

    // 1D Spectrum Real-time Hover Inspector & Position Readout
    canvas1d.addEventListener('mousemove', (e) => {
      if (!currentProjSpec) return;
      const rect = canvas1d.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const pr = getPlotRect1D();
      const axis = parseInt(document.getElementById('projAxisSelect').value);
      const isSynced = (document.getElementById('projRangeSelect').value === 'synced');

      let chStart = 0, chEnd = currentProjSpec.length - 1;
      if (isSynced) {
        if (axis === 0) {
          chStart = Math.max(0, Math.floor(view.x0));
          chEnd = Math.min(currentProjSpec.length - 1, Math.ceil(view.x1));
        } else {
          chStart = Math.max(0, Math.floor(view.y0));
          chEnd = Math.min(currentProjSpec.length - 1, Math.ceil(view.y1));
        }
      }
      const span = Math.max(1, chEnd - chStart + 1);
      const frac = (mouseX - pr.x) / pr.w;
      if (frac >= 0 && frac <= 1) {
        const ch = Math.floor(chStart + frac * span);
        cursor1DChannel = Math.max(0, Math.min(currentProjSpec.length - 1, ch));
        const val = currentProjSpec[cursor1DChannel] || 0;
        const energy = chToEnergy(cursor1DChannel).toFixed(1);
        const detName = (axis === 0) ? 'Det 1 (1D)' : 'Det 2 (1D)';
        document.getElementById('hudCoords').innerText = `${detName}: ch ${cursor1DChannel} (${energy} keV) | Proj Counts: ${val}`;
      } else {
        cursor1DChannel = null;
      }
      render1DSpectrum();
    });

    canvas1d.addEventListener('mouseleave', () => {
      cursor1DChannel = null;
      render1DSpectrum();
    });
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
