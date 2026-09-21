# 3D Matrix Analysis & Web Viewer

The `cmat3d_webviewer.py` server and companion `cmat3d_webviewer.html` client provide an ultra-responsive, browser-based spectroscopy environment for navigating 3D matrix cubes in the `.cmat` format (e.g. $\gamma$-$\gamma$-$\text{Rings}$, $\gamma$-$\gamma$-$\Delta t$, or 3-fold coincidence volumes).

The viewer couples pure Python/NumPy memory-mapped slice decompression with a high-performance HTML5 Canvas frontend featuring real-time multi-plane orthoslicing, 3D multi-gate coincidence cuts, 2D graphical Banana ROI gates, 1D/2D photopeak fitting, and publication-quality vector PDF export.

---

## Launching the 3D Web Viewer

Run the 3D viewer from your terminal, pointing to a 3D `.cmat` file:

```bash
# Open a 3D matrix cube
python cmat3d_webviewer.py /path/to/matrix3d.cmat

# Open multiple 3D matrices for rapid cycling
python cmat3d_webviewer.py cube1.cmat cube2.cmat
```

The server binds to `http://0.0.0.0:8080` (or the configured host/port) and automatically opens your default web browser.

---

## Architectural Overview

```
+-----------------------------------------------------------------------------------+
|                              cmat3d_webviewer.py                                  |
|                                                                                   |
|  +---------------------------+   +---------------------------------------------+  |
|  | CMAT3DReader (cmat3d.py)  |   | ThreadingHTTPServer                         |  |
|  | - IVF Slice Decompressor  |   | - Multi-threaded concurrent request handler |  |
|  | - .cmat3d_cache/ (memmap) |   | - /api/tile (Subregion Slicing: ~1.6 ms)    |  |
|  | - 3D Coincidence Slicing  |   | - /api/projection_region (~2.1 ms)          |  |
|  | - 2D Banana Polygon ROI   |   | - /api/gate_1d (Coincidence Cuts)           |  |
|  +---------------------------+   +---------------------------------------------+  |
+------------------------------------------^----------------------------------------+
                                           | HTTP / REST (JSON + Binary ArrayBuffers)
+------------------------------------------v----------------------------------------+
|                             cmat3d_webviewer.html                                 |
|                                                                                   |
|  +---------------------------+   +---------------------------------------------+  |
|  | 2D Density Heatmap Canvas |   | Triple 1D Spectroscopic Displays            |  |
|  | - Turbo/Viridis LUT Engine|   | - Det 1 (Axis 0) Stepped Histogram          |  |
|  | - Multi-Plane: 0-1, 0-2, 1-2 | - Det 2 (Axis 1) Stepped Histogram          |  |
|  | - 2D Banana ROI (Shift+G) |   | - Rings / dt (Axis 2) Stepped Histogram     |  |
|  | - AbortController Sync   |   | - 60 FPS Client-Side Instant Channel Zoom   |  |
|  +---------------------------+   +---------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## Key Features & Capabilities

### 1. Memory-Mapped 3D Volume Engine (`.cmat3d_cache/`)
3D `.cmat` files frequently contain large data volumes (e.g. $4096 \times 4096 \times 128 = 2.14 \times 10^9$ channels, $\approx 8.5\text{ GB}$ uncompressed).
- **Zero RAM Bloat**: `CMAT3DReader` decompresses slices on first access into a binary cache file (`.cmat3d_cache/<filename>.dat`) and accesses the data as a 3D `numpy.memmap` (`int32`).
- **Instant Subsequent Startup**: Subsequent launches read the cached memory-mapped file instantly ($<50\text{ ms}$).

### 2. Multi-Plane Orthogonal Projections
The viewer allows real-time orthogonal slicing across any pair of dimensions:
- **Plane 0-1 (`Det 1 vs Det 2`)**: Standard $4096 \times 4096$ $\gamma$-$\gamma$ coincidence matrix, sliced over the visible/gated 3rd axis.
- **Plane 0-2 (`Det 1 vs Rings`)**: $4096 \times 128$ matrix showing $\gamma$-ray energy on Det 1 as a function of detector ring or timing.
- **Plane 1-2 (`Det 2 vs Rings`)**: $4096 \times 128$ matrix showing $\gamma$-ray energy on Det 2 as a function of detector ring or timing.

Switch planes instantly using the header toolbar buttons (`0-1`, `0-2`, `1-2`) or dropdown menu.

### 3. Real-Time Subregion Slicing & Accelerated Navigation
- **Pre-Sum Slicing**: When zooming into a 2D region $[x_0..x_1, y_0..y_1]$, the backend extracts only the required subvolume from the 3D memory map *before* summing over the 3rd axis, accelerating tile generation from $76.5\text{ ms}$ to **$1.6\text{ ms}$ (~48× speedup)**.
- **Multi-Threaded Server**: `ThreadingHTTPServer` processes simultaneous tile requests, 1D projections, and peak fits on separate background threads without blocking.
- **Debounced 2D Synchronization**: 1D wheel zooming and channel panning update the stepped histograms client-side at 60 FPS ($<1\text{ ms}$), while heavy 2D tile rendering is debounced (45 ms) with `AbortController` cancellation to prevent network congestion.

---

## 3D Coincidence Gating & Banana ROIs

### Multi-Gate Coincidence Slicing ($W$ & $X$)
Set coincidence gates directly on any 1D spectrum:
1. Hover cursor over a 1D spectrum panel (Det 1, Det 2, or Rings).
2. Press **`W`** at the lower gate limit, move cursor to the upper limit, and press **`W`** again to confirm.
3. (Optional) Press **`X`** or **`B`** twice to define background subtraction windows.
4. The remaining spectra and 2D projections immediately update to display the background-subtracted coincidence cut:

$$\text{Net Spec}(i) = \text{Raw Gate}(i) - \frac{\sum \text{Peak Gate Widths}}{\sum \text{BG Gate Widths}} \times \text{BG Spec}(i)$$

- Press **`Z`** or click **Clear Gate** to reset active coincidence gates.

### 2D Banana Gate Polygon Engine (`Shift + G` / `Shift + B`)
For particle-gamma identification, ring discrimination, or kinematic curve gating:
1. **Draw Peak Banana (W)**: Press **`Shift + G`** (or click **Draw Peak [Shift+G]**). Click points on the active 2D plane to define the Peak ROI polygon (rendered in gold `#ffd600`). Close the polygon by clicking near the starting vertex or pressing **`Enter`**.
2. **Draw Background Banana (B)**: Press **`Shift + B`** (or click **Draw Bg [Shift+B]**). Click points on the 2D plane to define the Background ROI polygon (rendered in magenta `#ff4081`). Close the polygon by clicking near the starting vertex or pressing **`Enter`**.
3. **Area-Normalized Subtraction**:
   The viewer computes the continuous geometric surface area (in $\text{ch}^2$ via the Shoelace algorithm) and discrete rasterized pixel count ($\text{px}$) for both polygons. It normalizes background subtraction by the ratio of their surface areas:

   $$\text{scale} = \frac{\text{Area}_{\text{peak}}}{\text{Area}_{\text{bg}}}$$
   $$\text{Net Spec}(i) = \text{Spec}_{\text{peak}}(i) - \text{scale} \times \text{Spec}_{\text{bg}}(i)$$

4. **Terminal Reporting**:
   Applying a banana gate prints a detailed diagnostic report in the terminal listing surface areas (in pixels and $\text{ch}^2$), count totals inside each banana, the scale factor, and net counts:
   ```text
   [2D Banana Gate Cut] matrix.cmat -> Plane 0-1 => Axis 3 (Z):
     • Peak Banana (W): 400 px (area: 400.0 ch², 4 vertices) | Counts: 228,000 cts
     • Bg Banana (B):   200 px (area: 200.0 ch², 4 vertices) | Counts: 64,000 cts (Scale factor: 2.0000)
     • Net Area Counts: 100,000.0 counts
   ```
5. **Clear Gate**: Click **Clear Bananas** or press **`Z`** to remove active banana gates.

---

## 1D & 2D Peak Fitting

The 3D viewer incorporates the complete spectroscopy peak fitting engine:
- **Gaussian**: Standard symmetric photopeak with linear/quadratic continuum baseline.
- **RadWare / SAMPO**: Photopeak with an exponential low-energy tail ($A, \eta$) for high-energy HPGe detectors.
- **Hypermet EMG**: Convolved Gaussian exponential tail plus complementary error function ($\text{erfc}$) Compton step.
- **Multiplet Fits**: Place multiple candidate markers with **`P`** or click fit region with **`R`**, then press **`V`** to perform multi-component non-linear least-squares fitting.
- **2D Peak Fit**: `Ctrl+Click` on the 2D plane to perform 2D Gaussian peak fitting with Gamba coincidence background decomposition.

---

## Configuration (`python-cmat3d-config.txt`)

The 3D viewer automatically reads and writes configuration files in the working directory:

```ini
[General]
port = 8080
colormap = turbo
scale_mode_2d = log
scroll_sensitivity = 0.04
active_plane = 0-1
proj_range_mode = synced

[Axes]
axis0_name = Det 1
axis1_name = Det 2
axis2_name = Rings

[Calibration]
axis0_cal = 0.0, 0.5, 0.0
axis1_cal = 0.0, 0.5, 0.0
axis2_cal = 0.0, 1.0, 0.0

[PeakFit]
fit_type = gaussian
fwhm_est = 4.0
fwhm_mult = 4.0
```

---

## Python Library API (`cmat3d.py`)

You can also use `cmat3d.py` programmatically in Python scripts and Jupyter notebooks:

```python
from cmat3d import CMAT3DReader

# 1. Open 3D .cmat file
reader = CMAT3DReader("GeE-Rings3D.cmat")
print(f"Matrix shape: {reader.shape}")  # e.g. (4096, 4096, 128)
print(f"Total counts: {reader.total_counts:,}")

# 2. Access memory-mapped 3D volume
mmap_vol = reader.to_memmap()  # Shape: (4096, 4096, 128), dtype: int32

# 3. Extract 2D orthogonal plane projection
# Plane options: '0-1' (Det 1 vs 2), '0-2' (Det 1 vs Rings), '1-2' (Det 2 vs Rings)
plane_01 = reader.get_2d_plane("0-1")
plane_gated = reader.get_2d_plane("0-1", gate_3rd=[[10, 12]]) # Gated on Ring 10-12

# 4. Extract 1D total projection
proj_det1 = reader.get_projection_1d(axis=0)

# 5. Coincidence Gating
net_spec, bg_spec = reader.get_gate_1d(
    target_axis=0,
    w_gates={2: [[11, 11]]},        # Gate on Ring 11
    b_gates={2: [[1, 2], [20, 21]]} # Background rings
)

# 6. 2D Banana Gate Polygon (Peak & Background Subtraction)
poly_peak = [[1000, 10], [1500, 12], [1480, 15], [980, 13]] # [[x, y], ...]
poly_bg   = [[900, 10], [950, 12], [940, 15], [890, 13]]
banana_res = reader.get_banana_gate(
    plane="0-2",
    polygon_peak=poly_peak,
    polygon_bg=poly_bg
)
print(f"Peak: {banana_res['pixel_count_peak']} px, Bg: {banana_res['pixel_count_bg']} px (scale: {banana_res['scale']:.4f})")
print(f"Net Gated Counts: {banana_res['total_gated_counts']:,}")
```
