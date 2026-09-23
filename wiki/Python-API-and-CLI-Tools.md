# Python API & CLI Tools

`python-cmat` provides both a pure Python library API (`cmat.py`) for integration into custom scripts and Jupyter notebooks, and a standalone command-line converter (`cmat2amat.py`) for converting `.cmat` files into standard ASCII formats.

---

## Python Library API (`cmat.py`)

The core library revolves around the `CMATReader` class, which handles binary parsing, IVF segment navigation, and on-demand block decompression.

### Basic API Usage Example

```python
from cmat import CMATReader

# 1. Open .cmat file (supports symmetric, asymmetric, and arbitrary step sizes)
reader = CMATReader("GeE-symm.cmat")

# 2. Inspect file header and IVF metadata
info = reader.get_info()
print("Filename:     ", info["filename"])
print("Shape:        ", info["shape"])         # (res1, res2)
print("Matrix Mode:  ", info["matrix_mode"])   # 0 = symmetric, 1 = normal, etc.
print("Step Size:    ", info["step"])          # (step1, step2)
print("Total Blocks: ", info["blocks"])

# 3. Decompress the full matrix into a NumPy 2D array (int32)
# Resulting array shape is (res2, res1) = (Height/Y, Width/X)
matrix = reader.to_numpy()
print(f"Decompressed shape: {matrix.shape}")
print(f"Total matrix counts: {matrix.sum():,}")

# 4. Extract total 1D projections
# axis=0: Det 1 (X) projection; axis=1: Det 2 (Y) projection
proj_det1 = reader.get_projection(axis=0)
proj_det2 = reader.get_projection(axis=1)

# 5. Extract coincidence-gated 1D projection
# Gate channels 500 to 520 on Det 1 -> returns coincidence spectrum on Det 2
net_spec, bg_spec, raw_spec = reader.get_gate(500, 520, axis=0)

# 6. Extract 2D Banana Graphical Polygon ROI & Net Counts
# Computes pixel mask, geometric area, raw counts, and normalized subtraction
banana = reader.get_banana_roi(
    polygon_peak=[[100, 100], [150, 120], [140, 160], [90, 140]],
    polygon_bg=[[80, 80], [170, 100], [160, 180], [70, 160]]
)
print(f"Net Banana Area: {banana['net_counts']:,.1f} +/- {banana['net_err']:,.1f}")

# 7. Export directly to ASCII (.amat) format
reader.export_amat("GeE-symm.amat", format_type="dense")
```

---

### `CMATReader` Method Reference

#### `CMATReader(filepath: str | Path)`
Initializes the reader, parses IVF header records, builds block offset tables, and extracts stored projections if available.

#### `get_info() -> dict`
Returns a metadata dictionary containing:
- `filename`: Absolute path to the matrix file.
- `shape`: Tuple `(res1, res2)` representing matrix dimensions.
- `step`: Tuple `(step1, step2)` representing sub-matrix block size.
- `matrix_mode`: Integer mode flag (e.g. 0 for folded symmetric, 1 for normal full).
- `blocks`: Total count of IVF blocks stored in the container.

#### `to_numpy() -> np.ndarray`
Decompresses all sub-blocks into a 2D NumPy array of `int32` elements.
- Returns array with shape `(res2, res1)`.
- For symmetric matrices, automatically mirrors the triangular blocks across the diagonal ($M_{ij} = M_{ji}$).

#### `get_projection(axis: int = 0) -> np.ndarray`
Returns a 1D NumPy array representing the total projection:
- `axis=0`: Det 1 / X projection.
- `axis=1`: Det 2 / Y projection.
- Uses IVF pre-computed projection segments when available, falling back to full-matrix summation if missing.

#### `get_gate(gate_min: int, gate_max: int, axis: int = 0, bg_min: int = None, bg_max: int = None) -> tuple`
Extracts a coincidence-gated 1D slice along the specified axis:
- Slices between `gate_min` and `gate_max`.
- If `bg_min` and `bg_max` are provided, performs normalized background subtraction scaled by channel window width.
- Returns `(net_spectrum, bg_spectrum, raw_spectrum)`.

#### `get_banana_roi(polygon_peak: list, polygon_bg: list = None) -> dict`
Calculates discrete pixel containment, continuous geometric Shoelace area, raw counts, area-normalized scale factor, and net background-subtracted counts for arbitrary 2D polygon ROIs:
- `polygon_peak`: List of `[x, y]` coordinate pairs defining the Peak polygon ROI.
- `polygon_bg`: Optional list of `[x, y]` coordinate pairs defining the Background polygon ROI.
- Returns dictionary containing:
  - `peak`: Sub-dictionary with `area_geom`, `pixels`, `counts`, and `vertices`.
  - `bg`: Sub-dictionary with background metrics (or `None`).
  - `scale`: Area normalization factor ($\text{Area}_{\text{peak}} / \text{Area}_{\text{bg}}$).
  - `net_counts`: Area-normalized net peak counts ($C_{\text{peak}} - \text{scale} \times C_{\text{bg}}$).
  - `net_err`: Statistical uncertainty $\sqrt{C_{\text{peak}} + \text{scale}^2 \times C_{\text{bg}}}$.

#### `export_amat(out_path: str | Path, format_type: str = "dense", roi: tuple = None)`
Exports matrix counts to an ASCII `.amat` file:
- `format_type="dense"`: Rectangular grid of count values.
- `format_type="sparse"`: Triplet list of `x y counts` for non-zero channels.
- `roi`: Optional `(xmin, xmax, ymin, ymax)` tuple to restrict exported area.

---

## Unified Matrix Launcher (`pycmat`)

`pycmat` is a standalone executable Python dispatcher that automatically inspects input `.cmat` file headers in $<0.1\text{ ms}$ to determine dimensionality (`ndim`), seamlessly forwarding all CLI flags and options to the appropriate viewer:
- **2D Matrices (`ndim == 2`)**: Launches `cmat_webviewer.py`
- **3D Matrices (`ndim == 3`)**: Launches `cmat3d_webviewer.py`

### CLI Usage Examples

```bash
# Launch interactive viewer for any 2D or 3D matrix (auto-detected)
./pycmat /path/to/matrix.cmat

# Forward options such as custom port, calibration, or browser settings
./pycmat run1.cmat --port 8085 --no-browser

# Execute batch headless spectroscopy commands
./pycmat GeE-symm.cmat -c "info; search 0"
```

---

## CLI Matrix Converters (`cmat2amat.py` & `amat2cmat.py`)

`python-cmat` provides bidirectional conversion between proprietary binary `.cmat` files and standard plain-text ASCII (`.amat`, `.dat`, `.txt`, `.csv`) or NumPy (`.npy`) files.

### 1. `cmat2amat.py` (.cmat $\rightarrow$ ASCII / NumPy)

Converts `.cmat` files into dense 2D ASCII grids, sparse coordinate lists, or NumPy `.npy` arrays.

```bash
# Convert entire matrix to a dense 2D ASCII grid:
python3 cmat2amat.py GeE-symm.cmat -o GeE-symm.amat

# Convert a specific Region of Interest (ROI):
python3 cmat2amat.py GeE-symm.cmat -o GeE_roi.amat --range-x 0 200 --range-y 0 200

# Export as a sparse list of non-zero channels (x y counts):
python3 cmat2amat.py GeE-symm.cmat -o GeE_sparse.amat --format sparse

# Save decompressed array to NumPy binary format (.npy):
python3 cmat2amat.py GeE-symm.cmat --npy
```

### 2. `amat2cmat.py` (ASCII / NumPy $\rightarrow$ .cmat)

Inverse converter that takes any 2D ASCII grid, sparse triplet list (`x y counts`), or NumPy array, and compresses it into a GASPware-compliant `.cmat` binary file.

```bash
# Convert dense ASCII matrix to .cmat:
python3 amat2cmat.py GeE-symm.amat -o GeE-symm_reconstructed.cmat

# Convert sparse ASCII matrix with explicit shape:
python3 amat2cmat.py GeE_sparse.amat -o GeE_sparse.cmat --shape 4096 4096

# Convert NumPy binary array (.npy) to symmetric .cmat:
python3 amat2cmat.py matrix.npy -o matrix.cmat --symmetric

# Inspect ASCII matrix statistics without exporting:
python3 amat2cmat.py GeE-symm.amat --info
```

### Programmatic Python Matrix Writer (`cmat.py`)

You can also compress and save matrices directly from Python scripts:

```python
from cmat import write_cmat, CMATWriter
import numpy as np

# Create or load a 2D matrix
matrix = np.random.poisson(lam=5, size=(4096, 4096)).astype(np.int32)
# Symmetrize if desired
matrix = (matrix + matrix.T) // 2

# Method 1: Functional write_cmat
write_cmat("simulated.cmat", matrix, symmetric=True, step1=128, step2=128)

# Method 2: OOP CMATWriter
writer = CMATWriter(matrix, symmetric=True)
writer.save("simulated.cmat")
```

---

## 3D Matrix Python API (`cmat3d.py`)

`python-cmat` includes `CMAT3DReader` for handling 3D matrix cubes with memory-mapped array caching and multi-plane slicing.

### Basic 3D API Example

```python
from cmat3d import CMAT3DReader

# 1. Initialize reader and build/load memory-mapped cache
reader3d = CMAT3DReader("GeE-Rings3D.cmat")
print(f"3D Shape: {reader3d.shape}")       # (4096, 4096, 128)
print(f"Total counts: {reader3d.total_counts:,}")

# 2. Access 3D volume directly as NumPy array / memmap (int32)
vol = reader3d.to_memmap()

# 3. Extract 2D orthogonal plane projection
# plane: '0-1' (Det 1 vs Det 2), '0-2' (Det 1 vs Rings), '1-2' (Det 2 vs Rings)
plane_01 = reader3d.get_2d_plane("0-1")
plane_01_sub = reader3d.get_2d_plane("0-1", x0=1000, x1=2000, y0=1000, y1=2000)

# 4. Extract total 1D projection along any axis (0, 1, or 2)
spec_det1 = reader3d.get_projection_1d(axis=0)

# 5. Extract 1D coincidence cuts with background subtraction
net_spec, bg_spec = reader3d.get_gate_1d(
    target_axis=0,
    w_gates={2: [[11, 11]]},        # Gate on Ring 11 (axis 2)
    b_gates={2: [[1, 2], [20, 21]]} # Background rings
)

# 6. Extract 2D Banana Graphical Polygon Cut
banana_cut = reader3d.get_banana_gate(
    plane="0-2",
    polygon=[[1000, 10], [1500, 12], [1480, 15], [980, 13]]
)
print("Banana cut net spectrum sum:", banana_cut["net_spec"].sum())
```

---

## ENSDF Isotope Identification API & CLI (`ensdf_search.py`)

`python-cmat` includes `ENSDFSearchEngine` for automated offline isotope identification of 1D photopeaks and 2D coincidence cascades using physical cascade topologies and global mass-clustering parsimony.

### Command-Line Interface

```bash
# Identify full fit results log file:
python3 ensdf_search.py fit_results_20260922_155405.txt --top 5

# Manual 2D coincidence search:
python3 ensdf_search.py -c 1434.2 935.3 --tol 1.5 --top 5

# Manual 1D single-energy search:
python3 ensdf_search.py -g 1480.3 --tol 1.5

# Database status & statistics:
python3 ensdf_search.py --status
```

### Python API Usage

```python
from ensdf_search import ENSDFSearchEngine

engine = ENSDFSearchEngine()

# Identify entire fit results file with global parsimonious set cover
report = engine.identify_fit_results_file(
    "fit_results_20260922_155405.txt",
    tol=1.5,
    top_candidates=5
)

print(f"Dominant Mass Center: A ≈ {report['dominant_mass']}")
print(f"Minimal Isotope Set: {report['parsimonious_isotopes']}")

for res in report["results_2d"]:
    fit = res["fit"]
    best = res["best_match"]
    print(f"2D Fit {fit['energy1']} x {fit['energy2']} keV -> {best['nuclide']} (Score: {best['score']}, {best['cascade_type']})")
```

---

## Nuclear Half-Life & Lifetime Fitting Tool (`halflife.py`)

`halflife.py` is an analytical nuclear lifetime fitting engine based on `halflife.c` for analyzing time-difference spectra (TAC, TDC, digital CFD timestamp differences) exported from `cmat_webviewer.py` and `cmat3d_webviewer.py`.

It supports both an interactive terminal REPL (matching the classic menu-driven workflow) and a fully scriptable command-line interface.

### Command-Line Interface (CLI)

```bash
# 1. Interactive terminal menu REPL (classic halflife.c experience):
python3 halflife.py -i spectrum.dat

# 2. Scriptable automated fitting with initial parameters:
python3 halflife.py spectrum.dat --t12 19.5 --fwhm 15.2 --centroid 482.0 --bg 10.0 --range 450 750

# 3. Fit and export ASCII .fit file and publication vector PDF:
python3 halflife.py spectrum.dat --t12 20.0 --fwhm 15.0 --range 450 750 --out fit_result.fit --pdf fit_plot.pdf

# 4. Perform chi-square profile scan over background:
python3 halflife.py spectrum.dat --scan-bg 0.0 50.0 50

# 5. Compress spectrum by factor of 2 and zero-suppress:
python3 halflife.py spectrum.dat --compress 2 --zero-suppress --t12 10.0 --range 200 600
```

#### CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_file` | `str` | *None* | Path to input ASCII spectrum (`.dat`, `.txt`, `.csv`) |
| `-i`, `--interactive` | `flag` | `False` | Launch interactive terminal REPL menu |
| `--t12` | `float` | Auto | Initial half-life estimate in channels ($>0$ for right tail, $<0$ for left tail, $0$ for prompt) |
| `--fix-t12` | `flag` | `False` | Fix half-life during fit (e.g. to fit pure prompt IRF) |
| `--fwhm` | `float` | Auto | Initial prompt time resolution FWHM ($0$ for pure exponential) |
| `--fix-fwhm` | `flag` | `False` | Fix FWHM during fit |
| `--centroid` | `float` | Auto | Initial prompt centroid / peak position |
| `--fix-centroid` | `flag` | `False` | Fix centroid during fit |
| `--bg` | `float` | Auto | Initial constant background baseline |
| `--fix-bg` | `flag` | `False` | Fix background during fit |
| `--range` | `int int` | `[min, max]` | Fit range window bounds in channels |
| `--compress` | `int` | `1` | Rebin / compress spectrum channels by factor $N$ |
| `--zero-suppress` | `flag` | `False` | Suppress non-positive counts during Poisson weighting |
| `--scan-bg` | `float float int` | Auto | Background $\chi^2$ profile scan `[min max steps]` |
| `--out` | `str` | *None* | Save fit parameters, covariance, and model curve to `.fit` |
| `--pdf` | `str` | *None* | Export publication-quality vector PDF plot |

---

### Python API Usage

```python
from halflife import HalfLifeFitter

# 1. Load spectrum from file or NumPy arrays
# Supports 1, 2, 3, or 4 column ASCII exports
fitter = HalfLifeFitter.from_file("LaE-TAC.cmat_Det2_Y_Gated_1437_2458.dat")

# 2. Configure initial parameters & free/fixed flags
fitter.set_parameters(
    t12=20.0, fix_t12=False,
    fwhm=15.0, fix_fwhm=False,
    centroid=482.0, fix_centroid=False,
    bg=10.0, fix_bg=False,
    fit_min=450, fit_max=750
)

# 3. Perform Levenberg-Marquardt / TRF non-linear fit
res = fitter.fit()

print(f"Half-life: {res.t12:.4f} +/- {res.t12_err:.4f} ch")
print(f"FWHM:      {res.fwhm:.4f} +/- {res.fwhm_err:.4f} ch")
print(f"Centroid:  {res.centroid:.4f} +/- {res.centroid_err:.4f} ch")
print(f"Chi2/NDF:  {res.chi2_ndf:.3f} (NDF={res.ndf})")

# 4. Perform background chi-square exploration scan
scan = fitter.scan_background(bg_min=0.0, bg_max=40.0, num_steps=40)
print(f"Best background from scan: {scan['best_bg']:.2f}")

# 5. Export results
fitter.export_fit_file("output.fit")
fitter.export_pdf("output.pdf", title="138La Nuclear Lifetime Fit")
```
