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

# 6. Export directly to ASCII (.amat) format
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

#### `export_amat(out_path: str | Path, format_type: str = "dense", roi: tuple = None)`
Exports matrix counts to an ASCII `.amat` file:
- `format_type="dense"`: Rectangular grid of count values.
- `format_type="sparse"`: Triplet list of `x y counts` for non-zero channels.
- `roi`: Optional `(xmin, xmax, ymin, ymax)` tuple to restrict exported area.

---

## CLI Matrix Converter (`cmat2amat.py`)

`cmat2amat.py` provides a convenient command-line interface for converting proprietary `.cmat` files into standard plain-text ASCII files compatible with ROOT, GNUplot, MATLAB, and spreadsheet software.

### CLI Usage Examples

```bash
# Convert entire matrix to a dense 2D ASCII grid:
python3 cmat2amat.py GeE-symm.cmat GeE-symm.amat

# Convert a specific Region of Interest (ROI):
# Channels 0 to 200 on X, 0 to 200 on Y
python3 cmat2amat.py GeE-symm.cmat GeE_roi.amat --roi 0 200 0 200

# Export as a sparse list of non-zero channels (x y counts):
python3 cmat2amat.py GeE-symm.cmat GeE_sparse.amat --format sparse
```

### CLI Command Options

| Argument | Description |
|---|---|
| `input_file` | Path to the source `.cmat` binary matrix file. |
| `output_file` | Destination path for the generated ASCII `.amat` file. |
| `--format` | Output structure: `dense` (default grid) or `sparse` (`x y counts`). |
| `--roi xmin xmax ymin ymax` | Bounding box coordinates to export a sub-region rather than the whole matrix. |
| `--verbose` | Output detailed decompression diagnostics and block statistics to stdout. |
