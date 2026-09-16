# Spectroscopy Commands & Macros

In addition to its browser GUI, `python-cmat` includes a headless spectroscopy engine. It allows experimentalists to script complex analysis pipelines, run batch coincidence gating, execute multi-peak deconvolution, and export publication-ready vector PDFs directly from terminal environments or cluster jobs without opening a graphical window.

---

## Headless Execution Modes

### 1. Batch Macro Script Execution (`-m` / `--macro`)
Run an automated spectroscopy macro file (`*.mac`):

```bash
python3 cmat_webviewer.py -m analysis_template.mac
```

Macro scripts are plain text files containing sequentially executed commands, supporting comments (lines starting with `#`), variable delays (`sleep`), and status messages (`echo`).

### 2. Direct CLI One-Liner Execution (`-c` / `--command`)
Execute one or more semicolon-delimited commands directly from your terminal or shell scripts:

```bash
python3 cmat_webviewer.py -c "load GeE-symm.cmat run1; cal 0 0.5 1.002; gate 0 w 1170 1176 b 1150 1160; fit_1d 1 1332; pdf_1d 1 gated.pdf --fit; exit"
```

### 3. Interactive Headless Shell (`-i` / `--headless`)
Launch an interactive REPL shell with command completion, line editing, and history:

```bash
python3 cmat_webviewer.py -i
```

```text
cmat [no-matrix]> load GeE-symm.cmat inbeam
cmat [inbeam]> cal show
cmat [inbeam]> search 0 --snr 10.0
cmat [inbeam]> gate 0 w 1170 1176
cmat [inbeam]> fit_1d 1 1332.5
cmat [inbeam]> pdf_1d 1 inbeam_gated.pdf --fit
cmat [inbeam]> quit
```

---

## Per-Axis Energy Calibration

`python-cmat` supports independent energy calibrations for each detector axis, enabling straightforward analysis of asymmetric matrices (e.g. Energy vs. Time, Particle Energy vs. Gamma Energy, or Detector Segment vs. Energy):

### Calibration Equation
$$E(\text{ch}) = a_0 + a_1 \cdot \text{ch} + a_2 \cdot \text{ch}^2$$

### Command Syntax
- `cal <axis> <a0> <a1> [a2]`: Set quadratic calibration for Det 1 / Axis 0 (`0`) or Det 2 / Axis 1 (`1`).
- `cal <a0> <a1> [a2]`: Set identical calibration parameters for both axes (for symmetric matrices).
- `cal show`: Print active calibration polynomials, quadratic terms, and units for both axes.
- `cal clear [axis]`: Clear calibration and revert to raw channel coordinates.

### CLI Startup Flags
You can also supply calibration coefficients at startup:
- `--cal-0 "0.5 1.002 0.000001"`: Set calibration for Det 1 (Axis 0 / X).
- `--cal-1 "0.4 1.001 0.0000005"`: Set calibration for Det 2 (Axis 1 / Y).
- `--cal "0.0 1.0 0.0"`: Set symmetric calibration for both axes.

---

## Spectroscopy Command Reference

| Command | Arguments | Description |
|---|---|---|
| **`load`** | `<filepath> [alias]` | Load a `.cmat` matrix file into the session with an optional alias name. |
| **`matrix`** | `<name_or_index>` | Switch the active matrix for analysis while preserving existing coincidence gates and fits. |
| **`list`** | *None* | List all loaded matrices with shapes, total counts, and symmetry status. |
| **`info`** | *None* | Display active matrix dimensions, step sizes, statistics, and calibration formulas. |
| **`close`** | `[name_or_index]` | Unload the specified (or active) matrix from memory. |
| **`cal`** | `<axis> <a0> <a1> [a2]` | Set quadratic energy calibration polynomial coefficients. |
| **`cal show`** | *None* | Display the calibration table and formulas for all axes. |
| **`cal clear`** | `[axis]` | Reset calibration to raw integer channel numbers. |
| **`gate`** | `<axis> w <w0> <w1> [x <x0> <x1>]` | Set coincidence peak (`w`) and normalized background subtraction (`x`) windows. |
| **`gate clear`** | `[axis]` | Clear active coincidence gates and restore full matrix projections. |
| **`gate show`** | *None* | Display active gate slices, channel widths, and background normalization scale factor. |
| **`search`** | `[axis] [--method M] [--snr N]` | Perform automated peak detection (`cwt`, `prominence`, or `mariscotti`). |
| **`fit_1d`** | `<axis> <ch_or_e> [--model M]` | Single peak fit (`gaussian`, `gaussian_tail`, or `hypermet`) with error propagation. |
| **`fit_multiplet`**| `<axis> <p1> <p2> ...` | Simultaneously fit a coupled multiplet cluster with full parameter covariance. |
| **`fit_all`** | `[axis] [--range min max] [--snr N]`| Auto-fit all candidate peaks in range on the Peak-Aware Continuum baseline. |
| **`bg_1d`** | `<axis> <b0> <b1> [...]` or `clear` | Define or clear discrete 1D spectroscopy background windows for fitting and integration. |
| **`integrate`** | `<axis> <r0> <r1> [--bg ..] [--poly 1\|2]` | Integrate peak area over region with linear or quadratic polynomial background subtraction. |
| **`fit_2d`** | `<x> <y> [--roi N] [--verbose]` | True 2D coincidence peak fit (Gamba 4-component decomposition). |
| **`clear_fits`** | `[1d\|2d\|all]` | Clear stored fit results from memory. |
| **`pdf_1d`** | `<axis> <out.pdf> [--fit] [--title T]`| Export publication-grade vector PDF of 1D or gated spectrum. |
| **`pdf_2d`** | `<out.pdf> [--x ..] [--y ..] [--fit]` | Export publication-grade vector PDF of 2D coincidence matrix. |
| **`export_1d`** | `<axis> <out.dat>` | Export 1D spectrum to ASCII data table with Poisson standard deviations. |
| **`export_amat`** | `<out.mat>` | Export 2D matrix to ASCII matrix format (`dense` or `sparse`). |
| **`macro`** | `<filepath>` | Execute commands from an external macro script file. |
| **`echo`** | `<message>` | Print message or status text to the console. |
| **`sleep`** | `<seconds>` | Pause execution for the specified duration. |
| **`quit` / `exit`**| *None* | Terminate macro execution or exit interactive shell. |

---

## Macro Script Example (`analysis_template.mac`)

A comprehensive macro template is provided in [`analysis_template.mac`](https://github.com/rlica/python-cmat/blob/main/analysis_template.mac) in the repository root. Below is an annotated excerpt illustrating a complete automated analysis workflow:

```text
# ==============================================================================
# python-cmat Automated Spectroscopy Macro
# ==============================================================================

echo "=== Loading Coincidence Matrix ==="
load GeE-symm.cmat run1

echo "=== Setting Quadratic Energy Calibrations ==="
cal 0 0.5 1.0024 0.00000012
cal 1 0.5 1.0024 0.00000012
cal show

echo "=== Automated Peak Search on Det 1 Total Projection ==="
search 0 --method cwt --snr 9.0

echo "=== Setting 1173 keV Coincidence Gate with Normalized Background ==="
gate 0 w 1168.0 1178.0 x 1145.0 1155.0
gate show

echo "=== Fitting Coincident Photopeaks on Det 2 ==="
fit_1d 1 1332.5 --model gaussian_tail

echo "=== Exporting Vector PDFs and ASCII Data ==="
pdf_1d 1 run1_gated_1332keV.pdf --fit --title "1173 keV Gate -> 1332 keV Coincidence"
export_1d 1 run1_gated_1332keV.dat

echo "=== True 2D Coincidence Decomposition ==="
fit_2d 1173.2 1332.5 --roi 18 --verbose
pdf_2d run1_2D_fit.pdf --fit

echo "=== Analysis Complete ==="
exit
```
