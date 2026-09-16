# Configuration Guide (`python-cmat-config.txt`)

`python-cmat` supports directory-specific, portable configuration files (`python-cmat-config.txt`). This ensures that calibrations, visual preferences, fit parameters, and server settings persist for each analysis folder without hardcoding values in scripts or modifying source code.

---

## Configuration File Discovery

When `cmat_webviewer.py` launches, it follows an automatic discovery workflow:

1. **Working Directory Check**: The script inspects the current working directory (`Path.cwd()`) for `python-cmat-config.txt`.
2. **Automatic Creation**: If no file is found, it automatically writes a default `python-cmat-config.txt` with well-commented standard settings:
   ```text
   [*] No config file found. Created default config: python-cmat-config.txt
   ```
3. **Loading**: If present, it parses and applies the settings immediately:
   ```text
   [*] Loaded configuration from python-cmat-config.txt
   ```
4. **CLI Flag Precedence**: Explicit command-line arguments (such as `--cal-0`, `-p / --port`, `--browser`) take precedence and override the settings in the configuration file for that session.

---

## Parameter Reference

Below is a complete, fully annotated example of `python-cmat-config.txt`:

```ini
# ==============================================================================
# python-cmat configuration file
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Energy Calibration (Quadratic: E = a0 + a1*ch + a2*ch^2)
# ------------------------------------------------------------------------------
# Set independent calibrations per axis:
# Det 1 / X-axis (cal_0) and Det 2 / Y-axis (cal_1)
cal_0 = 0.0, 1.0, 0.0
cal_1 = 0.0, 1.0, 0.0

# Shorthand for symmetric matrices (applies to both axes if cal_0/cal_1 omitted):
cal = 0.0, 1.0, 0.0

# ------------------------------------------------------------------------------
# 2. Peak Fitting & Model Options
# ------------------------------------------------------------------------------
# Default Peak Function Model:
# - gaussian: Standard symmetric Gaussian with linear baseline
# - gaussian_tail: RadWare / SAMPO piecewise exponential left tail
# - hypermet: Convolved exponential tail + erfc Compton step
fit_type = gaussian

# 1D Peak Fitting Region multiplier (times estimated peak FWHM, e.g. 1.0 to 10.0)
fwhm_mult_1d = 4.0

# 2D Coincidence ROI half-width in channels (e.g. 6 to 36)
roi_half_width_2d = 16

# Fit results verbosity in terminal: compact, detailed
fit_verbosity = compact

# ------------------------------------------------------------------------------
# 3. 2D Matrix Display & Colormaps
# ------------------------------------------------------------------------------
# Default 2D Colormap:
# turbo, viridis, plasma, inferno, hot, jet, gray
colormap = turbo

# Default 2D Intensity Scale Mode:
# log (logarithmic), sqrt (power/square root), linear
scale_mode = log

# Default Max Contrast (vmax, 0-1000) and Min Threshold (vmin)
vmax = 500
vmin = 1

# Mouse wheel zoom sensitivity percentage per step (1 to 15)
scroll_sensitivity = 4

# ------------------------------------------------------------------------------
# 4. 1D Projection & Peak Search Parameters
# ------------------------------------------------------------------------------
# 1D Automatic Peak Search Method:
# - cwt: Continuous Wavelet Transform ridge detection (default, highly robust)
# - prominence: Topographical prominence with Poisson noise threshold
# - mariscotti: GASPware second-difference filtering
peak_search_method = cwt

# 1D Peak Search Sensitivity / Minimum Signal-to-Noise Ratio (e.g. 1.0 to 15.0)
peak_search_snr = 9.0

# 1D Projection Display Range:
# - synced: Matches the visible X/Y range of the 2D matrix
# - full: Displays the full 0-4096 channel range regardless of 2D zoom
proj_range = synced

# 1D Projection Vertical Scale Mode: linear, log
proj_scale = linear

# ------------------------------------------------------------------------------
# 5. Web Server & Browser Launch Settings
# ------------------------------------------------------------------------------
# Bind interface (0.0.0.0 binds to all available network interfaces)
host = 0.0.0.0

# Web server TCP port
port = 8080

# Automatically open web browser upon server launch: true, false
open_browser = true

# Preferred browser executable:
# default, firefox, google-chrome, chromium, safari, none
browser = default
```

---

## Saving Settings from the Web UI

You do not need to manually edit `python-cmat-config.txt` by hand:
1. Adjust any sliders, colormap selectors, fit models, or calibration parameters directly in the **Configuration** panel of the Web Viewer.
2. Click **`Save Config to File`** (or use the sidebar button).
3. The server immediately rewrites `python-cmat-config.txt` in the working directory with your active values.
