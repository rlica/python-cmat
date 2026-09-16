# Keyboard Shortcuts & Controls

The `cmat_webviewer` GUI is designed for high-efficiency nuclear spectroscopy with comprehensive keyboard shortcuts and intuitive mouse interactions inspired by the classic [GASPware](https://github.com/csteke/GASPware) `cmat` and `xtrackn` workflows.

---

## Complete Shortcut Reference

### Mouse Interactions

| Action | Shortcut / Gesture | Target Panel | Description |
|---|---|---|---|
| **Box Zoom** | `Click & Drag` | 2D Matrix | Zoom into selected rectangular region of the 2D coincidence matrix |
| **1D Energy Zoom** | `Click & Drag` | 1D Spectrum | Zoom into selected energy/channel range on the X axis (synchronizes 2D viewport & orthogonal 1D projection) |
| **2D Wheel Zoom** | `Mouse Wheel` | 2D Matrix | Zoom In / Out centered on the current crosshair position in discrete increments |
| **1D Y-Axis Zoom** | `Mouse Wheel` | 1D Spectrum | Dynamic vertical zoom In / Out on counts scale (preserves zero baseline while adjusting maximum) |
| **1D X-Axis Zoom** | `Shift + Mouse Wheel` | 1D Spectrum | Zoom In / Out horizontally centered at the cursor position |
| **Full Zoom Out (2D)** | `Double Click` | 2D Matrix | Fully zoom out both the 2D matrix and both 1D spectra to the full 4096×4096 matrix view |
| **Full Zoom Out (1D)** | `Double Click` | 1D Spectrum | Reset horizontal and vertical zoom for the clicked 1D projection only (preserves coincidence gates) |
| **Adjust Panel Width** | `Click & Drag Divider` | Split Bar | Adjust relative width ratio between 2D matrix and 1D projections (double-click divider to reset) |

---

### Peak Fitting & Area Integration

| Action | Shortcut | Target Panel | Description |
|---|---|---|---|
| **2D Coincidence Peak Fit** | `Ctrl / Cmd + Click` or `V` | 2D Matrix | True 2D coincidence peak fit (Gaussian, RadWare, or Hypermet) with Gamba & Morhác 4-component background decomposition ($p\|p^t$, $p\|bg$, $bg\|p$, $bg\|bg$). Displays FWHM ellipses, centroid crosshairs, and adds centroid markers to both 1D projections until cleared with `=`. |
| **1D Peak / Multiplet Fit** | `Ctrl / Cmd + Click` or `V` | 1D Spectrum | Fit single 1D peak or coupled multiplet cluster with full parameter covariance error propagation. Uses background regions `B` if set. Fits remain persistently on display. |
| **Automatic 1D Peak Search** | `P` or `p` | 1D Spectrum | Automatically detect candidate photopeaks in the focused 1D projection (Continuous Wavelet Transform `CWT`, Prominence, or Mariscotti methods). |
| **Add / Remove Manual Peak Marker**| `G` or `g` | 1D Spectrum | Add a manual peak marker at the cursor with parabolic sub-channel centroid refinement (or delete existing marker if hovering within 1.2 channels). |
| **Set Fit / Integration Region** | `R` or `r` | 1D Spectrum | Press once to set left limit ($R_{\text{left}}$); move cursor and press again to set right limit ($R_{\text{right}}$). Binds fitting and integration to $[R_{\text{left}}, R_{\text{right}}]$ with straight-line background across averaging wings. |
| **Set 1D Background Windows** | `B` or `b` | 1D Spectrum | Define discrete left and right baseline sample intervals for 1D peak fitting (`V`) and integration (`I`). |
| **Integrate Peak Area** | `I` or `i` | 1D Spectrum | Numerically integrate peak area over Region `R`, subtracting linear or polynomial background (`B` or boundary wings). |
| **Fit All Visible Peaks** | `H` or `h` | 1D Spectrum | Automatically fit all visible candidate photopeaks on display using the Peak-Aware Continuum baseline engine (averages noise grass without clipping bias). |
| **Clear Peak Fits & Markers** | `=` or `+` | Global | Clear all active persistent 1D and 2D fits, fit curves, centroid labels, integration results, and peak search markers. |

---

### Coincidence Gate Slicing & Background Subtraction

| Action | Shortcut | Target Panel | Description |
|---|---|---|---|
| **Set Coincidence Peak Gate** | `W` or `w` | 1D Spectrum | Press once to set left limit ($W_{k,\min}$); press again to set right limit ($W_{k,\max}$). Slices the 2D matrix along the orthogonal axis. Multiple gates can be defined sequentially. |
| **Set Coincidence Background Gate**| `X` or `x` | 1D Spectrum | Set left and right limits for background slices ($X_m$). Subtracted with channel-width normalization scale factor $\sum \Delta W / \sum \Delta X$. |
| **Clear Active Coincidence Gate** | `Z` or `z` | 1D Spectrum | Clear all active coincidence gates and background cuts (or cancel in-progress limit marker), restoring the full matrix projection. |

---

### Viewport Navigation & Marker Limits

| Action | Shortcut | Target Panel | Description |
|---|---|---|---|
| **Pan 1D Channels (Horizontal)** | `←` / `→` | 1D Focus | Pan 1D spectrum left / right (`Shift` for 2× speed). |
| **Pan 1D Counts (Vertical)** | `↑` / `↓` | 1D Focus | Adjust vertical counts scale (`↑` magnify, `↓` compress). |
| **Set Left Limit ($X_{\min}$)** | `Left Arrow (←)` | 2D Focus | Place Left limit marker at cursor X coordinate. |
| **Set Right Limit ($X_{\max}$)** | `Right Arrow (→)` | 2D Focus | Place Right limit marker at cursor X coordinate. |
| **Set Down Limit ($Y_{\min}$)** | `Down Arrow (↓)` | 2D Focus | Place Lower limit marker at cursor Y coordinate. |
| **Set Up Limit ($Y_{\max}$)** | `Up Arrow (↑)` | 2D Focus | Place Upper limit marker at cursor Y coordinate. |
| **Pan 2D View** | `Shift + Arrows` | 2D Focus | Pan visible 2D matrix view in steps determined by Scroll Sensitivity setting. |
| **Expand to Limits** | `E` or `e` | 2D Matrix | Zoom into the rectangular region bounded by active limit markers. |
| **Full Matrix View** | `F` or `f` | Global | Reset zoom to the full $4096 \times 4096$ matrix range. |

---

### Display, Scale & Colormap Controls

| Action | Shortcut | Target Panel | Description |
|---|---|---|---|
| **Contextual Scale Cycle** | `L` or `l` | Focused Panel | Cycle intensity/counts scale of panel currently under cursor:<br>&bull; **2D**: Log $\rightarrow$ Linear $\rightarrow$ Power ($\sqrt{N}$)<br>&bull; **1D**: Linear $\leftrightarrow$ Log |
| **2D Linear Scale** | `1` | 2D Matrix | Switch 2D matrix colormap scaling directly to Linear. |
| **2D Power ($\sqrt{N}$) Scale** | `2` | 2D Matrix | Switch 2D matrix colormap scaling directly to Power (Square Root). |
| **2D Logarithmic Scale** | `4` | 2D Matrix | Switch 2D matrix colormap scaling directly to Logarithmic. |
| **Cycle Colormap** | `C` or `c` | 2D Matrix | Cycle forward through colormaps: Turbo $\rightarrow$ Viridis $\rightarrow$ Plasma $\rightarrow$ Inferno $\rightarrow$ Hot $\rightarrow$ Jet $\rightarrow$ Gray. |
| **Toggle Sidebar Menu** | `M` or `m` | Global | Expand or collapse the left control sidebar to maximize canvas space. |

---

### File, Session & Export Controls

| Action | Shortcut / Button | Description |
|---|---|---|
| **Previous / Next Matrix** | `[` / `]` | Switch active matrix to previous/next loaded file while locking viewport coordinates, coincidence gates, and auto-refitted peak fits. |
| **Browse Server Files** | `Ctrl + O` or `📂` | Open interactive server filesystem browser modal to navigate folders and load `.cmat` files. |
| **Export 2D Vector PDF** | `Print PDF` (2D footer) | Export publication-grade vector PDF of visible 2D matrix with Times New Roman typography, calibrated keV axes, colorbar, 2D fit ellipses, and fit labels. |
| **Export 1D Vector PDF** | `Print PDF` (1D header) | Export publication-grade vector PDF of active 1D spectrum with stepped histogram, calibrated keV axis, continuous fit curves, baselines, and labeled centroids. |
| **Save Configuration** | `Save Config` (Sidebar) | Save active UI parameters, calibrations, and colormaps to `python-cmat-config.txt` in working directory. |
| **Keyboard Shortcuts Help** | `?` | Display interactive keyboard shortcut and help reference overlay. |
| **Quit Viewer** | `Q` or `q` | Close browser window and terminate the terminal server process. |
