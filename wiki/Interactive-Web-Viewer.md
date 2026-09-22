# Interactive Web Viewer

The `cmat_webviewer.py` server and companion `cmat_webviewer.html` client provide an ultra-responsive, browser-based GUI for exploring 2D matrices in the `.cmat` format and conducting nuclear physics data analysis.

While the viewer functions as a high-performance visualizer for arbitrary 2D `.cmat` datasets, it features dedicated subroutines tailored for $\gamma$-$\gamma$ coincidence matrices (including multi-gate coincidence slicing with normalized background subtraction, simultaneous dual 1D projections, and 2D coincidence peak fitting). Future releases will extend support to 3D matrices and time-difference spectra.

<p align="center">
  <img width="900" alt="cmat_webviewer interface" src="https://github.com/user-attachments/assets/460ab83c-cbfa-4ee8-9194-f57c58e5215e" />
</p>

---

## Launching the Web Viewer

Run the server script from your terminal, pointing to one or more `.cmat` files:

```bash
# Open a single matrix
python cmat_webviewer.py /path/to/matrix.cmat

# Open multiple matrices
python cmat_webviewer.py run01.cmat run02.cmat run03.cmat

# Load all matrices in a folder
python cmat_webviewer.py ./data/*.cmat
```

The server binds to `http://0.0.0.0:8080` (or the configured host/port) and automatically launches your default web browser.

---

## Windows Subsystem for Linux (WSL) Configuration

When running inside WSL on Windows, launching native GUI applications via X11 can cause window-grab issues and slow rendering. `cmat_webviewer` leverages your host Windows browser directly for full hardware-accelerated WebGL/Canvas rendering.

To enable seamless browser launching from WSL, install `wslu` and configure `BROWSER`:

```bash
sudo apt update && sudo apt install wslu
echo 'export BROWSER=wslview' >> ~/.bashrc
source ~/.bashrc
```

Once configured, launching `cmat_webviewer.py` inside WSL will automatically open your default Windows browser.

---

## Remote Server & SSH Usage

When analyzing data stored on a remote Linux workstation or cluster via SSH:

### 1. Direct Terminal Link (`Ctrl + Click`)
The viewer prints the server's network address in the terminal upon startup (e.g. `http://192.168.1.50:8080`). Modern terminals (VS Code, Windows Terminal, iTerm2, GNOME Terminal) allow you to **`Ctrl+Click` (or `Cmd+Click` on macOS)** the URL to open the remote session in your local browser immediately.

### 2. Automatic Browser Suppression
When running inside an SSH session (detected via `$SSH_CLIENT` / `$SSH_TTY`), `cmat_webviewer.py` automatically suppresses launching headless or remote X11 browsers, keeping the SSH connection fast and lightweight.

### 3. SSH Port Forwarding
If the remote machine is behind a firewall or lacks direct port exposure:

```bash
# Forward remote port 8080 to your local machine:
ssh -L 8080:localhost:8080 user@remote-server

# Run the viewer on the remote machine:
python3 cmat_webviewer.py matrix.cmat
```

Then open `http://localhost:8080` in your local browser.

---

## Core Architecture & GUI Features

### 1. Pixel-Matched 2D Canvas with Max-Pooling
- Renders matrices up to $6144 \times 6144$ channels effortlessly by fetching only as many data points as screen pixels available.
- **2D Max-Pooling**: When zoomed out over large channel ranges, downsampling uses maximum-intensity pooling rather than naive averaging or nearest-neighbor interpolation. This ensures sharp, narrow gamma photopeaks never disappear at full zoom.
- Direct scale options: **Logarithmic**, **Linear**, and **Power** ($\sqrt{N}$) scaling with real-time colormap mapping (Turbo, Viridis, Plasma, Inferno, Hot, Jet, Gray).

### 2. Simultaneous Dual 1D Projections
- **Det 1 (X Projection)**: Slices the 2D matrix along the visible Y-window, showing total or gated counts vs. Det 1 energy.
- **Det 2 (Y Projection)**: Slices the 2D matrix along the visible X-window, showing total or gated counts vs. Det 2 energy.
- Synchronized crosshair cursors: Moving the mouse in the 2D matrix updates highlighted crosshairs and channel/energy readouts simultaneously across both 1D panels.
- Stepped staircase rendering with dynamic Y-axis auto-scaling, calibrated keV readouts, and ASCII `.dat` export.

### 3. Collapsible Sidebar & Resizable Layout
- Press **`M`** (or click the `Menu` toggle) to collapse the left control panel, maximizing matrix visualization space.
- An interactive vertical divider bar allows click-and-drag resizing between the 2D matrix canvas and the 1D projection histograms (double-click the divider to reset to equal width).

### 4. Context-Aware Arrow Navigation
Keyboard arrow keys automatically route actions based on which panel is under cursor focus:
- **1D Spectrum Focus**:
  - `←` / `→` pans channels horizontally (`Shift` for 2× speed).
  - `↑` / `↓` pans and magnifies the vertical counts scale.
- **2D Matrix Focus**:
  - `←` / `→` / `↓` / `↑` sets Left, Right, Down, and Up limit markers at the crosshair.
  - `Shift + Arrows` pans the 2D view.

### 5. Multi-Matrix Differential Analysis
Open multiple matrices simultaneously to perform comparative spectroscopy:
- **Instant Switching**: Cycle between loaded matrices using the header dropdown or the `[` and `]` bracket keys.
- **Analysis State Locking**: Switching matrices preserves:
  - Viewport coordinates and zoom ranges
  - Active coincidence gates ($W$) and background subtractions ($X$), which are instantly re-evaluated against the new matrix
  - Active 1D single-peak and multiplet fits (automatically refitted with updated areas and FWHMs)
  - Active 2D coincidence peak fits (refitted at identical coincidence coordinates)
  - Visual contrast settings ($V_{\max}$, $V_{\min}$, colormap, scale mode)

### 6. Interactive Server Filesystem Browser
Click **`📂 Browse Server Matrices...`** in the header (or press **`Ctrl + O`**) to open the server-side file browser modal. You can navigate the remote directory structure, view file sizes and timestamps, filter files by keyword, and load `.cmat` matrices directly without typing paths.

### 7. Publication-Quality Vector PDF Export
Click **`Print PDF`** on the 2D matrix footer or either 1D projection header:
- Outputs standalone, vector PDFs rendered with classic publication typography (Times New Roman).
- Embeds energy-calibrated axes, stepped histogram paths, all active continuous fit curves, background baselines, labeled centroid pointers, 2D FWHM confidence ellipses, and unobtrusive 3-line fit summary labels.

### 8. 2D Banana ROIs & Area Determination (Peak & Background Subtraction)
For irregular or curve-shaped features on the 2D coincidence matrix (e.g. bananas, Doppler-shifted diagonal ridges, or non-rectangular regions of interest), `cmat_webviewer` provides interactive polygonal ROI drawing with area-normalized background subtraction and terminal reporting:

- **Draw Peak Banana (W)**: Press **`Shift + G`** (or click **Draw Peak [Shift+G]** in the 2D footer or sidebar). Click vertices on the 2D matrix canvas to trace the Peak ROI boundary (rendered in gold `#ffd600`). Close the polygon by clicking near the starting vertex or pressing **`Enter`**.
- **Draw Background Banana (B)**: Press **`Shift + B`** (or click **Draw Bg [Shift+B]**). Click vertices on the 2D matrix canvas to trace the Background ROI boundary (rendered in magenta `#ff4081`). Close with **`Enter`** or by clicking near the start vertex.
- **Area-Normalized Subtraction**:
  - Exact discrete pixel containment ($N_{\text{px}}$) and continuous geometric Shoelace area ($A\ \text{ch}^2$) are calculated for both polygons.
  - Scale factor: $\text{Scale} = \text{Area}_{\text{peak}} / \text{Area}_{\text{bg}}$.
  - Net Area Counts:
    $$\text{Net Counts} = C_{\text{peak}} - \text{Scale} \times C_{\text{bg}} \quad \pm \quad \sqrt{C_{\text{peak}} + \text{Scale}^2 \times C_{\text{bg}}}$$
- **Real-Time Readout & Diagnostics**:
  - A persistent top badge overlay on the 2D matrix and the sidebar card display live vertex counts, surface area ($\text{ch}^2$), discrete pixel count, raw counts, scale factor, and net background-subtracted counts.
  - A formatted diagnostic summary is automatically printed to the terminal console whenever a banana ROI is closed or when switching between matrices:
    ```text
    ================================================================================
    [2D Banana ROI Analysis] GeE-symm.cmat (4096 x 4096)
    --------------------------------------------------------------------------------
      • Peak Banana (W):  1,250 px (area: 1250.0 ch², 5 vertices) | Counts: 345,210 cts
      • Bg Banana (B):    2,500 px (area: 2500.0 ch², 4 vertices) | Counts: 110,400 cts (Scale factor: 0.5000)
    --------------------------------------------------------------------------------
      => Net Area Counts: 290,010.0 ± 608.2 cts
    ================================================================================
    ```
- **Clear Banana ROIs**: Press **`Z`** (or click **Clear Bananas**) to remove both peak and background polygons and reset the display.

### 9. Automatic 2D Coincidence Peak Search (`P` in 2D)
Press **`P`** while focusing the 2D matrix (or click **`Find 2D Peaks [P]`** in the 2D matrix footer or Sidebar Section 4.3):
- **4-Stage Hybrid Search Engine**:
  1. **Projection Seeding**: Performs multi-scale CWT peak search on both Det 1 and Det 2 projections to generate candidate coordinate pairs $(x_i, y_j)$.
  2. **Local Gamba 4-Component Decomposition**: Fits each candidate subregion to separate true coincidence volume ($p|p^t$) from vertical ridges ($p|bg$), horizontal ridges ($bg|p$), and continuum ($bg|bg$).
  3. **Compton Scattering Ridge & Cross-Talk Rejection**: Automatically rejects 1D single-gamma Compton cross-ridges (where $\Pi = n_{p|p}^t / n_{p|p}^m \le \text{threshold}$) and detector cross-talk diagonal artifacts ($\rho_{xy} \approx -1$).
  4. **Non-Maximum Suppression (NMS) & Levenberg-Marquardt Fit**: Eliminates redundant split fits and refines coincidence centroids, widths ($\text{FWHM}_X, \text{FWHM}_Y$), and net volume.
- **Configurable Sensitivity**: Adjust **Min Peak SNR** (1.5–20.0) and **Min Gamba $\Pi$ Ratio** (0.01–0.90) sliders in Sidebar Section 4.3 to tune sensitivity for weak transitions vs. noisy matrices.
- **Collision-Free, Zoom-Adaptive 2D Labels**:
  - At wide zoom levels, clean crosshair markers and FWHM ellipses are shown to keep the view uncluttered.
  - As you zoom into coincidence regions, text labels ($E_X \times E_Y$, net counts) smoothly appear using dynamic bounding box collision avoidance.
  - Hovering any peak marker instantly brings its full energy label and summary badge to the foreground.
- **Automated Logging**: When **`📝 Log Fits: ON`** is enabled, all detected 2D coincidence fits are automatically saved to `fit_results_<timestamp>.txt`.
- **Terminal Report**: A structured summary table with energies, net volume, Gamba areas, FWHMs, $\chi^2$, and peak-to-background ratios is printed directly to the terminal.

### 10. Fit Results Text File Logging
- Toggleable via the **`📝 Log Fits: ON/OFF`** UI button in the viewer header, the `--fit-log` startup flag, or the `fit_log [on|off]` macro command.
- Each 1D photopeak or 2D coincidence fit (both manual `Ctrl+Click` / `V` and automatic `P` in 2D / `H` in 1D) is cleanly appended as a single fixed-width, right-aligned row into `fit_results_<timestamp>.txt` in the server directory.
- Accommodates 10+ digit area counts and error bars with perfect column alignment:
  - **1D Fits**: `Energy(err)`, `Net_Area(err)`, `FWHM(err)`, `Chi2`, `Peak_to_BG`
  - **2D Fits**: `Energy1(err)`, `Energy2(err)`, `Net_Area(err)`, `Gamba_Area(err)`, `FWHM1(err)`, `FWHM2(err)`, `Chi2`, `Peak_to_BG`

### 11. Automated ENSDF Isotope Identification Pop-up
Click **`🔬 Isotope Identification`** in the header to launch the standalone nuclear structure search window (`/ensdf_popup.html`):
- **Server File Browser (`📂 Browse...`)**: Interactively search and select any `fit_results_*.txt` log file on the server.
- **Top 5 Candidate Isotopes**: For each 2D coincidence fit, displays the top physical cascade matches with colored badges (`Direct Cascade (Prompt Coincidence)`, `Sequential Cascade`, `Same Level Scheme`) and expandable sub-tables.
- **Global Parsimony & Mass-Clustering**: Automatically clusters isotopes by dominant reaction/decay mass ($\bar{A}$) and finds the minimal set of isotopes explaining all 1D and 2D features.
- See [ENSDF Isotope Identification](ENSDF-Isotope-Identification) for full theory and algorithms.



