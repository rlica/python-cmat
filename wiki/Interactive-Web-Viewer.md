# Interactive Web Viewer

The `cmat_webviewer.py` server and companion `cmat_webviewer.html` client provide an ultra-responsive, browser-based GUI for exploring 2D $\gamma$-$\gamma$ coincidence matrices and conducting nuclear structure analysis.

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
