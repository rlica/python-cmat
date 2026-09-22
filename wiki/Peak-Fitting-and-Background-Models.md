# Peak Fitting & Background Models

The peak fitting algorithms, baseline estimation engines, and background decomposition methodologies implemented in `python-cmat` adhere to established analytical practices in high-resolution semiconductor $\gamma$-ray spectroscopy.

---

## 1. Photopeak Models (1D & 2D)

While ideal charge generation and preamplifier electronics produce symmetric Gaussian photopeaks, real High-Purity Germanium (HPGe) and Ge(Li) detectors exhibit low-energy tailing caused by:
- **Incomplete Charge Collection & Hole Trapping**: Lower drift mobility and radiation damage defect centers trap charge carriers before collection, attenuating pulse height.
- **Ballistic Deficit**: Charge collection rise times varying with interaction location, occasionally exceeding amplifier shaping times.
- **Small-Angle Compton Scattering**: Photons scattering in cryostat windows, dead layers, or target holders prior to full absorption.

`python-cmat` provides three analytical models for 1D projections and 2D coincidence peak fitting:

---

### Option 1: Standard Symmetric Gaussian (`gaussian`)
A pure Gaussian photopeak sitting on a linear baseline:

$$G(x) = H \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)$$

- **Net Peak Area**:
  $$A_{\text{net}} = \sqrt{2\pi} H \sigma \approx 2.5066 \cdot H \cdot \sigma$$
- **Full Width at Half Maximum (FWHM)**:
  $$\text{FWHM} = 2\sqrt{2\ln 2} \sigma \approx 2.35482 \cdot \sigma$$
- **Best For**: Fast preliminary fits, well-shaped scintillation detector peaks (LaBr$_3$, NaI), or HPGe peaks with negligible trapping.

---

### Option 2: RadWare / SAMPO Piecewise Left Exponential Tail (`gaussian_tail`)
The classic formulation established in RadWare (`gf3`), SAMPO, and `xtrackn` (*Helmer & Lee, 1980*; *Radford, 1995*; *Routti & Prussin, 1969*). The exponential tail joins the Gaussian at $z = (x - \mu)/\sigma = -\alpha$ with continuous amplitude and continuous first derivative:

$$P(x) = \begin{cases} H \exp\left(-\frac{1}{2} z^2\right), & z \ge -\alpha \\ H \exp\left(\frac{1}{2}\alpha^2 + \alpha z\right), & z < -\alpha \end{cases}$$

- **Net Peak Area**:
  $$A_{\text{net}} = H \sigma \left[ \sqrt{\frac{\pi}{2}}\left(1 + \text{erf}\left(\frac{\alpha}{\sqrt{2}}\right)\right) + \frac{1}{\alpha}\exp\left(-\frac{1}{2}\alpha^2\right) \right]$$
- **Parameters**: $\alpha > 0$ defines the tail join distance in units of $\sigma$ below centroid $\mu$.
- **Best For**: Standard HPGe spectroscopy with mild low-energy tailing.

---

### Option 3: Hypermet Analytical Model (`hypermet`)
The rigorous analytical formulation combining a symmetric Gaussian $G(x)$, an analytical Exponentially Modified Gaussian (EMG) convolved tail $T(x)$, and an error-function Compton step $S(x)$ (*Phillips & Marlow, 1976*; *Campbell & Maxwell, 1993*):

$$P(x) = G(x) + T(x) + S(x)$$

where:

$$G(x) = H \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)$$

$$T(x) = \frac{f_T}{2\beta} \exp\left[\frac{x-\mu}{\beta} + \frac{\sigma^2}{2\beta^2}\right] \text{erfc}\left[\frac{x-\mu}{\sqrt{2}\sigma} + \frac{\sigma}{\sqrt{2}\beta}\right]$$

$$S(x) = \frac{A_S}{2} \text{erfc}\left[\frac{x-\mu}{\sqrt{2}\sigma}\right]$$

- **Parameters**:
  - $H$: Gaussian peak amplitude.
  - $\mu, \sigma$: Centroid and Gaussian dispersion width.
  - $f_T$: Integrated area of the convolved exponential tail.
  - $\beta$: Exponential decay slope length (channels).
  - $A_S$: Amplitude of the Compton step function.
- **Total Net Peak Area**:
  $$A_{\text{net}} = \sqrt{2\pi} H \sigma + f_T$$
- In 2D coincidence matrices, the true coincidence peak is evaluated as the 2D tensor product $H \cdot P_X(x) \cdot P_Y(y)$ with total net volume:
  $$V_{\text{fit}} = H (\sqrt{2\pi}\sigma_x + \eta_{Tx})(\sqrt{2\pi}\sigma_y + \eta_{Ty})$$

---

## 2. Channel Bin Centering Convention (`! al centro del canale`)

In multichannel analyzer (MCA/ADC) spectroscopy and the standard [GASPware](https://github.com/csteke/GASPware) / `xtrackn` suite, an integer histogram bin $k$ represents counts collected in the channel interval $[k, k+1)$ with its physical center located at:

$$\text{Physical Center} = k + 0.5$$

`python-cmat` strictly adheres to this Fortran `xtrackn` convention (*"al centro del canale"*, `src/xtrack/trackn.F`):
- All 1D and 2D continuous peak fitting routines evaluate analytical models on bin centers ($x_i = k + 0.5, y_j = l + 0.5$).
- Calibrated peak energies $E(\mu) = a_0 + a_1 \mu + a_2 \mu^2$ are evaluated directly at the continuous centroid $\mu$, guaranteeing exact 1:1 energy calibration agreement with `xtrackn`.
- Peak curves and centroid marker stems naturally align with the exact center of histogram bins on both browser canvases and vector PDF exports.

---

## 3. Self-Consistent 2D $\gamma$-$\gamma$ Background Decomposition (Gamba & Morhác)

In 2D $\gamma$-$\gamma$ coincidence spectroscopy, counts in the vicinity of a coincidence peak $(E_d, E_f)$ are composed of four distinct topological components (*Gamba et al., NIM A 928 (2019) 93–103*; *Morhác et al., NIM A 401 (1997) 113*):

1. **$p|p^t$ (True Net Coincidence Peak)**: Genuine correlated full-energy cascade events ($E_d \otimes E_f$), modeled as a 2D peak profile $H \cdot P_X(x) \cdot P_Y(y)$.
2. **$p|bg$ (Det 1 Peak with Det 2 Continuum / Random Ridge)**: Full energy in Det 1 ($E_d$) and Compton continuum (or randoms) in Det 2 $\rightarrow$ forms a vertical cross-ridge $R_x \cdot P_X(x)$.
3. **$bg|p$ (Det 2 Peak with Det 1 Continuum / Random Ridge)**: Full energy in Det 2 ($E_f$) and Compton continuum (or randoms) in Det 1 $\rightarrow$ forms a horizontal cross-ridge $R_y \cdot P_Y(y)$.
4. **$bg|bg$ (2D Compton Continuum & Accidental Coincidences)**: Compton continuum and time-uncorrelated random coincidences in both detectors $\rightarrow$ modeled as a 2D planar baseline $b_0 + b_x(x - x_c) + b_y(y - y_c)$.

### No User-Tuned Random Fraction Required
Because accidental (time-uncorrelated) coincidences naturally distribute into the exact same topological structures (FEP–Random into ridges, Compton–Random into continuum, Random–Random into the local plane), **no arbitrary user-tuned random fraction is required**.

The 2D fitting engine extracts all parameters simultaneously via Levenberg-Marquardt optimization with Poisson statistical weights ($w_{ij} = 1 / \sqrt{\max(1, M_{ij})}$), and calculates the discrete Gamba net area:

$$n^t_{p|p} = n^m_{p|p} - n^m_{p|bg} - n^m_{bg|p} + n^m_{bg|bg}$$

and the Peak-to-Total-Background ratio:

$$\Pi = \frac{n^t_{p|p}}{n^m_{p|p}}$$

---

## 4. 1D Multi-Gate Slicing & Normalized Background Subtraction (`xtrackn W/X/Z`)

Following the classic [GASPware](https://github.com/csteke/GASPware) `xtrackn` gate methodology:

- **Peak Gate Windows ($W$)**: Arbitrary consecutive peak windows $\{ [W_{k,\min}, W_{k,\max}] \}_{k=1}^K$ defined using `W`. Total peak width:
  $$\Delta W_{\text{tot}} = \sum_{k=1}^K (W_{k,\max} - W_{k,\min} + 1)$$
- **Background Gate Windows ($X$)**: Arbitrary consecutive background windows $\{ [X_{m,\min}, X_{m,\max}] \}_{m=1}^M$ defined using `X`. Total background width:
  $$\Delta X_{\text{tot}} = \sum_{m=1}^M (X_{m,\max} - X_{m,\min} + 1)$$
- **Coincidence Slicing**: Slicing matrix $M$ along the gated axis produces 1D projection vectors:
  $$S_{W_k}(i) = \sum_{j = W_{k,\min}}^{W_{k,\max}} M_{i, j}, \quad S_{X_m}(i) = \sum_{j = X_{m,\min}}^{X_{m,\max}} M_{i, j}$$
- **Normalized Background Subtraction**: When background regions are present ($\Delta X_{\text{tot}} > 0$), the net coincidence spectrum $S_{\text{net}}$ is:
  $$\text{Scale} = \frac{\Delta W_{\text{tot}}}{\Delta X_{\text{tot}}}$$
  $$S_{\text{net}}(i) = \sum_{k=1}^K S_{W_k}(i) - \text{Scale} \cdot \sum_{m=1}^M S_{X_m}(i)$$
- **Visual Overlays & Negative Baseline**:
  - Gated peak windows render in translucent green (`#22c55e30`), background in amber (`#f59e0b30`).
  - The 2D matrix renders matching vertical/horizontal band overlays across gated channels.
  - A dashed zero-baseline is drawn on the gated 1D projection when over-subtracted continuum channels fluctuate below zero counts.

---

## 5. Continuous Wavelet Transform (CWT) Photopeak Search

Automated peak finding in $\gamma$-ray spectra is challenged by Poisson statistical noise, varying peak widths, and steep Compton scattering continuum baselines. `python-cmat` implements Continuous Wavelet Transform (CWT) multi-scale peak detection (*Du et al., 2006*):

### Ricker Wavelet (Mexican Hat)
$$\psi(t) = \frac{2}{\sqrt{3\sigma}\pi^{1/4}} \left(1 - \left(\frac{t}{\sigma}\right)^2\right) \exp\left(-\frac{t^2}{2\sigma^2}\right)$$

Because $\int_{-\infty}^{\infty} \psi(t) dt = 0$, convolving a spectrum with $\psi(t)$ at characteristic scale $s \approx \text{FWHM} / 2.355$ effectively subtracts polynomial continuum baselines while transforming photopeaks into pronounced local maxima in wavelet scale space.

### Wavelet Ridge Lines & Prominence
Wavelet coefficients $C(s, \tau) = \frac{1}{\sqrt{s}} \int x(t) \psi\left(\frac{t-\tau}{s}\right) dt$ are evaluated across multiple scales surrounding the expected peak FWHM. Photopeaks create continuous ridge lines across adjacent scales with high signal-to-noise ratio ($SNR = C(s, \tau) / \sigma_{\text{noise}}$), cleanly separating real doublets and weak transitions from random channel-to-channel Poisson fluctuations.

---

## 6. Automated Multi-Peak Fitting with Peak-Aware Background Estimation

Pressing **`H`** (or clicking `Fit All [H]`) executes automated multi-peak fitting across all visible candidate photopeaks on display. The continuum baseline is estimated using a **Peak-Aware Continuum Average** engine designed specifically to handle both high-statistics projections and low-statistics / coincidence-gated spectra without noise-clipping bias:

### The Low-Statistics Under-Fit Problem & Solution
Standard morphological baseline filters like plain SNIP rely on concave lower-envelope clipping ($v_i = \min(v_i, \frac{v_{i-p} + v_{i+p}}{2})$). In low-statistics spectra or background-subtracted coincidence cuts ($S_{\text{net}} = S_{\text{gross}} - k S_{\text{bg}}$), large noise grass fluctuations cause plain SNIP to clip down to the lower noise envelope ($\approx \bar{y} - 2\sigma$), severely under-fitting the continuum and creating artificial false peaks.

Because the peak search routine (`P`) has already identified candidate photopeaks, channels outside the candidate peak clusters are known *a priori* to be pure continuum:

1. **Peak Exclusion Masking**:
   $$\text{ROI}_{\text{cl}} = \left[ \min(\text{cluster}) - 2.0 \times \text{FWHM}, \ \max(\text{cluster}) + 2.0 \times \text{FWHM} \right]$$
2. **Continuum Rolling Average**: Across pure background channels ($\mathbb{E}[\epsilon_i] = 0$), the continuum is smoothed using a rolling Bartlett filter with window $W \approx 5.0 \times \text{FWHM}$. This estimates the central expectation value ($\bar{y}$) directly through the center of the noise grass ($< 0.1$ count bias), with full tolerance for negative count bins.
3. **Cluster Baseline Bridging**: Across each peak cluster, the baseline connects the smoothed left continuum to the smoothed right continuum linearly without clipping or artificial bowing.
4. **Dense Forest Fallback**: If peak coverage exceeds 92% of the display window, the engine automatically falls back to SNIP.

### Multiplet Clustering & Adaptive ROIs
Adjacent peaks separated by $\le 3.2 \times \text{FWHM}$ are grouped into multiplets. Each cluster is fitted simultaneously with Levenberg-Marquardt optimization using variance weights $w_i = 1 / \sqrt{\max(1.0, |y_i|)}$, supporting Gaussian, RadWare / SAMPO, or Hypermet profiles. Centroids are constrained within $\pm 2.5$ channels of markers to prevent line migration.

### Manual Peak Marking (`G`) & Multiplet Deconvolution (`V`)
- **`G`**: Snaps to the local peak summit within $\pm 2$ channels and computes parabolic sub-channel centroid refinement ($\delta = \frac{1}{2} \frac{y_0 - y_2}{y_0 - 2y_1 + y_2}$). Hovering an existing marker within 1.2 channels deletes it.
- **`V`**: Pressing `V` anywhere within a multiplet cluster automatically fits all components simultaneously with full parameter covariance error propagation.

### Region-Constrained Fitting with Straight-Line Background (`R`)
Pressing `R` twice sets left and right region limits $[R_{\text{left}}, R_{\text{right}}]$. The continuum level is calculated by averaging 1 FWHM on either side:
$$\bar{y}_{\text{left}} = \frac{1}{w} \sum_{c = R_{\text{left}} - w}^{R_{\text{left}} - 1} y[c], \quad \bar{y}_{\text{right}} = \frac{1}{w} \sum_{c = R_{\text{right}} + 1}^{R_{\text{right}} + w} y[c] \quad (w = \text{round}(\text{FWHM}))$$
Pressing `V` inside the region fits all peaks within the boundary on top of this straight-line background.

---

## 7. Fit Results File Logging

To streamline spectroscopic reporting and automate data export for external calibration programs, `python-cmat` supports real-time appending of all 1D and 2D peak fit results to a clean, fixed-width text log.

### Fixed-Width Column Formatting
To ensure clean vertical alignment across text editors and avoid tab misalignment for large peak counts (10+ digits with uncertainties), columns are right-aligned with fixed character widths:

- **1D Fits**:
  ```text
  # 1D Fits:     Energy(err)               Net_Area(err)               FWHM(err)          Chi2    Peak_to_BG
             1164.257(0.003)             339769.5(735.4)            3.060(0.006)        144.37          5.84
               1164.76(0.00)             339769.5(735.4)              3.06(0.01)        144.37          5.84
  ```
  *(Energy and FWHM display in keV when calibrated or channels when uncalibrated).*

- **2D Coincidence Fits (Gamba Decomposition)**:
  ```text
  # 2D Fits:    Energy1(err)                Energy2(err)               Net_Area(err)             Gamba_Area(err)              FWHM1(err)              FWHM2(err)          Chi2    Peak_to_BG
             1164.215(0.052)            1345.264(99.304)                32.8(9123.4)                  25.6(16.1)            3.002(0.110)          0.636(119.698)          1.23          0.01
               1164.71(0.05)              1345.76(99.30)                32.8(9123.4)                  25.6(16.1)              3.00(0.11)            0.64(119.70)          1.23          0.01
  ```

### Enabling & Controls
- **Web Viewer GUI**: Click **`ON / OFF`** toggle in the sidebar *Peak Fitting* control group or set a custom filename.
- **CLI Startup Flag**: `python3 cmat_webviewer.py --fit-log [filename] matrix.cmat`
- **Headless Shell / Macro**: `fit_log on [filename]`, `fit_log off`, `fit_log status`
- **Default Filename**: If no filename is specified, an automatic timestamped file (`fit_results_YYYYMMDD_HHMMSS.txt`) is generated to prevent overwriting previous analyses.

### Automated Isotope Identification
Fit results logs can be automatically analyzed and matched against evaluated nuclear structure data to identify parent isotopes and 2D cascades using the [ENSDF Isotope Identification](ENSDF-Isotope-Identification) tool.

