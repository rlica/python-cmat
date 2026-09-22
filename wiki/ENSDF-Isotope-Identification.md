# ENSDF Automated Nuclear Isotope Identification

`python-cmat` includes a dedicated, fully offline nuclear structure search and automated isotope identification engine powered by the complete Evaluated Nuclear Structure Data File (ENSDF) database.

The system connects the experimental photopeak and coincidence fit results generated during 1D and 2D spectroscopy to known nuclear level schemes, decay modes, and transition cascades across all 300+ mass chains ($A = 1 \dots 295$).

---

## Key Features

1. **100% Offline Local SQLite Database (`ensdf.db`)**:
   - Indexed SQLite database holding over 3,400 nuclides, 18,000 datasets, 175,000 nuclear levels, and 330,000 evaluated $\gamma$-ray transitions.
   - Sub-millisecond indexed lookups by energy range, mass number $A$, element $Z$, dataset type (Decay, Adopted, Reaction), and parent half-life $T_{1/2}$.

2. **Physical 2D Coincidence Cascade Matching**:
   - Classifies 2D $\gamma$-$\gamma$ coincidence pairs into physical topologies:
     - **Direct Prompt Cascades** ($L_A \xrightarrow{\gamma_1} L_B \xrightarrow{\gamma_2} L_C$): Receives the highest confidence weight ($10\times$ boost).
     - **Sequential Cascades**: Multi-step cascades separated by intermediate transitions, scaled inversely with intermediate energy gap $\Delta E_{\text{gap}}$.
     - **High Excitation Energy Damping**: Penalizes coincidences originating from high excitation/resonance states ($E_{\text{exc}} \gg 3000\text{ keV}$).

3. **Global Parsimony & Mass-Clustering Optimization**:
   - **2D Anchoring**: High-confidence 2D coincidences establish dominant reaction/decay products and calculate the dominant mass center $\bar{A}$.
   - **Mass Proximity Constraint**: Prefers candidates with mass close to the dominant experiment cluster ($\Delta A \le 6-8$ mass units), damping distant isolated masses.
   - **Parsimonious Minimal Isotope Set Cover**: Solves for the smallest set of distinct #1 isotopes ($S = \{^{51}\text{Cr}, ^{52}\text{Cr}, ^{48}\text{Ti}, ^{55}\text{V}\}$) that collectively explains all 1D photopeaks and 2D coincidences.

4. **Standalone Web Pop-up & Server File Browser**:
   - Opens as an independent pop-up tool from both the 2D (`cmat_webviewer`) and 3D (`cmat3d_webviewer`) web interfaces.
   - Built-in interactive server filesystem browser (`📂 Browse...`) with breadcrumbs, directory navigation, and fit log search.
   - Expandable **Top 5 Candidate Isotopes** breakdown for each 2D coincidence fit.

---

## Physical Formulation & Ranking Engine

### 1. Energy Fit Confidence ($P_{\Delta E}$)
The Gaussian probability of energy agreement given user tolerance $\sigma$:
$$P_{\Delta E} = \exp\left(-\frac{1}{2}\left[\left(\frac{\Delta E_1}{\sigma_1}\right)^2 + \left(\frac{\Delta E_2}{\sigma_2}\right)^2\right]\right)$$

### 2. Cascade Topology Weight ($W_{\text{topo}}$)
- **Direct Cascade** ($|E(f_1) - E(i_2)| < 2.5\text{ keV}$ or $|E(f_2) - E(i_1)| < 2.5\text{ keV}$):
  $$W_{\text{topo}} = 10.0, \quad \Delta E_{\text{gap}} = 0\text{ keV}$$
- **Sequential Cascade** ($E(f_1) > E(i_2)$ or $E(f_2) > E(i_1)$):
  $$W_{\text{topo}} = \frac{4.0}{1.0 + \Delta E_{\text{gap}} / 600\text{ keV}}$$
- **Branching / Common Feeding**:
  $$W_{\text{topo}} = \frac{1.2}{1.0 + \Delta E_{\text{gap}} / 1200\text{ keV}}$$

### 3. Excitation Energy Penalty ($F_{\text{exc}}$)
Transitions originating from low-lying states ($< 3000\text{ keV}$) are characteristic of standard radioactive decay and low-spin spectroscopy:
$$F_{\text{exc}} = \frac{1.0}{1.0 + \left(\frac{\max(E_{i1}, E_{i2})}{3000\text{ keV}}\right)^{1.8}}$$

### 4. Mass Proximity Factor ($F_{\text{mass}}$)
Given the dominant mass center $\bar{A}$ extracted from the 2D coincidence cluster:
$$F_{\text{mass}} = \exp\left(-\frac{(A - \bar{A})^2}{2 \sigma_A^2}\right), \quad \sigma_A = 6.0\text{ mass units}$$

### 5. Parsimony (Reuse) Bonus ($F_{\text{parsimony}}$)
If a candidate isotope $N$ belongs to the actively selected minimal isotope set $S$:
$$F_{\text{parsimony}} = 5.0$$

---

## Interactive Web Pop-up Interface

The identification pop-up (`ensdf_popup.html`) can be launched directly from the top navigation bar of both `cmat_webviewer` and `cmat3d_webviewer` via the **`🔬 Isotope Identification`** button or at URL `/ensdf_popup.html`.

### Features in the Web Pop-up:
- **Server File Browser (`📂 Browse...`)**: Interactively navigate server directories, search for log files (`fit_results_*.txt`), and inspect file size/modification dates.
- **2D Coincidence Results Table**: Shows the primary #1 match with colored cascade badges (`Direct Cascade (Prompt Coincidence)`, `Sequential Cascade`, `Same Level Scheme`), excitation levels, and energy differences.
- **Expandable Top 5 Candidates**: Clicking **`▼ Top N Candidates`** on any row expands an inner sub-table detailing the top 5 candidate isotopes with confidence scores.
- **1D Photopeaks Table**: Displays net area, assigned candidate isotope, parent $T_{1/2}$, dataset source, ENSDF transition energy, and energy difference.
- **Constraint Filters**:
  - Mass range: $A_{\text{min}} \le A \le A_{\text{max}}$
  - Specific elements: e.g. `Cr Ti V Fe Ni`
  - Parent half-life: Minimum and maximum parent $T_{1/2}$ (e.g. `1m`, `1h`, `1d`, `1y`)
  - Dataset type: `All`, `Decay Schemes Only`, `Adopted Levels Only`, `Reactions / In-Beam`
  - Energy search tolerance: $\pm\text{keV}$ (default: $1.5\text{ keV}$)
- **Export**: Export identified 1D and 2D results as clean tab-separated text files.

---

## Command-Line Interface (`ensdf_search.py`)

`ensdf_search.py` can be used directly from the command line for automated scripting and terminal inspection.

### 1. Analyze a Fit Results Log File
```bash
# Automatically identify all 1D photopeaks and 2D coincidences in a log file
python3 ensdf_search.py fit_results_20260922_155405.txt

# Specify energy tolerance and list top 5 candidate isotopes
python3 ensdf_search.py fit_results_20260922_155405.txt --tol 1.5 --top 5

# Apply mass and half-life constraints
python3 ensdf_search.py fit_results_20260922_155405.txt --a-min 48 --a-max 56 --min-t12 1m
```

### 2. Manual 2D Coincidence Search
```bash
# Search for coincidence pair E1=1434.2 keV, E2=935.3 keV
python3 ensdf_search.py -c 1434.2 935.3 --tol 1.5 --top 5
```

### 3. Manual 1D Photopeak Search
```bash
# Search for single photopeak E=1480.3 keV
python3 ensdf_search.py -g 1480.3 --tol 1.5
```

### 4. Database Status & Statistics
```bash
python3 ensdf_search.py --status
```

### 5. (Re)building the ENSDF SQLite Database
To build or update the local database from a raw ENSDF mass directory (e.g. `ensdf_260901/`):
```bash
python3 ensdf_search.py --build-db /path/to/ensdf_directory/
```

---

## Python API Usage

You can also use the `ENSDFSearchEngine` directly in Python scripts or Jupyter notebooks:

```python
from ensdf_search import ENSDFSearchEngine

engine = ENSDFSearchEngine()

# 1. Search 2D coincidence pair
cands_2d = engine.search_2d(
    energy1=1434.15,
    energy2=935.26,
    tol1=1.5,
    tol2=1.5,
    limit=5,
    unique_nuclides=True
)
for c in cands_2d:
    print(f"{c['nuclide']}: Score={c['score']}, {c['cascade_type']}, Levels: {c['level1_init']}->{c['level1_final']}")

# 2. Identify entire fit results log file with global parsimony
report = engine.identify_fit_results_file("fit_results_20260922_155405.txt", tol=1.5)
print(f"Dominant Mass: A ≈ {report['dominant_mass']}")
print(f"Minimal Isotope Set: {report['parsimonious_isotopes']}")
```
