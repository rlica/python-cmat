# Format Reference & Decompression

[GASPware](https://github.com/csteke/GASPware) and `gsort` 2D matrices in the `.cmat` format are stored in an IVF (Indexed Variable Format) file container consisting of 512-byte binary records. `python-cmat` implements a pure Python + NumPy reverse-engineered reader that parses and decompresses all IVF records without external compiled dependencies.

While the format is widely known in nuclear structure experiments for storing 2D $\gamma$-$\gamma$ coincidence matrices, the container and compression routines are general to any 2D matrix (symmetric, asymmetric, or arbitrary block steps). Future releases will also incorporate handling of 3D matrices and time-difference spectra.

---

## IVF Container & Sub-Matrix Grid Architecture

### 1. Matrix Dimensions & Sub-Blocks
A `.cmat` matrix with resolution $res_1 \times res_2$ (channels) is partitioned into a 2D grid of sub-matrix blocks of dimension $step_1 \times step_2$:

$$ndiv_1 = \frac{res_1}{step_1}, \quad ndiv_2 = \frac{res_2}{step_2}$$

`python-cmat` supports both standard and non-standard geometries:
- **Symmetric Matrices** ($res_1 = res_2, step_1 = step_2$): E.g., $4096 \times 4096$ channels with $128 \times 128$ blocks $\rightarrow 32 \times 32$ grid. Due to symmetry ($M_{ij} = M_{ji}$), only the lower triangular portion containing $\frac{32 \times 33}{2} = 528$ blocks is physically stored.
  - **Diagonal Multiplicity**: In folded symmetric representation, diagonal elements $(E_1 == E_2)$ occupy single discrete bins while off-diagonal pairs $(E_1, E_2)$ occupy two symmetrical points. During unfolding in `CMATReader.to_numpy()`, diagonal elements are multiplied by 2 ($2 \times \text{val}$), ensuring continuous 2D background density and exact agreement with the 1D stored projection ($\sum_y M(y, x) = \text{proj}_0(x)$ with $\Delta = 0$).
- **Normal Matrices** (non-folded full matrices): E.g., $6144 \times 6144$ with $128 \times 128$ blocks $\rightarrow 2304$ stored blocks.
- **Asymmetric Matrices & Arbitrary Steps**: E.g., $2048 \times 4096$ with $32 \times 64$ sub-blocks $\rightarrow 4096$ stored blocks.

### 2. Dual Stored Projections (Extra Segments)
Standard IVF matrices record the total 1D projections directly in header segments:
- **Segment 2**: Det 1 / X total projection spectrum.
- **Segment 3**: Det 2 / Y total projection spectrum (in asymmetric matrices).

`python-cmat` reads these pre-computed projections instantly for rapid spectrum display, with automatic fallback to full 2D matrix summation if segments are absent or corrupted.

### 3. NumPy Array Coordinate Mapping
When decompressed via `CMATReader.to_numpy()`, the array is returned in standard NumPy row-major format:
- Shape: `(res2, res1)` corresponding to `(Height / Y, Width / X)`.
- Indexing: `matrix[y, x]` accesses counts at channel $x$ on Det 1 and channel $y$ on Det 2.

---

## Reverse-Engineered Decompression Algorithms

Each sub-block in the matrix is compressed independently according to its sparseness and count distribution using one of five algorithms identified by its IVF compression mode:

### 1. Modes 0–32: Fixed Bit-Width Channel Packing (`ccomp__0_decompress`)
- Used when counts within a sub-block can be represented in $b$ bits ($0 \le b \le 32$).
- Mode $b$ indicates that channels are packed into continuous bit-streams with $b$ bits per element.
- **Mode 0**: All channels in the block are strictly zero (zero storage overhead).
- **Modes 1–32**: Pure bitwise unpacking into signed 32-bit integers.

### 2. Mode 33: 16-Bit Word Sparse Channel List (`ccomp__1_decompressW`)
- Used for extremely sparse sub-blocks where only a small number of channels have non-zero counts.
- Encoded as pairs of 16-bit unsigned integers: `(channel_index, count)`.
- Decompressed by placing the 16-bit count values directly into the target sub-block buffer at their specified relative offsets.

### 3. Mode 34: 32-Bit Longword Sparse Channel List (`ccomp__1_decompressLW`)
- Similar to Mode 33, but employs 32-bit longwords for counts exceeding $2^{16}-1$.
- Encoded as `(16-bit channel_index, 32-bit count)`.

### 4. Mode 37: Variable-Length Tagged Token Bit-Stream (`ccomp__2_decompress`)
- A high-efficiency entropy coding scheme utilizing variable-length tagged tokens.
- A leading 2-bit or 4-bit prefix tag determines the byte-width and sign-extension of subsequent channel values and zero-run lengths.

### 5. Mode 41: Bit-Shift-Map (BSM) Unary Run-Length Compression (`ccomp__3_decompress`)
- Optimized for counting spectra with dense low-value clusters and sparse high-count spikes.
- Encoded with a bit-shift map bitmap followed by unary run-length deltas.

---

## 3D `.cmat` Cube Container Specification

3D `.cmat` matrices (e.g. $\gamma$-$\gamma$-$\text{Rings}$, $\gamma$-$\gamma$-$\Delta t$, or 3-fold symmetric $\gamma$-$\gamma$-$\gamma$ cubes) extend the IVF block architecture to 3-dimensional coordinate spaces:
- **Geometry**: Defined by dimensions $(N_x, N_y, N_z)$, such as $(4096, 4096, 128)$ or $(8192, 8192, 8192)$.
- **Storage Layout**: Concatenated sequence of 2D IVF sub-matrix planes along the 3rd axis, each containing its own independent IVF header and compressed block offset tables.
- **Tetrahedral Intra-Block Folding**: For symmetric 3D cubes (`matmode == 1`), blocks are indexed in the lower tetrahedron $s_1 \le s_2 \le s_3$. On diagonal boundary planes ($s_1=s_2<s_3$, $s_1<s_2=s_3$, $s_1=s_2=s_3$), internal sub-block channels are folded into sub-triangles and sub-tetrahedra. `CMAT3DReader` performs full 6-permutation sub-tetrahedral unfolding to guarantee complete volumetric symmetry.
- **Storage Engines**:
  - **Memory-Mapped Dense Engine (`.dat`)**: Sequentially decompresses moderate volumes to disk (`.cmat3d_cache/<file>.dat`) and memory-maps the binary array.
  - **Sparse On-Demand Engine (`.idx`)**: For massive cubes ($8192^3$), indexes block segment offsets into a tiny `.idx` file and decompresses requested sub-blocks on-the-fly in $<10\text{ ms}$.


