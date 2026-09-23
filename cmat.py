"""
cmat.py - Python Reader and Decompressor for GASPware/gsort .cmat Matrices

Reverse-engineered from GASPware source code (ivflib.F, cmtlib.F, complib.c).
Author: Antigravity Assistant & pair programming with user.
"""

import struct
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path


def _decompress_mode0(pack: bytes, nch: int, nbits: int) -> Tuple[np.ndarray, int]:
    """Mode 0..32: Fixed bit-width channel packing (ccomp__0_decompress)."""
    data = np.zeros(nch, dtype=np.int32)
    if nbits <= 0:
        return data, 0
    if nbits >= 32:
        return np.frombuffer(pack[:nch * 4], dtype="<i4").copy(), nch * 4
    if nbits == 16:
        return np.frombuffer(pack[:nch * 2], dtype="<u2").astype(np.int32), nch * 2
    if nbits == 8:
        return np.frombuffer(pack[:nch], dtype="u1").astype(np.int32), nch

    nbitsr = nbits
    wpack_offset = 0
    half = False
    if nbitsr >= 16:
        wdata = np.frombuffer(pack[:nch * 2], dtype="<u2").astype(np.int32)
        data[:] = wdata
        wpack_offset = nch * 2
        nbitsr -= 16
        half = True

    if nbitsr == 8:
        cdata = np.frombuffer(pack[wpack_offset:wpack_offset + nch], dtype="u1").astype(np.int32)
        if half:
            data[:] |= (cdata << 16)
        else:
            data[:] = cdata
        return data, wpack_offset + nch

    mask = (1 << nbitsr) - 1
    dd = 0
    npacked = 0
    ii = 0
    ptr = wpack_offset
    while ii < nch:
        wdd = struct.unpack_from("<H", pack, ptr)[0]
        ptr += 2
        dd = (dd << 16) | wdd
        npacked += 16
        while npacked >= nbitsr and ii < nch:
            val = (dd >> (npacked - nbitsr)) & mask
            if half:
                data[ii] |= (val << 16)
            else:
                data[ii] = val
            ii += 1
            npacked -= nbitsr
        dd &= (1 << npacked) - 1
    return data, ptr


def _decompress_mode1_W(pack: bytes, nch: int) -> Tuple[np.ndarray, int]:
    """Mode 33: 16-bit word-level non-zero sparse list (ccomp__1_decompressW)."""
    data = np.zeros(nch, dtype=np.int32)
    nonzero = struct.unpack_from("<h", pack, 0)[0]
    if nonzero <= 0:
        return data, 2
    offset = 2
    inddata = struct.unpack_from("<h", pack, offset)[0]
    offset += 2
    for _ in range(nonzero):
        data[inddata] += 1
        nexdata = struct.unpack_from("<h", pack, offset)[0]
        offset += 2
        if nexdata < 0:
            data[inddata] -= nexdata
            inddata = struct.unpack_from("<h", pack, offset)[0]
            offset += 2
        else:
            inddata = nexdata
    return data, offset


def _decompress_mode1_LW(pack: bytes, nch: int) -> Tuple[np.ndarray, int]:
    """Mode 34: 32-bit longword non-zero sparse list (ccomp__1_decompressLW)."""
    data = np.zeros(nch, dtype=np.int32)
    nonzero = struct.unpack_from("<i", pack, 0)[0]
    if nonzero <= 0:
        return data, 4
    offset = 4
    inddata = struct.unpack_from("<i", pack, offset)[0]
    offset += 4
    for _ in range(nonzero):
        data[inddata] += 1
        nexdata = struct.unpack_from("<i", pack, offset)[0]
        offset += 4
        if nexdata < 0:
            data[inddata] -= nexdata
            inddata = struct.unpack_from("<i", pack, offset)[0]
            offset += 4
        else:
            inddata = nexdata
    return data, offset


def _decompress_mode2(pack: bytes, isize: int) -> Tuple[np.ndarray, int]:
    """Mode 37: Variable-length tagged token compression (ccomp__2_decompress)."""
    data = np.zeros(isize, dtype=np.int32)
    cptr = 0
    lbin = pack[cptr]
    cptr += 1
    idpnt = 0

    while idpnt < isize:
        dd = pack[cptr]
        itag = dd & 3
        if itag == 0:
            icount = (dd & 0x7C) >> 2
            nbits = 0
        elif itag == 1:
            icount = (dd & 0x3C) >> 2
            nbits = 1 + ((dd & 0x7F) >> 6)
        elif itag == 2:
            icount = (dd & 0x1C) >> 2
            nbits = 3 + ((dd & 0x7F) >> 5)
        elif itag == 3:
            icount = 0
            nbits = 1 + ((dd & 0x7F) >> 2)
        else:
            icount = 0
            nbits = 0

        if (dd & 0x80) == 0:
            minval = 0
            cptr += 1
        else:
            dd2 = pack[cptr + 1]
            nextra = dd2 & 0x07
            minval = (dd2 >> 3) & 0x0F
            isign = dd2 & 0x80
            for ii in range(1, nextra):
                minval += pack[cptr + 1 + ii] << (8 * ii - 4)
            if isign:
                minval = -minval
            cptr += nextra + 1

        nch = lbin * (icount + 1)

        if nbits <= 0:
            data[idpnt:idpnt + nch] = minval
        elif nbits >= 32:
            for ii in range(nch):
                val = struct.unpack_from(">I", pack, cptr)[0]
                cptr += 4
                data[idpnt + ii] = val
        else:
            mask = (1 << nbits) - 1
            dd_buf = 0
            npacked = 0
            for ii in range(nch):
                while npacked < nbits:
                    dd_buf = (dd_buf << 8) | pack[cptr]
                    cptr += 1
                    npacked += 8
                val = ((dd_buf >> (npacked - nbits)) & mask) + minval
                data[idpnt + ii] = val
                npacked -= nbits
                dd_buf &= (1 << npacked) - 1

        idpnt += nch

    return data, cptr


def _decompress_mode3(pack: bytes, nch: int) -> Tuple[np.ndarray, int]:
    """Mode 41: Bit-shift-map unary run-length encoding (ccomp__3_decompress)."""
    data = np.zeros(nch, dtype=np.int32)
    cptr = 0
    ii = 0
    ll = 0
    while ii < nch:
        dd = pack[cptr]
        cptr += 1
        for bit in range(7, -1, -1):
            if (dd >> bit) & 1:
                ll += 1
            else:
                data[ii] = ll
                ii += 1
                if ii >= nch:
                    break
                ll = 0
    return data, cptr


def decompress_block(pack: bytes, nch: int, mode: int, minval: int = 0) -> np.ndarray:
    """
    Decompress a block of channels given the compression mode and minval offset.
    """
    if 0 <= mode <= 32:
        data, _ = _decompress_mode0(pack, nch, mode)
    elif mode == 33:
        data, _ = _decompress_mode1_W(pack, nch)
    elif mode == 34:
        data, _ = _decompress_mode1_LW(pack, nch)
    elif mode == 37:
        data, _ = _decompress_mode2(pack, nch)
    elif mode == 41:
        data, _ = _decompress_mode3(pack, nch)
    else:
        raise ValueError(f"Unknown or unsupported compression mode: {mode}")

    if minval != 0:
        data += minval
    return data


class CMATReader:
    """
    Reader and decompressor for GASPware/gsort compressed 2D matrices (.cmat).
    """

    def __init__(self, filename: Union[str, Path]):
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"File not found: {self.filename}")

        self._read_headers()

    def _read_headers(self):
        with open(self.filename, "rb") as f:
            # Read IVF Header (512 bytes = 128 int32)
            ivf_bytes = f.read(512)
            if len(ivf_bytes) < 512:
                raise ValueError("Corrupted file: IVF header too small.")
            self.ivf_hdr = struct.unpack("<128i", ivf_bytes)

            self.ivf_version = self.ivf_hdr[0]
            self.nsegtot = self.ivf_hdr[1]
            self.drecbits = self.ivf_hdr[2]
            self.consistent = (self.ivf_hdr[4] == 1)
            self.ndescr = self.ivf_hdr[10]
            self.fdescr = self.ivf_hdr[11]

            # Read IVF Descriptors
            f.seek((self.fdescr - 1) * 512)
            descr_bytes = f.read(self.ndescr * 512)
            self.descrs = [
                struct.unpack_from("<2i", descr_bytes, i * 8)
                for i in range(self.nsegtot)
            ]

            # Read CMT Header from segment 0
            cmt_nrec, cmt_frec = self.descrs[0]
            f.seek((cmt_frec - 1) * 512)
            cmt_bytes = f.read(cmt_nrec * 512)
            self.cmt_hdr = struct.unpack("<128i", cmt_bytes[:512])

            self.ndim = self.cmt_hdr[0]
            self.matmode = self.cmt_hdr[1]  # 0=normal, 1=symmetric, 2=half-symmetric
            self.res1 = self.cmt_hdr[3]
            self.step1 = self.cmt_hdr[4]
            self.ndiv1 = self.cmt_hdr[5]
            self.res2 = self.cmt_hdr[6]
            self.step2 = self.cmt_hdr[7]
            self.ndiv2 = self.cmt_hdr[8]

            # Validate dimensions and block sizes against division counts
            if self.step1 > 0 and self.ndiv1 > 0:
                self.res1 = max(self.res1, self.step1 * self.ndiv1)
            if self.step2 > 0 and self.ndiv2 > 0:
                self.res2 = max(self.res2, self.step2 * self.ndiv2)

            expected_segsize = self.step1 * self.step2
            raw_segsize = self.cmt_hdr[123]
            self.segsize = raw_segsize if (raw_segsize == expected_segsize and expected_segsize > 0) else expected_segsize
            self.nmatrix_segs = self.cmt_hdr[124]
            self.nextra = self.cmt_hdr[125]
            self.cmt_version = self.cmt_hdr[127]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.res1, self.res2)

    @property
    def is_symmetric(self) -> bool:
        return self.matmode == 1

    def get_info(self) -> Dict[str, Any]:
        return {
            "filename": str(self.filename),
            "dimensions": self.ndim,
            "shape": (self.res1, self.res2),
            "shape_yx": (self.res2, self.res1),
            "step": (self.step1, self.step2),
            "blocks": (self.ndiv1, self.ndiv2),
            "matrix_segments": self.nmatrix_segs,
            "extra_segments": self.nextra,
            "total_segments": self.nsegtot,
            "segment_size": self.segsize,
            "matrix_mode": "Symmetric" if self.matmode == 1 else ("Half-Symmetric" if self.matmode == 2 else "Normal"),
            "ivf_version": self.ivf_version,
            "cmt_version": self.cmt_version,
        }

    def _read_raw_segment(self, f, seg_idx: int) -> Optional[Tuple[int, int, bytes]]:
        if seg_idx < 0 or seg_idx >= len(self.descrs):
            return None
        nrec, frec = self.descrs[seg_idx]
        if nrec == 0:
            return 0, 0, b""
        f.seek((frec - 1) * 512)
        raw = f.read(nrec * 512)
        cmode, cminval = struct.unpack_from("<2i", raw, 0)
        return cmode, cminval, raw[8:]

    def get_projection(self, axis: int = 0) -> np.ndarray:
        """
        Get the 1D total projection spectrum.
        axis: 0 for Detector 1 / X projection (length res1);
              1 for Detector 2 / Y projection (length res2).
        """
        if axis not in (0, 1):
            axis = 0

        target_res = self.res1 if axis == 0 else self.res2
        proj_seg_idx = 2 if (axis == 0 or self.is_symmetric) else 3

        with open(self.filename, "rb") as f:
            seg_data = self._read_raw_segment(f, proj_seg_idx)
            if seg_data is not None and seg_data[0] != 0 and len(seg_data[2]) > 0:
                cmode, cminval, pack = seg_data
                decomp = decompress_block(pack, target_res, cmode, cminval)
                if len(decomp) == target_res:
                    return decomp
                elif len(decomp) < target_res:
                    return np.pad(decomp, (0, target_res - len(decomp)))
                else:
                    return decomp[:target_res]

        # If no stored projection or segment empty, compute from full matrix
        mat = self.to_numpy()
        if axis == 0:
            return np.sum(mat, axis=0)
        else:
            return np.sum(mat, axis=1)

    def to_numpy(self) -> np.ndarray:
        """
        Decompress and assemble the entire 2D matrix into a NumPy array.
        Returns:
            np.ndarray of shape (res2, res1) with dtype int32.
            Axis 0 (rows) corresponds to Detector 2 / Y (length res2).
            Axis 1 (columns) corresponds to Detector 1 / X (length res1).
        """
        dim_y = max(self.res2, self.ndiv2 * self.step2)
        dim_x = max(self.res1, self.ndiv1 * self.step1)
        expected_segsize = self.step1 * self.step2

        with open(self.filename, "rb") as f:
            if self.matmode == 1:  # Symmetrized 2D
                dim = max(dim_y, dim_x)
                mat = np.zeros((dim, dim), dtype=np.int32)
                for s2 in range(self.ndiv2):
                    for s1 in range(s2 + 1):
                        iseg = s1 + (s2 * (s2 + 1)) // 2
                        seg_idx = iseg + self.nextra
                        cmode, cminval, pack = self._read_raw_segment(f, seg_idx)
                        if cmode == 0 and len(pack) == 0:
                            continue

                        block = decompress_block(pack, expected_segsize, cmode, cminval)
                        if len(block) != expected_segsize:
                            if len(block) < expected_segsize:
                                block = np.pad(block, (0, expected_segsize - len(block)))
                            else:
                                block = block[:expected_segsize]

                        block_2d = block.reshape((self.step2, self.step1))
                        x0 = s1 * self.step1
                        y0 = s2 * self.step2

                        if s1 == s2:
                            # Diagonal block: lower triangular (ki1 <= ki2)
                            limit_k2 = min(self.step2, dim - y0)
                            limit_k1 = min(self.step1, dim - x0)
                            for ki2 in range(limit_k2):
                                for ki1 in range(min(ki2 + 1, limit_k1)):
                                    val = block_2d[ki2, ki1]
                                    if ki1 == ki2:
                                        # Diagonal events (E1 == E2) have multiplicity 2 in symmetric 2D
                                        # coincidence representation, matching surrounding 2D background density
                                        # and satisfying sum(mat, axis=0) == proj0 exactly.
                                        mat[y0 + ki2, x0 + ki1] = 2 * val
                                    else:
                                        mat[y0 + ki2, x0 + ki1] = val
                                        mat[x0 + ki1, y0 + ki2] = val
                        else:
                            # Off-diagonal block
                            y1 = min(y0 + self.step2, dim)
                            x1 = min(x0 + self.step1, dim)
                            blk_slice = block_2d[:y1 - y0, :x1 - x0]
                            mat[y0:y1, x0:x1] = blk_slice
                            mat[x0:x1, y0:y1] = blk_slice.T

                if self.res2 > 0 and self.res1 > 0 and (mat.shape[0] != self.res2 or mat.shape[1] != self.res1):
                    mat = mat[:self.res2, :self.res1]

            elif self.matmode == 0:  # Normal (Non-symmetric) 2D
                mat = np.zeros((dim_y, dim_x), dtype=np.int32)
                for s2 in range(self.ndiv2):
                    for s1 in range(self.ndiv1):
                        iseg = s1 + self.ndiv1 * s2
                        seg_idx = iseg + self.nextra
                        cmode, cminval, pack = self._read_raw_segment(f, seg_idx)
                        if cmode == 0 and len(pack) == 0:
                            continue

                        block = decompress_block(pack, expected_segsize, cmode, cminval)
                        if len(block) != expected_segsize:
                            if len(block) < expected_segsize:
                                block = np.pad(block, (0, expected_segsize - len(block)))
                            else:
                                block = block[:expected_segsize]

                        block_2d = block.reshape((self.step2, self.step1))
                        x0 = s1 * self.step1
                        y0 = s2 * self.step2
                        y1 = min(y0 + self.step2, dim_y)
                        x1 = min(x0 + self.step1, dim_x)
                        mat[y0:y1, x0:x1] = block_2d[:y1 - y0, :x1 - x0]

                if self.res2 > 0 and self.res1 > 0 and (mat.shape[0] != self.res2 or mat.shape[1] != self.res1):
                    mat = mat[:self.res2, :self.res1]

            else:
                raise NotImplementedError(f"Matrix mode {self.matmode} is not yet implemented.")

        return mat

    def get_slice(self, channel: int, axis: int = 0) -> np.ndarray:
        """
        Extract a 1D slice at a single channel.
        axis=0: slice along Y at fixed X=channel.
        axis=1: slice along X at fixed Y=channel.
        """
        mat = self.to_numpy()
        if axis == 0:
            return mat[:, channel]
        else:
            return mat[channel, :]

    def get_gate(
        self,
        gate_min: int,
        gate_max: int,
        axis: int = 0,
        bg_gates: Optional[list] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Compute coincidence 1D spectrum for a gate on the given axis.

        Args:
            gate_min: Lower channel of peak gate (inclusive).
            gate_max: Upper channel of peak gate (inclusive).
            axis: Axis to gate on (0 for X/Det1, 1 for Y/Det2).
            bg_gates: Optional list of (bg_min, bg_max) tuples for background subtraction.

        Returns:
            Tuple of (net_spectrum, bg_spectrum, raw_spectrum)
        """
        mat = self.to_numpy()
        max_ch = (self.res1 - 1) if axis == 0 else (self.res2 - 1)
        g_min = max(0, min(gate_min, gate_max))
        g_max = min(max_ch, max(gate_min, gate_max))

        if axis == 0:
            raw_gate = np.sum(mat[:, g_min:g_max + 1], axis=1)
        else:
            raw_gate = np.sum(mat[g_min:g_max + 1, :], axis=0)

        bg_spectrum = None
        if bg_gates and len(bg_gates) > 0:
            total_bg_ch = 0
            raw_bg = np.zeros_like(raw_gate, dtype=np.float64)
            for b_min, b_max in bg_gates:
                bm = max(0, min(b_min, b_max))
                bx = min(max_ch, max(b_min, b_max))
                n_ch = bx - bm + 1
                total_bg_ch += n_ch
                if axis == 0:
                    raw_bg += np.sum(mat[:, bm:bx + 1], axis=1)
                else:
                    raw_bg += np.sum(mat[bm:bx + 1, :], axis=0)

            gate_width = g_max - g_min + 1
            scale = gate_width / total_bg_ch if total_bg_ch > 0 else 1.0
            bg_spectrum = raw_bg * scale
            net_spectrum = raw_gate.astype(np.float64) - bg_spectrum
        else:
            net_spectrum = raw_gate.astype(np.float64)

        return net_spectrum, bg_spectrum, raw_gate

    def get_banana_roi(
        self,
        polygon_peak: list,
        polygon_bg: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Compute 2D Banana ROI area integration and area-normalized background subtraction.

        Args:
            polygon_peak: List of (x, y) or [x, y] coordinates defining peak polygon ROI.
            polygon_bg: Optional list of (x, y) coordinates defining background polygon ROI.

        Returns:
            Dict containing pixel_count_peak, pixel_count_bg, area_peak, area_bg,
            counts_peak, counts_bg, scale, net_counts, and net_err.
        """
        mat = self.to_numpy()
        from cmat_webviewer import compute_2d_banana_roi
        return compute_2d_banana_roi(mat, polygon_peak=polygon_peak, polygon_bg=polygon_bg)

    def export_amat(
        self,
        output_file: Union[str, Path],
        format_type: str = "dense",
        delimiter: str = " ",
        header: bool = True,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
    ):
        """
        Export matrix to ASCII format (.amat).

        Args:
            output_file: Path to destination file.
            format_type: 'dense' (2D grid rows) or 'sparse' (x y counts).
            delimiter: Separator between numbers (default ' ').
            header: If True, writes a metadata header line.
            x_range: Optional (xmin, xmax) channel range (inclusive).
            y_range: Optional (ymin, ymax) channel range (inclusive).
        """
        output_path = Path(output_file)
        mat = self.to_numpy()

        xmin = 0 if x_range is None else max(0, x_range[0])
        xmax = self.res1 - 1 if x_range is None else min(self.res1 - 1, x_range[1])
        ymin = 0 if y_range is None else max(0, y_range[0])
        ymax = self.res2 - 1 if y_range is None else min(self.res2 - 1, y_range[1])

        sub_mat = mat[ymin:ymax + 1, xmin:xmax + 1]

        with open(output_path, "w", encoding="utf-8") as out:
            if header:
                out.write(f"# CMAT to AMAT ASCII Export\n")
                out.write(f"# Source: {self.filename.name}\n")
                out.write(f"# Full Dimensions: {self.res1} x {self.res2}\n")
                out.write(f"# Export Range: X=[{xmin}, {xmax}], Y=[{ymin}, {ymax}]\n")
                out.write(f"# Sub-Matrix Shape: {sub_mat.shape[1]} (X) x {sub_mat.shape[0]} (Y)\n")
                out.write(f"# Format: {format_type}\n")

            if format_type.lower() == "sparse":
                # Export only non-zero bins
                y_indices, x_indices = np.nonzero(sub_mat)
                for yi, xi in zip(y_indices, x_indices):
                    val = sub_mat[yi, xi]
                    out.write(f"{xmin + xi}{delimiter}{ymin + yi}{delimiter}{val}\n")
            elif format_type.lower() == "dense":
                # Export row by row
                for y in range(sub_mat.shape[0]):
                    row_str = delimiter.join(map(str, sub_mat[y, :]))
                    out.write(row_str + "\n")
            else:
                raise ValueError(f"Unknown format_type: {format_type}. Use 'dense' or 'sparse'.")


def write_cmat(
    output_file: Union[str, Path],
    matrix: np.ndarray,
    symmetric: Optional[bool] = None,
    step1: int = 128,
    step2: int = 128,
) -> None:
    """
    Compress and write a 2D NumPy array to a GASPware/gsort compliant .cmat file.

    Args:
        output_file: Destination file path (.cmat).
        matrix: 2D NumPy array with shape (res2, res1) where axis 0 is Y/Det2 and axis 1 is X/Det1.
        symmetric: True for symmetric matrix (mode 1), False for normal (mode 0),
                   None to auto-detect if shape is square and matrix == matrix.T.
        step1: Sub-block width on X / Det 1 axis (default: 128).
        step2: Sub-block height on Y / Det 2 axis (default: 128).
    """
    mat = np.asarray(matrix, dtype=np.int32)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {mat.shape}")

    res2, res1 = mat.shape
    if symmetric is None:
        symmetric = (res1 == res2 and np.array_equal(mat, mat.T))

    matmode = 1 if symmetric else 0

    # Calculate grid divisions and pad if dimensions are not exact multiples of step
    ndiv1 = (res1 + step1 - 1) // step1
    ndiv2 = (res2 + step2 - 1) // step2
    res1_padded = ndiv1 * step1
    res2_padded = ndiv2 * step2

    if res1_padded != res1 or res2_padded != res2:
        pad_y = res2_padded - res2
        pad_x = res1_padded - res1
        mat = np.pad(mat, ((0, pad_y), (0, pad_x)), mode="constant")
        res2, res1 = mat.shape

    segsize = step1 * step2
    nextra = 5

    if symmetric:
        nmatrix_segs = (ndiv2 * (ndiv2 + 1)) // 2
    else:
        nmatrix_segs = ndiv1 * ndiv2

    nsegtot = nmatrix_segs + nextra

    # Calculate descriptors count
    # Each descriptor is 2 int32 (8 bytes)
    descr_bytes_len = nsegtot * 8
    ndescr = (descr_bytes_len + 511) // 512
    fdescr = 2  # record 2

    current_record = fdescr + ndescr

    descriptors = [(0, 0)] * nsegtot
    segment_data_blocks: Dict[int, bytes] = {}

    def compress_block(block_flat: np.ndarray) -> Tuple[int, int, bytes]:
        max_val = int(np.max(block_flat))
        min_val = int(np.min(block_flat))
        if max_val == 0 and min_val == 0:
            return 0, 0, b""

        # Check sparse mode 33 (if very sparse and within 16-bit word limits)
        nz = np.flatnonzero(block_flat)
        if len(nz) < len(block_flat) // 8 and max_val < 32767 and len(block_flat) <= 32768:
            nz_count = len(nz)
            out = bytearray()
            out.extend(struct.pack("<h", nz_count))
            out.extend(struct.pack("<h", int(nz[0])))
            for i, idx in enumerate(nz):
                cnt = int(block_flat[idx])
                if cnt > 1:
                    out.extend(struct.pack("<h", -(cnt - 1)))
                    if i + 1 < nz_count:
                        out.extend(struct.pack("<h", int(nz[i + 1])))
                    else:
                        out.extend(struct.pack("<h", 0))
                else:
                    if i + 1 < nz_count:
                        out.extend(struct.pack("<h", int(nz[i + 1])))
                    else:
                        out.extend(struct.pack("<h", 0))
            return 33, 0, bytes(out)

        if min_val >= 0 and max_val <= 255:
            return 8, 0, block_flat.astype(np.uint8).tobytes()
        elif min_val >= 0 and max_val <= 65535:
            return 16, 0, block_flat.astype("<u2").tobytes()
        else:
            return 32, 0, block_flat.astype("<i4").tobytes()

    def add_raw_segment(seg_idx: int, data_bytes: bytes):
        nonlocal current_record
        if len(data_bytes) == 0:
            descriptors[seg_idx] = (0, 0)
            return
        nrec = (len(data_bytes) + 511) // 512
        padded = data_bytes + b"\x00" * (nrec * 512 - len(data_bytes))
        frec = current_record
        descriptors[seg_idx] = (nrec, frec)
        segment_data_blocks[seg_idx] = padded
        current_record += nrec

    def add_compressed_segment(seg_idx: int, block_flat: np.ndarray):
        cmode, cminval, pack = compress_block(block_flat)
        if cmode == 0 and len(pack) == 0:
            descriptors[seg_idx] = (0, 0)
            return
        hdr = struct.pack("<2i", cmode, cminval)
        add_raw_segment(seg_idx, hdr + pack)

    # Segment 0: CMT Header (1 record = 512 bytes)
    cmt_hdr = [0] * 128
    cmt_hdr[0] = 2          # ndim
    cmt_hdr[1] = matmode    # matmode (0=normal, 1=symmetric)
    cmt_hdr[2] = max(res1, res2)
    cmt_hdr[3] = res1
    cmt_hdr[4] = step1
    cmt_hdr[5] = ndiv1
    cmt_hdr[6] = res2
    cmt_hdr[7] = step2
    cmt_hdr[8] = ndiv2
    cmt_hdr[123] = segsize
    cmt_hdr[124] = nmatrix_segs
    cmt_hdr[125] = nextra
    cmt_hdr[126] = nsegtot
    cmt_hdr[127] = 5        # cmt_version
    add_raw_segment(0, struct.pack("<128i", *cmt_hdr))

    # Segment 1: Reserved / Empty

    # Segment 2: Projection on axis 0 (Det 1 / X)
    proj0 = np.sum(mat, axis=0)
    add_compressed_segment(2, proj0)

    # Segment 3: Projection on axis 1 (Det 2 / Y)
    if not symmetric:
        proj1 = np.sum(mat, axis=1)
        add_compressed_segment(3, proj1)

    # Segment 4: Reserved / Empty

    # Matrix sub-blocks (Segment 5 onwards)
    if symmetric:
        for s2 in range(ndiv2):
            for s1 in range(s2 + 1):
                iseg = s1 + (s2 * (s2 + 1)) // 2
                seg_idx = iseg + nextra
                x0 = s1 * step1
                y0 = s2 * step2
                block_2d = mat[y0:y0 + step2, x0:x0 + step1].copy()
                if s1 == s2:
                    # In folded format, zero out upper triangle and store diagonal // 2
                    block_2d = np.tril(block_2d)
                    diag = np.diag(block_2d)
                    np.fill_diagonal(block_2d, (diag + 1) // 2)
                add_compressed_segment(seg_idx, block_2d.flatten())
    else:
        for s2 in range(ndiv2):
            for s1 in range(ndiv1):
                iseg = s1 + ndiv1 * s2
                seg_idx = iseg + nextra
                x0 = s1 * step1
                y0 = s2 * step2
                block_2d = mat[y0:y0 + step2, x0:x0 + step1]
                add_compressed_segment(seg_idx, block_2d.flatten())

    # Build IVF header
    total_records = current_record - 1
    ivf_hdr = [0] * 128
    ivf_hdr[0] = 5          # ivf_version
    ivf_hdr[1] = nsegtot    # nsegtot
    ivf_hdr[2] = 4096       # drecbits (512 * 8)
    ivf_hdr[4] = 1          # consistent
    ivf_hdr[10] = ndescr    # ndescr
    ivf_hdr[11] = fdescr    # fdescr
    ivf_hdr[19] = descriptors[0][1]  # frec of seg 0
    ivf_hdr[20] = total_records
    ivf_hdr[127] = 5        # ivf_version

    ivf_bytes = struct.pack("<128i", *ivf_hdr)

    # Build descriptor table
    descr_bytes = bytearray()
    for nrec, frec in descriptors:
        descr_bytes.extend(struct.pack("<2i", nrec, frec))
    descr_bytes.extend(b"\x00" * (ndescr * 512 - len(descr_bytes)))

    # Write out .cmat file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(ivf_bytes)
        f.write(descr_bytes)
        for seg_idx in range(nsegtot):
            if seg_idx in segment_data_blocks:
                f.write(segment_data_blocks[seg_idx])


class CMATWriter:
    """
    Writer and compressor for GASPware/gsort 2D compressed matrices (.cmat).
    """

    def __init__(
        self,
        matrix: np.ndarray,
        symmetric: Optional[bool] = None,
        step1: int = 128,
        step2: int = 128,
    ):
        self.matrix = np.asarray(matrix, dtype=np.int32)
        if self.matrix.ndim != 2:
            raise ValueError("Input matrix must be a 2D array.")
        self.symmetric = symmetric
        self.step1 = step1
        self.step2 = step2

    def save(self, output_file: Union[str, Path]) -> None:
        """Write the matrix to the specified .cmat file."""
        write_cmat(
            output_file=output_file,
            matrix=self.matrix,
            symmetric=self.symmetric,
            step1=self.step1,
            step2=self.step2,
        )


