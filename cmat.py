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

            self.segsize = self.cmt_hdr[123]
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

    def get_projection(self, axis: int = 1) -> np.ndarray:
        """
        Get the stored 1D total projection spectrum.
        For symmetric 2D matrices, segment 2 holds the stored projection.
        """
        proj_seg_idx = 2  # PROJESEG + lato = 1 + 1 = 2
        with open(self.filename, "rb") as f:
            seg_data = self._read_raw_segment(f, proj_seg_idx)
            if seg_data is None or seg_data[0] == 0:
                # If no stored projection, compute from full matrix
                mat = self.to_numpy()
                return np.sum(mat, axis=1) + (np.diag(mat) if self.is_symmetric else 0)
            cmode, cminval, pack = seg_data
            return decompress_block(pack, self.res1, cmode, cminval)

    def to_numpy(self) -> np.ndarray:
        """
        Decompress and assemble the entire 2D matrix into a NumPy array.
        Returns:
            np.ndarray of shape (res1, res2) with dtype int32.
        """
        mat = np.zeros((self.res1, self.res2), dtype=np.int32)

        with open(self.filename, "rb") as f:
            if self.matmode == 1:  # Symmetrized 2D
                for s2 in range(self.ndiv2):
                    for s1 in range(s2 + 1):
                        iseg = s1 + (s2 * (s2 + 1)) // 2
                        seg_idx = iseg + self.nextra
                        cmode, cminval, pack = self._read_raw_segment(f, seg_idx)
                        if cmode == 0 and len(pack) == 0:
                            continue

                        block = decompress_block(pack, self.segsize, cmode, cminval)
                        block_2d = block.reshape((self.step2, self.step1))

                        x0 = s1 * self.step1
                        y0 = s2 * self.step2

                        if s1 == s2:
                            # Diagonal block: upper triangular (ki1 <= ki2)
                            for ki2 in range(self.step2):
                                for ki1 in range(ki2 + 1):
                                    val = block_2d[ki2, ki1]
                                    mat[y0 + ki2, x0 + ki1] = val
                                    mat[x0 + ki1, y0 + ki2] = val
                        else:
                            # Off-diagonal block
                            mat[y0:y0 + self.step2, x0:x0 + self.step1] = block_2d
                            mat[x0:x0 + self.step1, y0:y0 + self.step2] = block_2d.T

            elif self.matmode == 0:  # Normal (Non-symmetric) 2D
                for s2 in range(self.ndiv2):
                    for s1 in range(self.ndiv1):
                        iseg = s1 + self.ndiv1 * s2
                        seg_idx = iseg + self.nextra
                        cmode, cminval, pack = self._read_raw_segment(f, seg_idx)
                        if cmode == 0 and len(pack) == 0:
                            continue

                        block = decompress_block(pack, self.segsize, cmode, cminval)
                        block_2d = block.reshape((self.step2, self.step1))

                        x0 = s1 * self.step1
                        y0 = s2 * self.step2
                        mat[y0:y0 + self.step2, x0:x0 + self.step1] = block_2d

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
        g_min = max(0, min(gate_min, gate_max))
        g_max = min(self.res1 - 1, max(gate_min, gate_max))

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
                bx = min(self.res1 - 1, max(b_min, b_max))
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

