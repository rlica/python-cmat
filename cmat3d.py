"""
cmat3d.py - Python Reader, Decompressor and Slicer for GASPware/gsort 3D .cmat Matrices

Reverse-engineered from GASPware source code (ivflib.F, cmtlib.F, complib.c).
Dedicated to 3-dimensional coincidence matrices (e.g., GeE-Rings3D.cmat, gamma-gamma-ring/time).
Author: Antigravity Assistant & pair programming with user.
"""

import os
import sys
import time
import math
import struct
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np

from cmat import decompress_block


class CMAT3DReader:
    """
    High-performance Reader and Slicer for GASPware/gsort compressed 3D matrices (.cmat).
    Features:
      - Direct parsing of 3D IVF/CMT headers and pre-stored total 1D projections (Seg 2, 3, 4).
      - Column-major 3D sub-block coordinate addressing.
      - Fast on-disk caching with memory mapping (np.memmap) for rapid loading (<10ms on repeat)
        and instantaneous arbitrary 3D slicing, single-gating, and double-gating.
      - Pre-computed 2D projections for all three plane orientations (0-1, 0-2, 1-2).
    """

    def __init__(
        self,
        filename: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.filename = Path(filename).resolve()
        if not self.filename.exists():
            raise FileNotFoundError(f"File not found: {self.filename}")

        self.use_cache = use_cache
        self.show_progress = show_progress
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = self.filename.parent / ".cmat3d_cache"

        self._read_headers()

        # Placeholders for data structures
        self.memmap_3d: Optional[np.memmap] = None
        self.proj_2d: Dict[str, np.ndarray] = {}
        self._total_counts: Optional[int] = None
        self._max_count: Optional[int] = None
        self._nonzero_voxels: Optional[int] = None

        # Build or load cache
        if self.use_cache:
            self._init_data_cache()

    def _read_headers(self):
        with open(self.filename, "rb") as f:
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

            # Read IVF Descriptors table
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
            if self.ndim != 3:
                raise ValueError(
                    f"File '{self.filename.name}' has {self.ndim} dimensions, but CMAT3DReader expects a 3D matrix (ndim=3)."
                )

            self.matmode = self.cmt_hdr[1]  # 0=normal
            # Axis 0 (X)
            self.res1 = self.cmt_hdr[3]
            self.step1 = self.cmt_hdr[4]
            self.ndiv1 = self.cmt_hdr[5]
            # Axis 1 (Y)
            self.res2 = self.cmt_hdr[6]
            self.step2 = self.cmt_hdr[7]
            self.ndiv2 = self.cmt_hdr[8]
            # Axis 2 (Z)
            self.res3 = self.cmt_hdr[9]
            self.step3 = self.cmt_hdr[10]
            self.ndiv3 = self.cmt_hdr[11]

            # Ensure resolutions match grid divisions
            if self.step1 > 0 and self.ndiv1 > 0:
                self.res1 = max(self.res1, self.step1 * self.ndiv1)
            if self.step2 > 0 and self.ndiv2 > 0:
                self.res2 = max(self.res2, self.step2 * self.ndiv2)
            if self.step3 > 0 and self.ndiv3 > 0:
                self.res3 = max(self.res3, self.step3 * self.ndiv3)

            self.segsize = self.step1 * self.step2 * self.step3
            self.nmatrix_segs = self.ndiv1 * self.ndiv2 * self.ndiv3
            self.nextra = self.cmt_hdr[125]
            self.cmt_version = self.cmt_hdr[127]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Matrix dimensions (res1, res2, res3) [X, Y, Z]."""
        return (self.res1, self.res2, self.res3)

    @property
    def step(self) -> Tuple[int, int, int]:
        """Sub-block step dimensions (step1, step2, step3)."""
        return (self.step1, self.step2, self.step3)

    @property
    def blocks(self) -> Tuple[int, int, int]:
        """Grid block counts (ndiv1, ndiv2, ndiv3)."""
        return (self.ndiv1, self.ndiv2, self.ndiv3)

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        """NumPy array shape: (Axis 2 / Z, Axis 1 / Y, Axis 0 / X)."""
        return (self.res3, self.res2, self.res1)

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
        Get 1D total projection spectrum for specified axis:
          axis 0: Axis 1 / X (length res1)
          axis 1: Axis 2 / Y (length res2)
          axis 2: Axis 3 / Z (length res3)
        """
        if axis not in (0, 1, 2):
            axis = 0

        target_res = self.res1 if axis == 0 else (self.res2 if axis == 1 else self.res3)
        # Stored projections: seg 2 = axis 0, seg 3 = axis 1, seg 4 = axis 2
        proj_seg_idx = 2 + axis

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

        # Fallback: compute from 3D array or 2D projections if available
        if "0-1" in self.proj_2d:
            if axis == 0:
                return np.sum(self.proj_2d["0-1"], axis=0)
            elif axis == 1:
                return np.sum(self.proj_2d["0-1"], axis=1)
            else:
                return np.sum(self.proj_2d["0-2"], axis=1)

        return np.zeros(target_res, dtype=np.int32)

    def _get_cache_paths(self) -> Tuple[Path, Path]:
        stat = self.filename.stat()
        mtime = int(stat.st_mtime)
        size = stat.st_size
        stem = self.filename.stem
        dat_name = f"{stem}_{mtime}_{size}_3d.bin"
        npz_name = f"{stem}_{mtime}_{size}_proj2d.npz"
        return self.cache_dir / dat_name, self.cache_dir / npz_name

    def _init_data_cache(self):
        """Initializes or loads memory-mapped 3D array and pre-summed 2D projections."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dat_path, npz_path = self._get_cache_paths()

        if dat_path.exists() and npz_path.exists():
            try:
                # Load pre-computed 2D projections
                with np.load(npz_path) as npz:
                    self.proj_2d["0-1"] = npz["p01"]
                    self.proj_2d["0-2"] = npz["p02"]
                    self.proj_2d["1-2"] = npz["p12"]
                    self._total_counts = int(npz.get("total_counts", np.sum(npz["p01"])))
                    self._max_count = int(npz.get("max_count", 0))
                    self._nonzero_voxels = int(npz.get("nonzero_voxels", 0))

                # Open memory mapped 3D array in read-only mode
                self.memmap_3d = np.memmap(
                    dat_path,
                    dtype=np.int32,
                    mode="r",
                    shape=self.shape_zyx,
                )
                return
            except Exception as e:
                print(f"[*] Cache invalid or corrupted ({e}), rebuilding...", file=sys.stderr)

        # Cache build required
        self._build_cache(dat_path, npz_path)

    def _build_cache(self, dat_path: Path, npz_path: Path):
        t0 = time.time()
        print(f"\n[*] Initializing 3D matrix cache for '{self.filename.name}'...")
        print(f"    Dimensions: {self.res1} (X) × {self.res2} (Y) × {self.res3} (Z)")
        print(f"    Sub-blocks: {self.ndiv1} × {self.ndiv2} × {self.ndiv3} = {self.nmatrix_segs:,} total")

        # Create memory mapped array file on disk
        # Shape: (res3, res2, res1) -> (Z, Y, X)
        # Pre-fill file with zeros if not existing
        shape_zyx = self.shape_zyx
        fp = np.memmap(dat_path, dtype=np.int32, mode="w+", shape=shape_zyx)

        p01 = np.zeros((self.res2, self.res1), dtype=np.int32)
        p02 = np.zeros((self.res3, self.res1), dtype=np.int32)
        p12 = np.zeros((self.res3, self.res2), dtype=np.int32)

        total_voxels_cnt = 0
        max_voxel_val = 0
        nonzero_cnt = 0

        # Scan non-zero matrix block descriptors
        fdescr = self.fdescr
        nextra = self.nextra
        step1, step2, step3 = self.step1, self.step2, self.step3
        ndiv1, ndiv2, ndiv3 = self.ndiv1, self.ndiv2, self.ndiv3
        segsize = self.segsize

        with open(self.filename, "rb") as f:
            f.seek((fdescr - 1) * 512 + nextra * 8)
            block_descrs = [
                struct.unpack_from("<2i", f.read(8))
                for _ in range(self.nmatrix_segs)
            ]

            nz_list = [(iseg, d) for iseg, d in enumerate(block_descrs) if d[0] > 0]
            n_nonzero = len(nz_list)
            print(f"    Non-zero sub-blocks to decompress: {n_nonzero:,} / {self.nmatrix_segs:,}")

            last_report = time.time()
            for idx, (iseg, (nrec, frec)) in enumerate(nz_list):
                # 3D block coordinate: iseg = s1 + ndiv1 * (s2 + ndiv2 * s3)
                s1 = iseg % ndiv1
                rem = iseg // ndiv1
                s2 = rem % ndiv2
                s3 = rem // ndiv2

                x0 = s1 * step1
                x1 = min(x0 + step1, self.res1)
                y0 = s2 * step2
                y1 = min(y0 + step2, self.res2)
                z0 = s3 * step3
                z1 = min(z0 + step3, self.res3)

                f.seek((frec - 1) * 512)
                raw = f.read(nrec * 512)
                cmode, cminval = struct.unpack_from("<2i", raw, 0)
                decomp = decompress_block(raw[8:], segsize, cmode, cminval)

                if len(decomp) < segsize:
                    decomp = np.pad(decomp, (0, segsize - len(decomp)))
                elif len(decomp) > segsize:
                    decomp = decomp[:segsize]

                # Reshape to (step3, step2, step1) = (Z, Y, X)
                b3d = decomp.reshape((step3, step2, step1))

                # Crop if block boundaries exceed resolution
                dz = z1 - z0
                dy = y1 - y0
                dx = x1 - x0
                b_slice = b3d[:dz, :dy, :dx]

                # Store into memmap
                fp[z0:z1, y0:y1, x0:x1] = b_slice

                # Accumulate 2D projections
                p01[y0:y1, x0:x1] += b_slice.sum(axis=0)
                p02[z0:z1, x0:x1] += b_slice.sum(axis=1)
                p12[z0:z1, y0:y1] += b_slice.sum(axis=2)

                b_sum = int(np.sum(b_slice))
                b_max = int(np.max(b_slice))
                b_nz = int(np.count_nonzero(b_slice))

                total_voxels_cnt += b_sum
                if b_max > max_voxel_val:
                    max_voxel_val = b_max
                nonzero_cnt += b_nz

                if self.show_progress and (time.time() - last_report > 0.5 or idx == n_nonzero - 1):
                    pct = (idx + 1) * 100.0 / n_nonzero
                    elapsed = time.time() - t0
                    eta = (elapsed / (idx + 1)) * (n_nonzero - idx - 1)
                    print(
                        f"\r    [Progress: {pct:5.1f}% | Block {idx + 1:,}/{n_nonzero:,} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s]",
                        end="",
                        flush=True,
                    )
                    last_report = time.time()

        print("\n    Flushing cache to disk...", flush=True)
        fp.flush()
        del fp

        # Save 2D projections to npz
        np.savez_compressed(
            npz_path,
            p01=p01,
            p02=p02,
            p12=p12,
            total_counts=total_voxels_cnt,
            max_count=max_voxel_val,
            nonzero_voxels=nonzero_cnt,
        )

        self.proj_2d["0-1"] = p01
        self.proj_2d["0-2"] = p02
        self.proj_2d["1-2"] = p12
        self._total_counts = total_voxels_cnt
        self._max_count = max_voxel_val
        self._nonzero_voxels = nonzero_cnt

        self.memmap_3d = np.memmap(dat_path, dtype=np.int32, mode="r", shape=shape_zyx)
        elapsed_tot = time.time() - t0
        print(f"[+] 3D Matrix successfully cached in {elapsed_tot:.1f}s! Total counts: {total_voxels_cnt:,}\n")

    def get_2d_plane(
        self,
        plane: str = "0-1",
        gate_3rd: Optional[Tuple[int, int]] = None,
        gates_3rd: Optional[List[Tuple[int, int]]] = None,
        x0: Optional[int] = None,
        x1: Optional[int] = None,
        y0: Optional[int] = None,
        y1: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract a 2D projection or gated plane (optionally sliced to subregion x0..x1, y0..y1).
        Planes:
          "0-1": X = Axis 0 (Axis 1 / X), Y = Axis 1 (Axis 2 / Y), 3rd = Axis 2 (Axis 3 / Z)
          "0-2": X = Axis 0 (Axis 1 / X), Y = Axis 2 (Axis 3 / Z), 3rd = Axis 1 (Axis 2 / Y)
          "1-2": X = Axis 1 (Axis 2 / Y), Y = Axis 2 (Axis 3 / Z), 3rd = Axis 0 (Axis 1 / X)

        If gate_3rd or gates_3rd is specified, sums only over the given 3rd axis range(s).
        Otherwise returns the full projection. Slicing subregion before summation accelerates
        rendering by up to 50x.
        """
        all_gates = []
        if gate_3rd is not None:
            all_gates.append(gate_3rd)
        if gates_3rd:
            all_gates.extend(gates_3rd)

        plane_norm = str(plane).strip()
        if plane_norm not in ("0-1", "0-2", "1-2"):
            plane_norm = "0-1"

        # Determine dimension bounds
        if plane_norm == "0-1":
            dim_x, dim_y = self.res1, self.res2
        elif plane_norm == "0-2":
            dim_x, dim_y = self.res1, self.res3
        else:
            dim_x, dim_y = self.res2, self.res3

        rx0 = 0 if x0 is None else max(0, min(dim_x - 1, x0))
        rx1 = dim_x if x1 is None else max(rx0 + 1, min(dim_x, x1))
        ry0 = 0 if y0 is None else max(0, min(dim_y - 1, y0))
        ry1 = dim_y if y1 is None else max(ry0 + 1, min(dim_y, y1))
        is_subregion = (rx0 > 0 or rx1 < dim_x or ry0 > 0 or ry1 < dim_y)

        # If no gating on 3rd axis, return from cached total 2D projection immediately (<0.1ms)
        if not all_gates:
            if plane_norm in self.proj_2d:
                p = self.proj_2d[plane_norm]
                return p[ry0:ry1, rx0:rx1] if is_subregion else p

        if self.memmap_3d is None:
            self._init_data_cache()

        # Plane 0-1: shape (res2, res1), 3rd axis is Z (res3), memmap is (res3, res2, res1)
        if plane_norm == "0-1":
            if not all_gates:
                return np.sum(self.memmap_3d[:, ry0:ry1, rx0:rx1], axis=0, dtype=np.int32)
            out = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.int32)
            for z0, z1 in all_gates:
                zm = max(0, min(self.res3 - 1, min(z0, z1)))
                zx = max(0, min(self.res3, max(z0, z1) + 1))
                if zx > zm:
                    out += np.sum(self.memmap_3d[zm:zx, ry0:ry1, rx0:rx1], axis=0, dtype=np.int32)
            return out

        # Plane 0-2: shape (res3, res1), 3rd axis is Y (res2), memmap is (res3, res2, res1)
        elif plane_norm == "0-2":
            if not all_gates:
                return np.sum(self.memmap_3d[ry0:ry1, :, rx0:rx1], axis=1, dtype=np.int32)
            out = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.int32)
            for y0_g, y1_g in all_gates:
                ym = max(0, min(self.res2 - 1, min(y0_g, y1_g)))
                yx = max(0, min(self.res2, max(y0_g, y1_g) + 1))
                if yx > ym:
                    out += np.sum(self.memmap_3d[ry0:ry1, ym:yx, rx0:rx1], axis=1, dtype=np.int32)
            return out

        # Plane 1-2: shape (res3, res2), 3rd axis is X (res1), memmap is (res3, res2, res1)
        else:
            if not all_gates:
                return np.sum(self.memmap_3d[ry0:ry1, rx0:rx1, :], axis=2, dtype=np.int32)
            out = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.int32)
            for x0_g, x1_g in all_gates:
                xm = max(0, min(self.res1 - 1, min(x0_g, x1_g)))
                xx = max(0, min(self.res1, max(x0_g, x1_g) + 1))
                if xx > xm:
                    out += np.sum(self.memmap_3d[ry0:ry1, rx0:rx1, xm:xx], axis=2, dtype=np.int32)
            return out

    def get_projections_for_region(
        self,
        plane: str = "0-1",
        x0: int = 0,
        x1: Optional[int] = None,
        y0: int = 0,
        y1: Optional[int] = None,
        gate_3rd: Optional[Tuple[int, int]] = None,
        gates_3rd: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 3 1D histograms (spec0, spec1, spec2) for the visible 2D box (x0..x1, y0..y1)
        and any optional gate on the 3rd axis.
        """
        if self.memmap_3d is None:
            self._init_data_cache()

        all_gates_3rd = []
        if gate_3rd is not None:
            all_gates_3rd.append(gate_3rd)
        if gates_3rd:
            all_gates_3rd.extend(gates_3rd)

        plane_norm = str(plane).strip()
        if plane_norm not in ("0-1", "0-2", "1-2"):
            plane_norm = "0-1"

        if plane_norm == "0-1":
            rx0 = max(0, min(self.res1 - 1, x0))
            rx1 = max(rx0 + 1, min(self.res1, x1 if x1 is not None else self.res1))
            ry0 = max(0, min(self.res2 - 1, y0))
            ry1 = max(ry0 + 1, min(self.res2, y1 if y1 is not None else self.res2))

            # Full 3rd axis or gated 3rd axis
            if not all_gates_3rd:
                # Fast path using 2D projections when not gated on 3rd axis
                if ry0 == 0 and ry1 >= self.res2:
                    spec0 = self.get_projection(0)
                else:
                    spec0 = np.sum(self.proj_2d["0-1"][ry0:ry1, :], axis=0, dtype=np.int64)

                if rx0 == 0 and rx1 >= self.res1:
                    spec1 = self.get_projection(1)
                else:
                    spec1 = np.sum(self.proj_2d["0-1"][:, rx0:rx1], axis=1, dtype=np.int64)

                if rx0 == 0 and rx1 >= self.res1 and ry0 == 0 and ry1 >= self.res2:
                    spec2 = self.get_projection(2)
                else:
                    sub = self.memmap_3d[:, ry0:ry1, rx0:rx1]
                    spec2 = np.sum(sub, axis=(1, 2), dtype=np.int64)
            else:
                # Direct slice summation without allocating zero volume
                spec0 = np.zeros(self.res1, dtype=np.int64)
                spec1 = np.zeros(self.res2, dtype=np.int64)
                for zm, zx in all_gates_3rd:
                    z_lo = max(0, min(self.res3 - 1, min(zm, zx)))
                    z_hi = max(0, min(self.res3, max(zm, zx) + 1))
                    if z_hi > z_lo:
                        chunk = self.memmap_3d[z_lo:z_hi, ry0:ry1, rx0:rx1]
                        spec0[rx0:rx1] += np.sum(chunk, axis=(0, 1), dtype=np.int64)
                        spec1[ry0:ry1] += np.sum(chunk, axis=(0, 2), dtype=np.int64)

                spec2 = np.sum(self.memmap_3d[:, ry0:ry1, rx0:rx1], axis=(1, 2), dtype=np.int64)

            return spec0, spec1, spec2

        elif plane_norm == "0-2":
            # X = Axis 0 (Axis 1 / X), Y = Axis 2 (Axis 3 / Z), 3rd = Axis 1 (Axis 2 / Y)
            rx0 = max(0, min(self.res1 - 1, x0))
            rx1 = max(rx0 + 1, min(self.res1, x1 if x1 is not None else self.res1))
            rz0 = max(0, min(self.res3 - 1, y0))
            rz1 = max(rz0 + 1, min(self.res3, y1 if y1 is not None else self.res3))

            if not all_gates_3rd:
                if rz0 == 0 and rz1 >= self.res3:
                    spec0 = self.get_projection(0)
                else:
                    spec0 = np.sum(self.proj_2d["0-2"][rz0:rz1, :], axis=0, dtype=np.int64)

                if rx0 == 0 and rx1 >= self.res1:
                    spec2 = self.get_projection(2)
                else:
                    spec2 = np.sum(self.proj_2d["0-2"][:, rx0:rx1], axis=1, dtype=np.int64)

                sub = self.memmap_3d[rz0:rz1, :, rx0:rx1]
                spec1 = np.sum(sub, axis=(0, 2), dtype=np.int64)
            else:
                spec0 = np.zeros(self.res1, dtype=np.int64)
                spec2 = np.zeros(self.res3, dtype=np.int64)
                for ym, yx in all_gates_3rd:
                    y_lo = max(0, min(self.res2 - 1, min(ym, yx)))
                    y_hi = max(0, min(self.res2, max(ym, yx) + 1))
                    if y_hi > y_lo:
                        chunk = self.memmap_3d[rz0:rz1, y_lo:y_hi, rx0:rx1]
                        spec0[rx0:rx1] += np.sum(chunk, axis=(0, 1), dtype=np.int64)
                        spec2[rz0:rz1] += np.sum(chunk, axis=(1, 2), dtype=np.int64)

                spec1 = np.sum(self.memmap_3d[rz0:rz1, :, rx0:rx1], axis=(0, 2), dtype=np.int64)

            return spec0, spec1, spec2

        else:
            # Plane 1-2: X = Axis 1 (Axis 2 / Y), Y = Axis 2 (Axis 3 / Z), 3rd = Axis 0 (Axis 1 / X)
            ry0 = max(0, min(self.res2 - 1, x0))
            ry1 = max(ry0 + 1, min(self.res2, x1 if x1 is not None else self.res2))
            rz0 = max(0, min(self.res3 - 1, y0))
            rz1 = max(rz0 + 1, min(self.res3, y1 if y1 is not None else self.res3))

            if not all_gates_3rd:
                if rz0 == 0 and rz1 >= self.res3:
                    spec1 = self.get_projection(1)
                else:
                    spec1 = np.sum(self.proj_2d["1-2"][rz0:rz1, :], axis=0, dtype=np.int64)

                if ry0 == 0 and ry1 >= self.res2:
                    spec2 = self.get_projection(2)
                else:
                    spec2 = np.sum(self.proj_2d["1-2"][:, ry0:ry1], axis=1, dtype=np.int64)

                sub = self.memmap_3d[rz0:rz1, ry0:ry1, :]
                spec0 = np.sum(sub, axis=(0, 1), dtype=np.int64)
            else:
                spec1 = np.zeros(self.res2, dtype=np.int64)
                spec2 = np.zeros(self.res3, dtype=np.int64)
                for xm, xx in all_gates_3rd:
                    x_lo = max(0, min(self.res1 - 1, min(xm, xx)))
                    x_hi = max(0, min(self.res1, max(xm, xx) + 1))
                    if x_hi > x_lo:
                        chunk = self.memmap_3d[rz0:rz1, ry0:ry1, x_lo:x_hi]
                        spec1[ry0:ry1] += np.sum(chunk, axis=(0, 2), dtype=np.int64)
                        spec2[rz0:rz1] += np.sum(chunk, axis=(1, 2), dtype=np.int64)

                spec0 = np.sum(self.memmap_3d[rz0:rz1, ry0:ry1, :], axis=(0, 1), dtype=np.int64)

            return spec0, spec1, spec2

    def get_info(self) -> Dict[str, Any]:
        return {
            "filename": str(self.filename),
            "dimensions": self.ndim,
            "shape": (self.res1, self.res2, self.res3),
            "step": (self.step1, self.step2, self.step3),
            "blocks": (self.ndiv1, self.ndiv2, self.ndiv3),
            "matrix_segments": self.nmatrix_segs,
            "extra_segments": self.nextra,
            "total_segments": self.nsegtot,
            "segment_size": self.segsize,
            "matrix_mode": "Normal" if self.matmode == 0 else f"Mode {self.matmode}",
            "ivf_version": self.ivf_version,
            "cmt_version": self.cmt_version,
            "total_counts": self._total_counts,
            "max_count": self._max_count,
            "nonzero_voxels": self._nonzero_voxels,
        }


def parse_gate_ranges(gate_str: str) -> List[List[int]]:
    """Parse comma/space separated channel ranges like '100-120, 150:160' or '100,120'."""
    if not gate_str or not str(gate_str).strip():
        return []
    cleaned = str(gate_str).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        import json
        try:
            val = json.loads(cleaned)
            res = []
            for item in val:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    c0 = int(round(float(item[0])))
                    c1 = int(round(float(item[1])))
                    res.append([min(c0, c1), max(c0, c1)])
            return res
        except Exception:
            pass

    tokens = cleaned.replace(";", ",").split(",")
    pairs = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok and not tok.startswith("-"):
            parts = tok.split("-")
        elif ":" in tok:
            parts = tok.split(":")
        elif " " in tok:
            parts = tok.split()
        else:
            parts = [tok, tok]

        if len(parts) >= 2:
            try:
                c0 = int(round(float(parts[0])))
                c1 = int(round(float(parts[1])))
                pairs.append([min(c0, c1), max(c0, c1)])
            except ValueError:
                continue
    return pairs


def compute_3d_gate(
    reader: CMAT3DReader,
    target_axis: int,
    gate_specs: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute single-gated or double-gated 1D coincidence spectrum on target_axis.

    Args:
        reader: CMAT3DReader instance
        target_axis: 0, 1, or 2 (axis for the resulting 1D coincidence spectrum)
        gate_specs: dict keyed by gated axis (e.g. 1 and/or 2):
          {
             axis_idx: {
                 'w': [[w0, w1], ...],  # peak gate ranges
                 'b': [[b0, b1], ...],  # bg gate ranges
             }
          }
    """
    if reader.memmap_3d is None:
        reader._init_data_cache()

    target_res = reader.res1 if target_axis == 0 else (reader.res2 if target_axis == 1 else reader.res3)
    active_gated_axes = [ax for ax in sorted(gate_specs.keys()) if ax in (0, 1, 2) and ax != target_axis]

    if not active_gated_axes:
        proj = reader.get_projection(target_axis).astype(np.float64)
        return {
            "success": True,
            "target_axis": target_axis,
            "net_spec": proj.tolist(),
            "raw_spec": proj.tolist(),
            "bg_spec": np.zeros_like(proj).tolist(),
            "scale": 0.0,
            "w_width": 0,
            "b_width": 0,
            "is_double_gate": False,
        }

    def make_mask(axis_len: int, ranges: List[List[int]]) -> np.ndarray:
        m = np.zeros(axis_len, dtype=bool)
        for r in ranges:
            c0 = max(0, min(axis_len - 1, int(round(float(r[0])))))
            c1 = max(0, min(axis_len - 1, int(round(float(r[1])))))
            if c1 >= c0:
                m[c0:c1 + 1] = True
        return m

    ax_to_dim = {0: 2, 1: 1, 2: 0}

    if len(active_gated_axes) == 1:
        ga = active_gated_axes[0]
        w_ranges = gate_specs[ga].get("w", [])
        b_ranges = gate_specs[ga].get("b", [])
        len_a = reader.res1 if ga == 0 else (reader.res2 if ga == 1 else reader.res3)

        w_mask = make_mask(len_a, w_ranges)
        b_mask = make_mask(len_a, b_ranges)
        w_width = int(np.sum(w_mask))
        b_width = int(np.sum(b_mask))

        if (target_axis, ga) in ((0, 1), (1, 0)):
            p2d = reader.proj_2d["0-1"]
            if ga == 1:
                raw_w = np.sum(p2d[w_mask, :], axis=0, dtype=np.float64) if w_width > 0 else np.zeros(target_res)
                raw_b = np.sum(p2d[b_mask, :], axis=0, dtype=np.float64) if b_width > 0 else np.zeros(target_res)
            else:
                raw_w = np.sum(p2d[:, w_mask], axis=1, dtype=np.float64) if w_width > 0 else np.zeros(target_res)
                raw_b = np.sum(p2d[:, b_mask], axis=1, dtype=np.float64) if b_width > 0 else np.zeros(target_res)

        elif (target_axis, ga) in ((0, 2), (2, 0)):
            p2d = reader.proj_2d["0-2"]
            if ga == 2:
                raw_w = np.sum(p2d[w_mask, :], axis=0, dtype=np.float64) if w_width > 0 else np.zeros(target_res)
                raw_b = np.sum(p2d[b_mask, :], axis=0, dtype=np.float64) if b_width > 0 else np.zeros(target_res)
            else:
                raw_w = np.sum(p2d[:, w_mask], axis=1, dtype=np.float64) if w_width > 0 else np.zeros(target_res)
                raw_b = np.sum(p2d[:, b_mask], axis=1, dtype=np.float64) if b_width > 0 else np.zeros(target_res)

        else:
            p2d = reader.proj_2d["1-2"]
            if ga == 2:
                raw_w = np.sum(p2d[w_mask, :], axis=0, dtype=np.float64) if w_width > 0 else np.zeros(target_res)
                raw_b = np.sum(p2d[b_mask, :], axis=0, dtype=np.float64) if b_width > 0 else np.zeros(target_res)
            else:
                raw_w = np.sum(p2d[:, w_mask], axis=1, dtype=np.float64) if w_width > 0 else np.zeros(target_res)
                raw_b = np.sum(p2d[:, b_mask], axis=1, dtype=np.float64) if b_width > 0 else np.zeros(target_res)

        if b_width > 0 and w_width > 0:
            scale = float(w_width) / float(b_width)
            bg_spec = raw_b * scale
            net_spec = raw_w - bg_spec
        else:
            scale = 0.0
            bg_spec = np.zeros_like(raw_w)
            net_spec = raw_w.copy()

        return {
            "success": True,
            "target_axis": target_axis,
            "net_spec": net_spec.tolist(),
            "raw_spec": raw_w.tolist(),
            "bg_spec": bg_spec.tolist(),
            "scale": scale,
            "w_width": w_width,
            "b_width": b_width,
            "is_double_gate": False,
        }

    else:
        g1, g2 = active_gated_axes[0], active_gated_axes[1]
        w1_ranges, b1_ranges = gate_specs[g1].get("w", []), gate_specs[g1].get("b", [])
        w2_ranges, b2_ranges = gate_specs[g2].get("w", []), gate_specs[g2].get("b", [])

        len1 = reader.res1 if g1 == 0 else (reader.res2 if g1 == 1 else reader.res3)
        len2 = reader.res1 if g2 == 0 else (reader.res2 if g2 == 1 else reader.res3)

        w1_mask, b1_mask = make_mask(len1, w1_ranges), make_mask(len1, b1_ranges)
        w2_mask, b2_mask = make_mask(len2, w2_ranges), make_mask(len2, b2_ranges)

        w1_width, b1_width = int(np.sum(w1_mask)), int(np.sum(b1_mask))
        w2_width, b2_width = int(np.sum(w2_mask)), int(np.sum(b2_mask))

        def project_region(m1: np.ndarray, m2: np.ndarray, n1: int, n2: int) -> np.ndarray:
            if n1 == 0 or n2 == 0:
                return np.zeros(target_res, dtype=np.float64)

            idx1 = np.where(m1)[0]
            idx2 = np.where(m2)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                return np.zeros(target_res, dtype=np.float64)

            min1, max1 = idx1[0], idx1[-1] + 1
            min2, max2 = idx2[0], idx2[-1] + 1

            sub_m1 = m1[min1:max1]
            sub_m2 = m2[min2:max2]

            slices = [slice(None), slice(None), slice(None)]
            dim1 = ax_to_dim[g1]
            dim2 = ax_to_dim[g2]
            slices[dim1] = slice(min1, max1)
            slices[dim2] = slice(min2, max2)

            sub_3d = reader.memmap_3d[tuple(slices)].astype(np.float64)

            if not np.all(sub_m1):
                idx_shape1 = [1, 1, 1]
                idx_shape1[dim1] = len(sub_m1)
                sub_3d *= sub_m1.reshape(idx_shape1)

            if not np.all(sub_m2):
                idx_shape2 = [1, 1, 1]
                idx_shape2[dim2] = len(sub_m2)
                sub_3d *= sub_m2.reshape(idx_shape2)

            sum_axes = tuple(sorted([dim1, dim2]))
            return np.sum(sub_3d, axis=sum_axes, dtype=np.float64)

        raw_ww = project_region(w1_mask, w2_mask, w1_width, w2_width)
        raw_bw = project_region(b1_mask, w2_mask, b1_width, w2_width)
        raw_wb = project_region(w1_mask, b2_mask, w1_width, b2_width)
        raw_bb = project_region(b1_mask, b2_mask, b1_width, b2_width)

        alpha = (float(w1_width) / float(b1_width)) if b1_width > 0 else 0.0
        beta = (float(w2_width) / float(b2_width)) if b2_width > 0 else 0.0

        bg_spec = alpha * raw_bw + beta * raw_wb - (alpha * beta) * raw_bb
        net_spec = raw_ww - bg_spec

        return {
            "success": True,
            "target_axis": target_axis,
            "net_spec": net_spec.tolist(),
            "raw_spec": raw_ww.tolist(),
            "bg_spec": bg_spec.tolist(),
            "scale_1": alpha,
            "scale_2": beta,
            "w1_width": w1_width,
            "b1_width": b1_width,
            "w2_width": w2_width,
            "b2_width": b2_width,
            "is_double_gate": True,
        }


def _polygon_area(pts: List[Tuple[float, float]]) -> float:
    """Compute 2D continuous geometric polygon area via Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _extract_banana_spectrum(
    reader: CMAT3DReader,
    plane: str,
    raw_points: List[Any],
    dim_x: int,
    dim_y: int,
    target_res: int,
) -> Tuple[np.ndarray, int, float, int]:
    """
    Extracts 1D projection spectrum, discrete pixel count, continuous surface area, and total counts for a 2D polygon.
    Returns: (spec_1d, pixel_count, surface_area, total_counts)
    """
    if not raw_points or len(raw_points) < 3:
        return np.zeros(target_res, dtype=np.float64), 0, 0.0, 0

    clean_pts = []
    for p in raw_points:
        if isinstance(p, dict):
            clean_pts.append((float(p.get("x", p.get(0, 0))), float(p.get("y", p.get(1, 0)))))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            clean_pts.append((float(p[0]), float(p[1])))

    if len(clean_pts) < 3:
        return np.zeros(target_res, dtype=np.float64), 0, 0.0, 0

    surface_area = _polygon_area(clean_pts)

    xs = [p[0] for p in clean_pts]
    ys = [p[1] for p in clean_pts]
    x_min = max(0, int(math.floor(min(xs))))
    x_max = min(dim_x, int(math.ceil(max(xs))))
    y_min = max(0, int(math.floor(min(ys))))
    y_max = min(dim_y, int(math.ceil(max(ys))))

    if x_max <= x_min or y_max <= y_min:
        return np.zeros(target_res, dtype=np.float64), 0, surface_area, 0

    nx = x_max - x_min
    ny = y_max - y_min

    from matplotlib.path import Path as MplPath

    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max) + 0.5, np.arange(y_min, y_max) + 0.5)
    points_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    poly_path = MplPath(clean_pts)
    mask = poly_path.contains_points(points_grid).reshape(ny, nx)
    pixel_count = int(np.sum(mask))

    if pixel_count == 0:
        return np.zeros(target_res, dtype=np.float64), 0, surface_area, 0

    if plane == "0-1":
        sub = reader.memmap_3d[:, y_min:y_max, x_min:x_max].astype(np.float64)
        spec = np.sum(sub * mask[np.newaxis, :, :], axis=(1, 2), dtype=np.float64)
    elif plane == "0-2":
        sub = reader.memmap_3d[y_min:y_max, :, x_min:x_max].astype(np.float64)
        spec = np.sum(sub * mask[:, np.newaxis, :], axis=(0, 2), dtype=np.float64)
    else:  # plane == "1-2"
        sub = reader.memmap_3d[y_min:y_max, x_min:x_max, :].astype(np.float64)
        spec = np.sum(sub * mask[:, :, np.newaxis], axis=(0, 1), dtype=np.float64)

    total_counts = int(round(float(np.sum(spec))))
    return spec, pixel_count, surface_area, total_counts


def compute_2d_banana_gate(
    reader: CMAT3DReader,
    plane: str,
    polygon_peak: Optional[List[Any]] = None,
    polygon_bg: Optional[List[Any]] = None,
    polygon_points: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Compute 1D coincidence spectrum on the remaining 3rd axis for a 2D banana (polygon) gate on `plane`.
    Supports peak banana with optional background banana subtraction normalized by surface areas.

    plane: '0-1', '0-2', or '1-2'
    polygon_peak / polygon_points: vertices for peak banana [x, y]
    polygon_bg: optional vertices for background banana [x, y]
    """
    if reader.memmap_3d is None:
        reader._init_data_cache()

    if polygon_peak is None and polygon_points is not None:
        polygon_peak = polygon_points

    plane = str(plane).strip()
    if plane == "0-1":
        dim_x, dim_y = reader.res1, reader.res2
        target_axis = 2
        target_res = reader.res3
    elif plane == "0-2":
        dim_x, dim_y = reader.res1, reader.res3
        target_axis = 1
        target_res = reader.res2
    elif plane == "1-2":
        dim_x, dim_y = reader.res2, reader.res3
        target_axis = 0
        target_res = reader.res1
    else:
        raise ValueError(f"Invalid plane '{plane}'. Must be '0-1', '0-2', or '1-2'.")

    has_peak = bool(polygon_peak and len(polygon_peak) >= 3)
    has_bg_poly = bool(polygon_bg and len(polygon_bg) >= 3)

    if not has_peak and not has_bg_poly:
        return {
            "success": False,
            "error": "Polygon must contain at least 3 vertices.",
            "plane": plane,
            "target_axis": target_axis,
            "net_spec": np.zeros(target_res, dtype=np.float64).tolist(),
            "total_gated_counts": 0,
            "total_counts": 0,
            "pixel_count": 0,
            "elapsed_ms": 0.0,
        }

    t0 = time.time()

    spec_peak, px_peak, area_peak, cts_peak = _extract_banana_spectrum(
        reader, plane, polygon_peak or [], dim_x, dim_y, target_res
    )

    has_bg = False
    spec_bg = np.zeros(target_res, dtype=np.float64)
    px_bg = 0
    area_bg = 0.0
    cts_bg = 0
    scale = 0.0

    if has_bg_poly:
        spec_bg, px_bg, area_bg, cts_bg = _extract_banana_spectrum(
            reader, plane, polygon_bg or [], dim_x, dim_y, target_res
        )
        if px_bg > 0 or area_bg > 0:
            has_bg = True
            norm_area_peak = px_peak if px_peak > 0 else area_peak
            norm_area_bg = px_bg if px_bg > 0 else area_bg
            scale = (norm_area_peak / norm_area_bg) if norm_area_bg > 0 else 1.0

    if has_bg:
        net_spec = spec_peak - scale * spec_bg
    else:
        net_spec = spec_peak

    net_counts = float(np.sum(net_spec))
    elapsed_ms = round((time.time() - t0) * 1000.0, 1)

    return {
        "success": True,
        "plane": plane,
        "target_axis": target_axis,
        "net_spec": net_spec.tolist(),
        "peak_spec": spec_peak.tolist() if px_peak > 0 else [],
        "bg_spec": spec_bg.tolist() if has_bg else [],
        "total_gated_counts": int(round(net_counts)),
        "total_counts": int(round(net_counts)),
        "net_counts": net_counts,
        "pixel_count": px_peak,
        "pixel_count_peak": px_peak,
        "pixel_count_bg": px_bg,
        "area_peak": area_peak,
        "area_bg": area_bg,
        "counts_peak": cts_peak,
        "counts_bg": cts_bg,
        "scale": scale,
        "has_bg": has_bg,
        "polygon": polygon_peak,
        "polygon_peak": polygon_peak,
        "polygon_bg": polygon_bg,
        "elapsed_ms": elapsed_ms,
    }

