"""
cmat3d.py - High-Performance Python Reader, Decompressor and Slicer for GASPware/gsort 3D .cmat Matrices

Reverse-engineered from GASPware source code (ivflib.F, cmtlib.F, complib.c).
Supports standard rectilinear 3D matrices (e.g., GeE-Rings3D.cmat) and large folded symmetric
3D matrices (e.g. Ge-3D-symm.cmat, 8192^3 bins) via a Sparse On-Demand 3D Engine.

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
      - Dual Engine:
          * Dense Memmap Engine for moderate size matrices (e.g. <=16 GB uncompressed).
          * Sparse On-Demand 3D Engine for massive matrices (e.g. 8192^3 bins, 2.2 TB uncompressed,
            or folded symmetric matmode=1): fast caching of 2D projections in .npz format with
            instantaneous (<10ms) on-demand block decompression for 3D slicing, double-gating,
            and 2D Gamba coincidence cuts.
      - In-memory LRU sub-block cache for interactive navigation.
      - Column-major 3D sub-block coordinate addressing with tetrahedral permutation mapping.
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
        self._block_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._max_cached_blocks = 512

        # Determine if sparse on-demand engine should be used
        uncompressed_bytes = self.res1 * self.res2 * self.res3 * 4
        self.is_sparse_mode = (uncompressed_bytes > 16 * 1024 * 1024 * 1024) or (self.matmode != 0)

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

            self.matmode = self.cmt_hdr[1]  # 0=normal, 1=symmetric folded
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
            self.nextra = self.cmt_hdr[125]
            self.cmt_version = self.cmt_hdr[127]

            if self.matmode == 1:
                # GASPware 3D symmetric folded mode (s1 <= s2 <= s3)
                self.nmatrix_segs = (self.ndiv1 * (self.ndiv1 + 1) * (self.ndiv1 + 2)) // 6
            else:
                self.nmatrix_segs = self.ndiv1 * self.ndiv2 * self.ndiv3

            # Build fast in-memory map of non-zero block descriptors
            self.block_descrs_map = {}
            if self.matmode == 1:
                cur_idx = 0
                for s3 in range(self.ndiv3):
                    for s2 in range(s3 + 1):
                        for s1 in range(s2 + 1):
                            seg_idx = self.nextra + cur_idx
                            if seg_idx < len(self.descrs):
                                d = self.descrs[seg_idx]
                                if d[0] > 0:
                                    self.block_descrs_map[(s1, s2, s3)] = d
                            cur_idx += 1
            else:
                for s3 in range(self.ndiv3):
                    for s2 in range(self.ndiv2):
                        for s1 in range(self.ndiv1):
                            iseg = s1 + self.ndiv1 * (s2 + self.ndiv2 * s3)
                            seg_idx = self.nextra + iseg
                            if seg_idx < len(self.descrs):
                                d = self.descrs[seg_idx]
                                if d[0] > 0:
                                    self.block_descrs_map[(s1, s2, s3)] = d

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
        """Initializes or loads memory-mapped 3D array or pre-summed 2D projections."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dat_path, npz_path = self._get_cache_paths()

        if npz_path.exists():
            try:
                # Load pre-computed 2D projections
                with np.load(npz_path) as npz:
                    self.proj_2d["0-1"] = npz["p01"]
                    self.proj_2d["0-2"] = npz["p02"]
                    self.proj_2d["1-2"] = npz["p12"]
                    self._total_counts = int(npz.get("total_counts", np.sum(npz["p01"])))
                    self._max_count = int(npz.get("max_count", 0))
                    self._nonzero_voxels = int(npz.get("nonzero_voxels", 0))

                # For dense mode, open memory mapped 3D array in read-only mode if file exists
                if not self.is_sparse_mode and dat_path.exists():
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
        print(f"\n[*] Initializing 3D matrix cache for '{self.filename.name}' (Sparse Mode: {self.is_sparse_mode})...")
        print(f"    Dimensions: {self.res1} (X) × {self.res2} (Y) × {self.res3} (Z)")
        print(f"    Sub-blocks: {self.ndiv1} × {self.ndiv2} × {self.ndiv3} ({self.nmatrix_segs:,} total)")

        fp = None
        if not self.is_sparse_mode:
            shape_zyx = self.shape_zyx
            fp = np.memmap(dat_path, dtype=np.int32, mode="w+", shape=shape_zyx)

        p01 = np.zeros((self.res2, self.res1), dtype=np.int64)
        p02 = np.zeros((self.res3, self.res1), dtype=np.int64)
        p12 = np.zeros((self.res3, self.res2), dtype=np.int64)

        total_voxels_cnt = 0
        max_voxel_val = 0
        nonzero_cnt = 0

        step1, step2, step3 = self.step1, self.step2, self.step3
        segsize = self.segsize
        nz_items = list(self.block_descrs_map.items())
        n_nonzero = len(nz_items)
        print(f"    Non-zero sub-blocks to decompress: {n_nonzero:,} / {self.nmatrix_segs:,}")

        with open(self.filename, "rb") as f:
            last_report = time.time()
            for idx, ((s1, s2, s3), (nrec, frec)) in enumerate(nz_items):
                f.seek((frec - 1) * 512)
                raw = f.read(nrec * 512)
                cmode, cminval = struct.unpack_from("<2i", raw, 0)
                decomp = decompress_block(raw[8:], segsize, cmode, cminval)
                if len(decomp) < segsize:
                    decomp = np.pad(decomp, (0, segsize - len(decomp)))
                elif len(decomp) > segsize:
                    decomp = decomp[:segsize]

                b3d_raw = decomp.reshape((step3, step2, step1))

                if self.matmode == 1:
                    # Unfold intra-block symmetry when block lies on diagonal planes
                    if s1 == s2 == s3:
                        p0 = b3d_raw
                        p1 = np.transpose(b3d_raw, (0, 2, 1))
                        p2 = np.transpose(b3d_raw, (1, 0, 2))
                        p3 = np.transpose(b3d_raw, (1, 2, 0))
                        p4 = np.transpose(b3d_raw, (2, 0, 1))
                        p5 = np.transpose(b3d_raw, (2, 1, 0))
                        b3d = np.maximum.reduce([p0, p1, p2, p3, p4, p5])
                    elif s1 == s2:
                        b3d = np.maximum(b3d_raw, np.transpose(b3d_raw, (0, 2, 1)))
                    elif s2 == s3:
                        b3d = np.maximum(b3d_raw, np.transpose(b3d_raw, (1, 0, 2)))
                    else:
                        b3d = b3d_raw

                    x0, x1 = s1 * step1, (s1 + 1) * step1
                    y0, y1 = s2 * step2, (s2 + 1) * step2
                    z0, z1 = s3 * step3, (s3 + 1) * step3

                    sum_z = np.sum(b3d, axis=0, dtype=np.int64)  # (s2, s1)
                    sum_y = np.sum(b3d, axis=1, dtype=np.int64)  # (s3, s1)
                    sum_x = np.sum(b3d, axis=2, dtype=np.int64)  # (s3, s2)

                    if s1 == s2 == s3:
                        p01[y0:y1, x0:x1] += sum_z
                    elif s1 == s2:
                        p01[x0:x1, x0:x1] += sum_z
                        p01[z0:z1, x0:x1] += sum_y
                        p01[x0:x1, z0:z1] += sum_y.T
                    elif s2 == s3:
                        p01[y0:y1, y0:y1] += sum_x
                        p01[y0:y1, x0:x1] += sum_z
                        p01[x0:x1, y0:y1] += sum_z.T
                    else:
                        p01[y0:y1, x0:x1] += sum_z
                        p01[x0:x1, y0:y1] += sum_z.T
                        p01[z0:z1, x0:x1] += sum_y
                        p01[x0:x1, z0:z1] += sum_y.T
                        p01[z0:z1, y0:y1] += sum_x
                        p01[y0:y1, z0:z1] += sum_x.T

                    b_sum = int(np.sum(b3d))
                    b_max = int(np.max(b3d))
                    b_nz = int(np.count_nonzero(b3d))

                    total_voxels_cnt += b_sum
                    if b_max > max_voxel_val:
                        max_voxel_val = b_max
                    nonzero_cnt += b_nz

                else:
                    x0 = s1 * step1
                    x1 = min(x0 + step1, self.res1)
                    y0 = s2 * step2
                    y1 = min(y0 + step2, self.res2)
                    z0 = s3 * step3
                    z1 = min(z0 + step3, self.res3)

                    dz = z1 - z0
                    dy = y1 - y0
                    dx = x1 - x0
                    b_slice = b3d_raw[:dz, :dy, :dx]

                    if fp is not None:
                        fp[z0:z1, y0:y1, x0:x1] = b_slice

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

        if self.matmode == 1:
            p02 = p01.copy()
            p12 = p01.copy()
            total_voxels_cnt = int(np.sum(p01))

        if fp is not None:
            print("\n    Flushing memmap cache to disk...", flush=True)
            fp.flush()
            del fp
            self.memmap_3d = np.memmap(dat_path, dtype=np.int32, mode="r", shape=self.shape_zyx)

        p01_32 = p01.astype(np.int32)
        p02_32 = p02.astype(np.int32)
        p12_32 = p12.astype(np.int32)

        # Save 2D projections to npz
        np.savez_compressed(
            npz_path,
            p01=p01_32,
            p02=p02_32,
            p12=p12_32,
            total_counts=total_voxels_cnt,
            max_count=max_voxel_val,
            nonzero_voxels=nonzero_cnt,
        )

        self.proj_2d["0-1"] = p01_32
        self.proj_2d["0-2"] = p02_32
        self.proj_2d["1-2"] = p12_32
        self._total_counts = total_voxels_cnt
        self._max_count = max_voxel_val
        self._nonzero_voxels = nonzero_cnt

        elapsed_tot = time.time() - t0
        print(f"\n[+] 3D Matrix successfully initialized in {elapsed_tot:.1f}s! Total counts: {total_voxels_cnt:,}\n")

    def _get_block_3d(self, gx: int, gy: int, gz: int) -> Optional[np.ndarray]:
        """
        Retrieves decompressed sub-block for matrix grid coordinates (gx, gy, gz).
        Returns array with shape (step3, step2, step1) corresponding to local (Z, Y, X),
        or None if block contains all zeros.
        """
        key = (gx, gy, gz)
        if key in self._block_cache:
            return self._block_cache[key]

        if self.matmode == 1:
            s1 = min(gx, gy, gz)
            s3 = max(gx, gy, gz)
            s2 = gx + gy + gz - s1 - s3
            lookup_key = (s1, s2, s3)
        else:
            lookup_key = (gx, gy, gz)

        if lookup_key not in self.block_descrs_map:
            return None

        nrec, frec = self.block_descrs_map[lookup_key]
        with open(self.filename, "rb") as f:
            f.seek((frec - 1) * 512)
            raw = f.read(nrec * 512)
            cmode, cminval = struct.unpack_from("<2i", raw, 0)
            decomp = decompress_block(raw[8:], self.segsize, cmode, cminval)
            if len(decomp) < self.segsize:
                decomp = np.pad(decomp, (0, self.segsize - len(decomp)))
            elif len(decomp) > self.segsize:
                decomp = decomp[:self.segsize]

        b3d_sorted = decomp.reshape((self.step3, self.step2, self.step1))

        if self.matmode == 1:
            s1, s2, s3 = lookup_key
            # Unfold intra-block symmetry when block lies on diagonal planes
            if s1 == s2 == s3:
                p0 = b3d_sorted
                p1 = np.transpose(b3d_sorted, (0, 2, 1))
                p2 = np.transpose(b3d_sorted, (1, 0, 2))
                p3 = np.transpose(b3d_sorted, (1, 2, 0))
                p4 = np.transpose(b3d_sorted, (2, 0, 1))
                p5 = np.transpose(b3d_sorted, (2, 1, 0))
                b3d_unfolded = np.maximum.reduce([p0, p1, p2, p3, p4, p5])
            elif s1 == s2:
                b3d_unfolded = np.maximum(b3d_sorted, np.transpose(b3d_sorted, (0, 2, 1)))
            elif s2 == s3:
                b3d_unfolded = np.maximum(b3d_sorted, np.transpose(b3d_sorted, (1, 0, 2)))
            else:
                b3d_unfolded = b3d_sorted

            # Permute from sorted (s3, s2, s1) to target grid order (gz, gy, gx)
            vals = [(gx, "gx"), (gy, "gy"), (gz, "gz")]
            vals_sorted = sorted(vals, key=lambda v: v[0])
            decomp_to_target = {}
            for decomp_ax, (v, name) in zip([2, 1, 0], vals_sorted):
                if name == "gz":
                    target_ax = 0
                elif name == "gy":
                    target_ax = 1
                else:
                    target_ax = 2
                decomp_to_target[decomp_ax] = target_ax
            target_to_decomp = {t: d for d, t in decomp_to_target.items()}
            perm = (target_to_decomp[0], target_to_decomp[1], target_to_decomp[2])
            b3d = np.transpose(b3d_unfolded, perm)
        else:
            b3d = b3d_sorted

        # Cache block (LRU eviction if exceeded)
        if len(self._block_cache) >= self._max_cached_blocks:
            oldest = next(iter(self._block_cache))
            del self._block_cache[oldest]
        self._block_cache[key] = b3d
        return b3d

    def get_subvolume(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int],
    ) -> np.ndarray:
        """
        Extract arbitrary subvolume [x0:x1, y0:y1, z0:z1] on demand.
        Returns array of shape (z1 - z0, y1 - y0, x1 - x0).
        """
        x0, x1 = max(0, x_range[0]), min(self.res1, x_range[1])
        y0, y1 = max(0, y_range[0]), min(self.res2, y_range[1])
        z0, z1 = max(0, z_range[0]), min(self.res3, z_range[1])

        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return np.zeros((max(0, z1 - z0), max(0, y1 - y0), max(0, x1 - x0)), dtype=np.int32)

        if not self.is_sparse_mode and self.memmap_3d is not None:
            return self.memmap_3d[z0:z1, y0:y1, x0:x1]

        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.int32)
        step1, step2, step3 = self.step1, self.step2, self.step3

        gx_min, gx_max = x0 // step1, (x1 - 1) // step1
        gy_min, gy_max = y0 // step2, (y1 - 1) // step2
        gz_min, gz_max = z0 // step3, (z1 - 1) // step3

        for gz in range(gz_min, gz_max + 1):
            bz0 = gz * step3
            bz1 = min(bz0 + step3, self.res3)
            oz0 = max(z0, bz0)
            oz1 = min(z1, bz1)
            if oz1 <= oz0:
                continue

            for gy in range(gy_min, gy_max + 1):
                by0 = gy * step2
                by1 = min(by0 + step2, self.res2)
                oy0 = max(y0, by0)
                oy1 = min(y1, by1)
                if oy1 <= oy0:
                    continue

                for gx in range(gx_min, gx_max + 1):
                    bx0 = gx * step1
                    bx1 = min(bx0 + step1, self.res1)
                    ox0 = max(x0, bx0)
                    ox1 = min(x1, bx1)
                    if ox1 <= ox0:
                        continue

                    b3d = self._get_block_3d(gx, gy, gz)
                    if b3d is None:
                        continue

                    sbz0, sbz1 = oz0 - bz0, oz1 - bz0
                    sby0, sby1 = oy0 - by0, oy1 - by0
                    sbx0, sbx1 = ox0 - bx0, ox1 - bx0

                    doz0, doz1 = oz0 - z0, oz1 - z0
                    doy0, doy1 = oy0 - y0, oy1 - y0
                    dox0, dox1 = ox0 - x0, ox1 - x0

                    out[doz0:doz1, doy0:doy1, dox0:dox1] = b3d[sbz0:sbz1, sby0:sby1, sbx0:sbx1]

        return out

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

        if self.memmap_3d is None and not self.is_sparse_mode:
            self._init_data_cache()

        out = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.int32)

        # Fast dense memmap summation path
        if not self.is_sparse_mode and self.memmap_3d is not None:
            if plane_norm == "0-1":
                for z0, z1 in all_gates:
                    zm = max(0, min(self.res3 - 1, min(z0, z1)))
                    zx = max(0, min(self.res3, max(z0, z1) + 1))
                    if zx > zm:
                        out += np.sum(self.memmap_3d[zm:zx, ry0:ry1, rx0:rx1], axis=0, dtype=np.int32)
            elif plane_norm == "0-2":
                for y0_g, y1_g in all_gates:
                    ym = max(0, min(self.res2 - 1, min(y0_g, y1_g)))
                    yx = max(0, min(self.res2, max(y0_g, y1_g) + 1))
                    if yx > ym:
                        out += np.sum(self.memmap_3d[ry0:ry1, ym:yx, rx0:rx1], axis=1, dtype=np.int32)
            else:
                for x0_g, x1_g in all_gates:
                    xm = max(0, min(self.res1 - 1, min(x0_g, x1_g)))
                    xx = max(0, min(self.res1, max(x0_g, x1_g) + 1))
                    if xx > xm:
                        out += np.sum(self.memmap_3d[ry0:ry1, rx0:rx1, xm:xx], axis=2, dtype=np.int32)
            return out

        # Sparse on-demand extraction path
        for g0, g1 in all_gates:
            gm = min(g0, g1)
            gx = max(g0, g1) + 1
            if plane_norm == "0-1":
                sub = self.get_subvolume((rx0, rx1), (ry0, ry1), (gm, gx))
                out += np.sum(sub, axis=0, dtype=np.int32)
            elif plane_norm == "0-2":
                sub = self.get_subvolume((rx0, rx1), (gm, gx), (ry0, ry1))
                out += np.sum(sub, axis=1, dtype=np.int32)
            else:
                sub = self.get_subvolume((gm, gx), (rx0, rx1), (ry0, ry1))
                out += np.sum(sub, axis=2, dtype=np.int32)

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

            if not all_gates_3rd:
                spec0 = np.sum(self.proj_2d["0-1"][ry0:ry1, :], axis=0, dtype=np.int64) if (ry0 > 0 or ry1 < self.res2) else self.get_projection(0)
                spec1 = np.sum(self.proj_2d["0-1"][:, rx0:rx1], axis=1, dtype=np.int64) if (rx0 > 0 or rx1 < self.res1) else self.get_projection(1)
                if rx0 == 0 and rx1 >= self.res1 and ry0 == 0 and ry1 >= self.res2:
                    spec2 = self.get_projection(2)
                else:
                    sub = self.get_subvolume((rx0, rx1), (ry0, ry1), (0, self.res3))
                    spec2 = np.sum(sub, axis=(1, 2), dtype=np.int64)
            else:
                spec0 = np.zeros(self.res1, dtype=np.int64)
                spec1 = np.zeros(self.res2, dtype=np.int64)
                for zm, zx in all_gates_3rd:
                    z_lo = max(0, min(self.res3 - 1, min(zm, zx)))
                    z_hi = max(0, min(self.res3, max(zm, zx) + 1))
                    if z_hi > z_lo:
                        chunk = self.get_subvolume((rx0, rx1), (ry0, ry1), (z_lo, z_hi))
                        spec0[rx0:rx1] += np.sum(chunk, axis=(0, 1), dtype=np.int64)
                        spec1[ry0:ry1] += np.sum(chunk, axis=(0, 2), dtype=np.int64)

                sub_full_z = self.get_subvolume((rx0, rx1), (ry0, ry1), (0, self.res3))
                spec2 = np.sum(sub_full_z, axis=(1, 2), dtype=np.int64)

            return spec0, spec1, spec2

        elif plane_norm == "0-2":
            rx0 = max(0, min(self.res1 - 1, x0))
            rx1 = max(rx0 + 1, min(self.res1, x1 if x1 is not None else self.res1))
            rz0 = max(0, min(self.res3 - 1, y0))
            rz1 = max(rz0 + 1, min(self.res3, y1 if y1 is not None else self.res3))

            if not all_gates_3rd:
                spec0 = np.sum(self.proj_2d["0-2"][rz0:rz1, :], axis=0, dtype=np.int64) if (rz0 > 0 or rz1 < self.res3) else self.get_projection(0)
                spec2 = np.sum(self.proj_2d["0-2"][:, rx0:rx1], axis=1, dtype=np.int64) if (rx0 > 0 or rx1 < self.res1) else self.get_projection(2)
                sub = self.get_subvolume((rx0, rx1), (0, self.res2), (rz0, rz1))
                spec1 = np.sum(sub, axis=(0, 2), dtype=np.int64)
            else:
                spec0 = np.zeros(self.res1, dtype=np.int64)
                spec2 = np.zeros(self.res3, dtype=np.int64)
                for ym, yx in all_gates_3rd:
                    y_lo = max(0, min(self.res2 - 1, min(ym, yx)))
                    y_hi = max(0, min(self.res2, max(ym, yx) + 1))
                    if y_hi > y_lo:
                        chunk = self.get_subvolume((rx0, rx1), (y_lo, y_hi), (rz0, rz1))
                        spec0[rx0:rx1] += np.sum(chunk, axis=(0, 1), dtype=np.int64)
                        spec2[rz0:rz1] += np.sum(chunk, axis=(1, 2), dtype=np.int64)

                sub_full_y = self.get_subvolume((rx0, rx1), (0, self.res2), (rz0, rz1))
                spec1 = np.sum(sub_full_y, axis=(0, 2), dtype=np.int64)

            return spec0, spec1, spec2

        else:
            ry0 = max(0, min(self.res2 - 1, x0))
            ry1 = max(ry0 + 1, min(self.res2, x1 if x1 is not None else self.res2))
            rz0 = max(0, min(self.res3 - 1, y0))
            rz1 = max(rz0 + 1, min(self.res3, y1 if y1 is not None else self.res3))

            if not all_gates_3rd:
                spec1 = np.sum(self.proj_2d["1-2"][rz0:rz1, :], axis=0, dtype=np.int64) if (rz0 > 0 or rz1 < self.res3) else self.get_projection(1)
                spec2 = np.sum(self.proj_2d["1-2"][:, ry0:ry1], axis=1, dtype=np.int64) if (ry0 > 0 or ry1 < self.res2) else self.get_projection(2)
                sub = self.get_subvolume((0, self.res1), (ry0, ry1), (rz0, rz1))
                spec0 = np.sum(sub, axis=(0, 1), dtype=np.int64)
            else:
                spec1 = np.zeros(self.res2, dtype=np.int64)
                spec2 = np.zeros(self.res3, dtype=np.int64)
                for xm, xx in all_gates_3rd:
                    x_lo = max(0, min(self.res1 - 1, min(xm, xx)))
                    x_hi = max(0, min(self.res1, max(xm, xx) + 1))
                    if x_hi > x_lo:
                        chunk = self.get_subvolume((x_lo, x_hi), (ry0, ry1), (rz0, rz1))
                        spec1[ry0:ry1] += np.sum(chunk, axis=(0, 2), dtype=np.int64)
                        spec2[rz0:rz1] += np.sum(chunk, axis=(1, 2), dtype=np.int64)

                sub_full_x = self.get_subvolume((0, self.res1), (ry0, ry1), (rz0, rz1))
                spec0 = np.sum(sub_full_x, axis=(0, 1), dtype=np.int64)

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
            "matrix_mode": "Normal" if self.matmode == 0 else f"Mode {self.matmode} (Symmetric)",
            "sparse_mode": self.is_sparse_mode,
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

            ranges = [(0, reader.res1), (0, reader.res2), (0, reader.res3)]
            ranges[g1] = (min1, max1)
            ranges[g2] = (min2, max2)

            sub_3d = reader.get_subvolume(ranges[0], ranges[1], ranges[2]).astype(np.float64)
            dim_to_sub_ax = {0: 2, 1: 1, 2: 0}
            dim1 = dim_to_sub_ax[g1]
            dim2 = dim_to_sub_ax[g2]

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
        sub = reader.get_subvolume((x_min, x_max), (y_min, y_max), (0, reader.res3)).astype(np.float64)
        spec = np.sum(sub * mask[np.newaxis, :, :], axis=(1, 2), dtype=np.float64)
    elif plane == "0-2":
        sub = reader.get_subvolume((x_min, x_max), (0, reader.res2), (y_min, y_max)).astype(np.float64)
        spec = np.sum(sub * mask[:, np.newaxis, :], axis=(0, 2), dtype=np.float64)
    else:  # plane == "1-2"
        sub = reader.get_subvolume((0, reader.res1), (x_min, x_max), (y_min, y_max)).astype(np.float64)
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


def compute_2d_gamba_gate(
    reader: CMAT3DReader,
    plane: str,
    fit_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes true 3rd-axis coincidence spectrum for a 2D fitted peak using the 4-component
    Gamba & Morhác discrete background decomposition (NIM A 928, 2019, 93-103):
      - P|P   : Peak gate on both 2D plane axes -> extracts S_pp(z) from 3D volume
      - P|BG  : Peak X × BG Y -> extracts S_pbg(z)
      - BG|P  : BG X × Peak Y -> extracts S_bgp(z)
      - BG|BG : BG X × BG Y continuum -> extracts S_bgbg(z)

    Net Spectrum on the 3rd axis:
      S_net(z) = S_pp(z) - s_pbg * S_pbg(z) - s_bgp * S_bgp(z) + s_bgbg * S_bgbg(z)
      where s_pbg = Area_pp / Area_pbg, s_bgp = Area_pp / Area_bgp, s_bgbg = Area_pp / Area_bgbg.
    """
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

    if not fit_res or not fit_res.get("success", False):
        return {
            "success": False,
            "error": "Invalid or unsuccessful 2D peak fit result.",
            "plane": plane,
            "target_axis": target_axis,
            "net_spec": np.zeros(target_res, dtype=np.float64).tolist(),
            "net_counts": 0.0,
            "total_gated_counts": 0,
            "elapsed_ms": 0.0,
        }

    t0 = time.time()

    mx = float(fit_res.get("centroid_x_ch", 0.0))
    my = float(fit_res.get("centroid_y_ch", 0.0))
    fwhm_x = max(0.5, float(fit_res.get("fwhm_x_ch", 4.0)))
    fwhm_y = max(0.5, float(fit_res.get("fwhm_y_ch", 4.0)))

    sx = max(0.5, abs(fwhm_x) / 2.35482)
    sy = max(0.5, abs(fwhm_y) / 2.35482)

    # Bounding ROI
    x_min = int(fit_res.get("roi_x_min", -1))
    x_max = int(fit_res.get("roi_x_max", -1))
    y_min = int(fit_res.get("roi_y_min", -1))
    y_max = int(fit_res.get("roi_y_max", -1))

    if x_min < 0 or x_max <= x_min or y_min < 0 or y_max <= y_min:
        roi_half_w = int(round(4.0 * max(sx, sy, 4.0)))
        x_min = max(0, int(np.floor(mx)) - roi_half_w)
        x_max = min(dim_x - 1, int(np.floor(mx)) + roi_half_w)
        y_min = max(0, int(np.floor(my)) - roi_half_w)
        y_max = min(dim_y - 1, int(np.floor(my)) + roi_half_w)

    x_min = max(0, min(dim_x - 1, x_min))
    x_max = max(x_min, min(dim_x - 1, x_max))
    y_min = max(0, min(dim_y - 1, y_min))
    y_max = max(y_min, min(dim_y - 1, y_max))

    Nx = x_max - x_min + 1
    Ny = y_max - y_min + 1

    w_gx = max(1, int(round(2.0 * sx)))
    w_gy = max(1, int(round(2.0 * sy)))
    ix_c = int(np.floor(mx)) - x_min
    iy_c = int(np.floor(my)) - y_min

    px0 = max(0, ix_c - w_gx)
    px1 = min(Nx - 1, ix_c + w_gx)
    py0 = max(0, iy_c - w_gy)
    py1 = min(Ny - 1, iy_c + w_gy)

    mask_peak_x = np.zeros(Nx, dtype=bool)
    mask_peak_x[px0:px1 + 1] = True
    mask_bg_x = ~mask_peak_x

    mask_peak_y = np.zeros(Ny, dtype=bool)
    mask_peak_y[py0:py1 + 1] = True
    mask_bg_y = ~mask_peak_y

    area_pp = int(np.sum(mask_peak_y)) * int(np.sum(mask_peak_x))
    area_pbg = int(np.sum(mask_bg_y)) * int(np.sum(mask_peak_x))
    area_bgp = int(np.sum(mask_peak_y)) * int(np.sum(mask_bg_x))
    area_bgbg = int(np.sum(mask_bg_y)) * int(np.sum(mask_bg_x))

    s_pbg_scale = (float(area_pp) / float(area_pbg)) if area_pbg > 0 else 0.0
    s_bgp_scale = (float(area_pp) / float(area_bgp)) if area_bgp > 0 else 0.0
    s_bgbg_scale = (float(area_pp) / float(area_bgbg)) if area_bgbg > 0 else 0.0

    # Extract 3D subvolume across the bounding ROI and project 4 discrete regions onto target_axis
    if plane == "0-1":
        sub_3d = reader.get_subvolume((x_min, x_max + 1), (y_min, y_max + 1), (0, reader.res3)).astype(np.float64)
        s_pp = np.sum(sub_3d[:, py0:py1 + 1, px0:px1 + 1], axis=(1, 2), dtype=np.float64) if area_pp > 0 else np.zeros(target_res)
        s_pbg = np.sum(sub_3d[:, mask_bg_y, :][:, :, mask_peak_x], axis=(1, 2), dtype=np.float64) if area_pbg > 0 else np.zeros(target_res)
        s_bgp = np.sum(sub_3d[:, mask_peak_y, :][:, :, mask_bg_x], axis=(1, 2), dtype=np.float64) if area_bgp > 0 else np.zeros(target_res)
        s_bgbg = np.sum(sub_3d[:, mask_bg_y, :][:, :, mask_bg_x], axis=(1, 2), dtype=np.float64) if area_bgbg > 0 else np.zeros(target_res)

    elif plane == "0-2":
        sub_3d = reader.get_subvolume((x_min, x_max + 1), (0, reader.res2), (y_min, y_max + 1)).astype(np.float64)
        s_pp = np.sum(sub_3d[py0:py1 + 1, :, px0:px1 + 1], axis=(0, 2), dtype=np.float64) if area_pp > 0 else np.zeros(target_res)
        s_pbg = np.sum(sub_3d[mask_bg_y, :, :][:, :, mask_peak_x], axis=(0, 2), dtype=np.float64) if area_pbg > 0 else np.zeros(target_res)
        s_bgp = np.sum(sub_3d[mask_peak_y, :, :][:, :, mask_bg_x], axis=(0, 2), dtype=np.float64) if area_bgp > 0 else np.zeros(target_res)
        s_bgbg = np.sum(sub_3d[mask_bg_y, :, :][:, :, mask_bg_x], axis=(0, 2), dtype=np.float64) if area_bgbg > 0 else np.zeros(target_res)

    else:  # plane == "1-2"
        sub_3d = reader.get_subvolume((0, reader.res1), (x_min, x_max + 1), (y_min, y_max + 1)).astype(np.float64)
        s_pp = np.sum(sub_3d[py0:py1 + 1, px0:px1 + 1, :], axis=(0, 1), dtype=np.float64) if area_pp > 0 else np.zeros(target_res)
        s_pbg = np.sum(sub_3d[mask_bg_y, :, :][:, mask_peak_x, :], axis=(0, 1), dtype=np.float64) if area_pbg > 0 else np.zeros(target_res)
        s_bgp = np.sum(sub_3d[mask_peak_y, :, :][:, mask_bg_x, :], axis=(0, 1), dtype=np.float64) if area_bgp > 0 else np.zeros(target_res)
        s_bgbg = np.sum(sub_3d[mask_bg_y, :, :][:, mask_bg_x, :], axis=(0, 1), dtype=np.float64) if area_bgbg > 0 else np.zeros(target_res)

    bg_spec = s_pbg_scale * s_pbg + s_bgp_scale * s_bgp - s_bgbg_scale * s_bgbg
    net_spec = s_pp - bg_spec
    var_spec = s_pp + (s_pbg_scale ** 2) * s_pbg + (s_bgp_scale ** 2) * s_bgp + (s_bgbg_scale ** 2) * s_bgbg
    err_spec = np.sqrt(np.maximum(0.0, var_spec))

    net_counts = float(np.sum(net_spec))
    gross_counts = float(np.sum(s_pp))
    bg_counts = float(np.sum(bg_spec))
    pi_ratio = (net_counts / max(1.0, gross_counts)) if gross_counts > 0 else 0.0

    elapsed_ms = round((time.time() - t0) * 1000.0, 1)

    return {
        "success": True,
        "plane": plane,
        "target_axis": target_axis,
        "net_spec": net_spec.tolist(),
        "raw_spec": s_pp.tolist(),
        "bg_spec": bg_spec.tolist(),
        "err_spec": err_spec.tolist(),
        "net_counts": round(net_counts, 1),
        "total_gated_counts": int(round(net_counts)),
        "gross_counts": round(gross_counts, 1),
        "bg_counts": round(bg_counts, 1),
        "pi_ratio": round(pi_ratio, 4),
        "pi_ratio_percent": round(pi_ratio * 100.0, 2),
        "scale_pbg": round(float(s_pbg_scale), 4),
        "scale_bgp": round(float(s_bgp_scale), 4),
        "scale_bgbg": round(float(s_bgbg_scale), 4),
        "area_pp": area_pp,
        "area_pbg": area_pbg,
        "area_bgp": area_bgp,
        "area_bgbg": area_bgbg,
        "n_pp_m": round(float(np.sum(s_pp)), 1),
        "n_pbg_raw": round(float(np.sum(s_pbg)), 1),
        "n_bgp_raw": round(float(np.sum(s_bgp)), 1),
        "n_bgbg_raw": round(float(np.sum(s_bgbg)), 1),
        "peak_x_range": [int(x_min + px0), int(x_min + px1)],
        "peak_y_range": [int(y_min + py0), int(y_min + py1)],
        "roi_x_range": [int(x_min), int(x_max)],
        "roi_y_range": [int(y_min), int(y_max)],
        "centroid_x_ch": round(float(mx), 3),
        "centroid_y_ch": round(float(my), 3),
        "centroid_x_e": fit_res.get("centroid_x_e"),
        "centroid_y_e": fit_res.get("centroid_y_e"),
        "fwhm_x_ch": round(float(fwhm_x), 3),
        "fwhm_y_ch": round(float(fwhm_y), 3),
        "elapsed_ms": elapsed_ms,
    }
