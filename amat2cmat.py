#!/usr/bin/env python3
"""
amat2cmat.py - Command-line tool to convert ASCII matrices (.amat, .dat, .txt, .csv)
or NumPy arrays (.npy) into GASPware/gsort compressed .cmat format.

Inverse tool of cmat2amat.py.

Usage examples:
    # Convert dense ASCII matrix (.amat) to .cmat:
    python3 amat2cmat.py GeE-symm.amat -o GeE-symm_reconstructed.cmat

    # Convert sparse ASCII matrix (x y counts) to .cmat with explicit dimensions:
    python3 amat2cmat.py GeE_sparse.amat -o GeE_sparse.cmat --shape 4096 4096

    # Convert NumPy array (.npy) to symmetric .cmat:
    python3 amat2cmat.py matrix.npy -o matrix.cmat --symmetric

    # Inspect ASCII matrix statistics without exporting:
    python3 amat2cmat.py GeE-symm.amat --info
"""

import argparse
import sys
import time
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from cmat import write_cmat, CMATReader


def parse_ascii_header(filepath: Path) -> dict:
    """Parse comments and metadata from header of an ASCII matrix file."""
    meta = {
        "format": None,
        "shape": None,
        "full_shape": None,
        "source": None,
        "comments": [],
    }
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith(("#", "!", ";", "//", "%")):
                meta["comments"].append(line_str)
                # Match Sub-Matrix Shape or Full Dimensions
                m_shape = re.search(r"(?:Sub-Matrix Shape|Full Dimensions|Dimensions|Shape)[:=]\s*(\d+)\s*(?:[xX,]\s*|\s+)(\d+)", line_str)
                if m_shape:
                    meta["shape"] = (int(m_shape.group(1)), int(m_shape.group(2)))
                m_fmt = re.search(r"Format[:=]\s*(\w+)", line_str, re.IGNORECASE)
                if m_fmt:
                    meta["format"] = m_fmt.group(1).lower()
                m_src = re.search(r"Source[:=]\s*(\S+)", line_str)
                if m_src:
                    meta["source"] = m_src.group(1)
            else:
                break
    return meta


def load_ascii_matrix(
    filepath: Path,
    format_type: str = "auto",
    delimiter: Optional[str] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Load an ASCII matrix file into a 2D int32 NumPy array.
    Returns (matrix, detected_format).
    """
    meta = parse_ascii_header(filepath)

    # If shape not given in CLI, check header metadata
    if shape is None and meta.get("shape") is not None:
        shape = meta["shape"]

    # Detect format if auto
    if format_type == "auto":
        if meta.get("format") in ("dense", "sparse"):
            format_type = meta["format"]
        else:
            # Peek first non-comment line
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s and not line_s.startswith(("#", "!", ";", "//", "%")):
                        tokens = line_s.split(delimiter) if delimiter else line_s.split()
                        if len(tokens) == 3:
                            format_type = "sparse"
                        else:
                            format_type = "dense"
                        break

    if format_type not in ("dense", "sparse"):
        format_type = "dense"

    if format_type == "sparse":
        # Load sparse x, y, counts
        x_coords = []
        y_coords = []
        counts = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_s = line.strip()
                if not line_s or line_s.startswith(("#", "!", ";", "//", "%")):
                    continue
                tokens = line_s.split(delimiter) if delimiter else line_s.split()
                if len(tokens) >= 3:
                    try:
                        x = int(float(tokens[0]))
                        y = int(float(tokens[1]))
                        c = int(float(tokens[2]))
                        x_coords.append(x)
                        y_coords.append(y)
                        counts.append(c)
                    except ValueError:
                        continue

        if not x_coords:
            raise ValueError(f"No valid sparse coordinate data found in {filepath}")

        max_x = max(x_coords)
        max_y = max(y_coords)

        dim_x = shape[0] if shape is not None else (max_x + 1)
        dim_y = shape[1] if shape is not None else (max_y + 1)

        mat = np.zeros((dim_y, dim_x), dtype=np.int32)
        for x, y, c in zip(x_coords, y_coords, counts):
            if y < dim_y and x < dim_x:
                mat[y, x] = c

        return mat, "sparse"

    else:
        # Dense matrix grid
        rows = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_s = line.strip()
                if not line_s or line_s.startswith(("#", "!", ";", "//", "%")):
                    continue
                tokens = line_s.split(delimiter) if delimiter else line_s.split()
                try:
                    row = [int(float(t)) for t in tokens]
                    rows.append(row)
                except ValueError:
                    continue

        if not rows:
            raise ValueError(f"No valid dense matrix data found in {filepath}")

        mat = np.array(rows, dtype=np.int32)
        if shape is not None and (mat.shape[1] != shape[0] or mat.shape[0] != shape[1]):
            # Pad or truncate if explicit shape given
            target_y, target_x = shape[1], shape[0]
            new_mat = np.zeros((target_y, target_x), dtype=np.int32)
            copy_y = min(mat.shape[0], target_y)
            copy_x = min(mat.shape[1], target_x)
            new_mat[:copy_y, :copy_x] = mat[:copy_y, :copy_x]
            mat = new_mat

        return mat, "dense"


def main():
    parser = argparse.ArgumentParser(
        description="Convert ASCII matrix (.amat, .dat, .txt) or NumPy (.npy) to GASPware .cmat format."
    )
    parser.add_argument("input", type=str, help="Path to input ASCII file (.amat, .dat, .txt) or .npy")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Path to output .cmat file (default: <input_base>.cmat)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["auto", "dense", "sparse"],
        default="auto",
        help="ASCII format type: 'auto' (default), 'dense' (2D matrix grid), or 'sparse' (x y counts list)",
    )
    parser.add_argument(
        "-d", "--delimiter", type=str, default=None, help="Column delimiter for ASCII input (default: auto whitespace/comma)"
    )
    parser.add_argument(
        "--shape",
        "--dim",
        nargs="+",
        type=int,
        metavar=("NX", "NY"),
        help="Target matrix dimensions Nx [Ny] (e.g. --shape 4096 4096). Inferred if omitted.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=None,
        help="Force symmetric matrix storage (Mode 1: lower triangle only)",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        default=False,
        help="Force normal / asymmetric matrix storage (Mode 0: full grid)",
    )
    parser.add_argument(
        "--step",
        nargs="+",
        type=int,
        default=[128, 128],
        metavar=("STEP1", "STEP2"),
        help="Sub-block step size (default: 128 128)",
    )
    parser.add_argument(
        "--info", action="store_true", help="Display matrix metadata, dimensions, and statistics without exporting"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f"[*] Reading input: {input_path}")

    target_shape = None
    if args.shape:
        if len(args.shape) == 1:
            target_shape = (args.shape[0], args.shape[0])
        else:
            target_shape = (args.shape[0], args.shape[1])

    step1 = args.step[0]
    step2 = args.step[1] if len(args.step) > 1 else args.step[0]

    if input_path.suffix.lower() == ".npy":
        mat = np.load(input_path).astype(np.int32)
        fmt = "numpy"
    else:
        mat, fmt = load_ascii_matrix(
            filepath=input_path,
            format_type=args.format,
            delimiter=args.delimiter,
            shape=target_shape,
        )

    t_read = time.time() - t0
    res_y, res_x = mat.shape
    total_counts = int(np.sum(mat))
    nonzero_bins = int(np.count_nonzero(mat))
    is_square = (res_x == res_y)
    is_symm = is_square and np.array_equal(mat, mat.T)

    print(
        f"[*] Loaded {res_x}x{res_y} matrix ({fmt} format) in {t_read:.2f}s "
        f"(Total counts: {total_counts:,}, Non-zero bins: {nonzero_bins:,})"
    )

    if args.info:
        print("\n=== Matrix Analysis & Statistics ===")
        print(f"  Dimensions (X x Y) : {res_x} x {res_y}")
        print(f"  Symmetric Data     : {'Yes (M == M.T)' if is_symm else 'No'}")
        print(f"  Total Integral     : {total_counts:,}")
        print(f"  Max Bin Count      : {int(np.max(mat)):,}")
        print(f"  Non-zero Bins      : {nonzero_bins:,} ({100 * nonzero_bins / (res_x * res_y):.3f}%)")
        print(f"  Sub-Block Step     : {step1} x {step2}")
        print(f"  Grid Divisions     : {(res_x + step1 - 1)//step1} x {(res_y + step2 - 1)//step2}")
        return

    # Determine symmetry mode
    if args.asymmetric:
        symmetric_flag = False
    elif args.symmetric:
        symmetric_flag = True
    else:
        symmetric_flag = is_symm

    out_path = Path(args.output) if args.output else input_path.with_suffix(".cmat")
    print(f"[*] Compressing and writing CMAT ({'Symmetric' if symmetric_flag else 'Normal'} mode): {out_path} ...")

    t1 = time.time()
    write_cmat(
        output_file=out_path,
        matrix=mat,
        symmetric=symmetric_flag,
        step1=step1,
        step2=step2,
    )
    t_write = time.time() - t1

    # Verify written file with CMATReader
    reader = CMATReader(out_path)
    info = reader.get_info()
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"[+] Successfully generated {out_path} ({file_size_mb:.2f} MB in {t_write:.2f}s)")
    print(
        f"    Verified CMAT: Mode={info['matrix_mode']}, Dimensions={info['shape'][0]}x{info['shape'][1]}, "
        f"Segments={info['total_segments']} ({info['matrix_segments']} matrix + {info['extra_segments']} header)"
    )


if __name__ == "__main__":
    main()
