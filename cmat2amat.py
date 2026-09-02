#!/usr/bin/env python3
"""
cmat2amat.py - Command-line tool to inspect and convert GASPware .cmat matrices to ASCII (.amat).

Usage examples:
    # Print matrix info
    python3 cmat2amat.py GeE-symm.cmat --info

    # Convert to dense ASCII matrix (.amat)
    python3 cmat2amat.py GeE-symm.cmat -o GeE-symm.amat

    # Convert to sparse ASCII (x y counts for non-zero channels)
    python3 cmat2amat.py GeE-symm.cmat -o GeE-symm_sparse.amat --format sparse

    # Convert a region of interest (e.g. 100 to 1500 keV)
    python3 cmat2amat.py GeE-symm.cmat -o GeE_roi.amat --range-x 100 1500 --range-y 100 1500

    # Save as NumPy array (.npy) for fast loading
    python3 cmat2amat.py GeE-symm.cmat --npy
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from cmat import CMATReader


def main():
    parser = argparse.ArgumentParser(
        description="Convert and inspect GASPware/gsort .cmat compressed matrices."
    )
    parser.add_argument("input", type=str, help="Path to input .cmat file")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Path to output .amat file (default: <input_base>.amat)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["dense", "sparse"],
        default="dense",
        help="ASCII format type: 'dense' (2D matrix grid) or 'sparse' (x y count list)",
    )
    parser.add_argument(
        "-d", "--delimiter", type=str, default=" ", help="Column delimiter for ASCII output (default: space)"
    )
    parser.add_argument(
        "--no-header", action="store_true", help="Omit comments/metadata header from ASCII output"
    )
    parser.add_argument(
        "--range-x",
        nargs=2,
        type=int,
        metavar=("XMIN", "XMAX"),
        help="Sub-range of channels on X axis (inclusive)",
    )
    parser.add_argument(
        "--range-y",
        nargs=2,
        type=int,
        metavar=("YMIN", "YMAX"),
        help="Sub-range of channels on Y axis (inclusive)",
    )
    parser.add_argument(
        "--info", action="store_true", help="Display matrix metadata, header info, and statistics without exporting"
    )
    parser.add_argument(
        "--projection", action="store_true", help="Export 1D total projection spectrum to ASCII (<input_base>_proj.dat)"
    )
    parser.add_argument(
        "--npy", action="store_true", help="Save the decompressed matrix as a NumPy binary file (.npy)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Reading CMAT: {input_path}")
    reader = CMATReader(input_path)
    info = reader.get_info()

    if args.info:
        print("\n=== CMAT Metadata ===")
        for k, v in info.items():
            print(f"  {k:18s}: {v}")

        # Compute projection stats
        proj = reader.get_projection()
        print("\n=== Stored 1D Projection Statistics ===")
        print(f"  Channels          : {len(proj)}")
        print(f"  Total Integral    : {np.sum(proj):,}")
        print(f"  Peak Channel      : {np.argmax(proj)} (Counts: {np.max(proj):,})")
        print(f"  Non-zero Channels : {np.count_nonzero(proj):,} ({100 * np.count_nonzero(proj)/len(proj):.2f}%)")
        return

    # Decompress matrix
    t0 = time.time()
    print("[*] Decompressing matrix...")
    mat = reader.to_numpy()
    t_decomp = time.time() - t0
    total_counts = np.sum(mat)
    nonzero_bins = np.count_nonzero(mat)
    print(
        f"[*] Decompressed {mat.shape[0]}x{mat.shape[1]} matrix in {t_decomp:.2f}s "
        f"(Total counts: {total_counts:,}, Non-zero bins: {nonzero_bins:,})"
    )

    # Save NumPy array if requested
    if args.npy:
        npy_path = input_path.with_suffix(".npy")
        print(f"[*] Saving NumPy matrix to: {npy_path}")
        np.save(npy_path, mat)

    # Save projection if requested
    if args.projection:
        proj_path = input_path.with_name(f"{input_path.stem}_proj.dat")
        proj = reader.get_projection()
        print(f"[*] Exporting 1D projection to: {proj_path}")
        np.savetxt(proj_path, proj, fmt="%d", header=f"1D Projection of {input_path.name}")

    # Export AMAT
    out_path = Path(args.output) if args.output else input_path.with_suffix(".amat")
    print(f"[*] Exporting to ASCII ({args.format} format): {out_path} ...")
    t0 = time.time()
    reader.export_amat(
        output_file=out_path,
        format_type=args.format,
        delimiter=args.delimiter,
        header=not args.no_header,
        x_range=tuple(args.range_x) if args.range_x else None,
        y_range=tuple(args.range_y) if args.range_y else None,
    )
    t_export = time.time() - t0
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully exported {out_path} ({file_size_mb:.2f} MB in {t_export:.2f}s)")


if __name__ == "__main__":
    main()
