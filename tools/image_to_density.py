#!/usr/bin/env python3
"""Converts a grayscale (or color, auto-desaturated) image into the text
density format tools/export_density.cu's --f0/--f1 flags (and its
load_density_text loader) already accept, so a real photo can be used as an
initial or final density instead of only the built-in Gaussians or a
hand-drawn tools/paint_density.py canvas.

File format written: a text file containing the header 'N P' followed by
(N+1)*(P+1) whitespace-separated doubles, row-major i*(P+1)+j. Values are
written un-normalized (raw 0-255 grayscale, optionally inverted);
export_density's loader already Riemann-sum-renormalizes to unit mass on
load, so this tool doesn't need to.

Usage:
    python tools/image_to_density.py photo0.jpg f0.txt --N 64
    python tools/image_to_density.py photo1.jpg f1.txt --N 64 --invert

Then, as usual:
    ./build/Release/export_density --f0 f0.txt --f1 f1.txt --max-iter 20000 --out frames.bin
    python tools/make_animation.py frames.bin transport.gif
"""

import argparse
import sys

from PIL import Image


def image_to_grid(image_path, N, P, invert):
    """Loads an image, desaturates it, and stretch-resizes it to an
    (N+1)x(P+1) grid (ignoring the original aspect ratio -- this project's
    domain is a plain [0,1]^2 grid with no notion of "aspect" to preserve).
    Returns grid[i][j], i=0..N (x), j=0..P (y), matching the convention
    every other centered-grid array in this project uses.
    """
    img = Image.open(image_path).convert("L")  # "L" = 8-bit grayscale
    img = img.resize((N + 1, P + 1), Image.LANCZOS)
    pixels = img.load()

    grid = [[0.0] * (P + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(P + 1):
            value = float(pixels[i, P - j])
            if invert:
                value = 255.0 - value
            grid[i][j] = value
    return grid


def save_density_text(path, N, P, grid):
    """Writes exactly the format load_density_text (tools/export_density.cu)
    parses."""
    with open(path, "w") as f:
        f.write(f"{N} {P}\n")
        for i in range(N + 1):
            f.write(" ".join(f"{v:.10g}" for v in grid[i]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_path", help="input image (JPG/PNG/etc. -- any format Pillow can open)")
    ap.add_argument("out_path", help="output density text file")
    ap.add_argument("--N", type=int, default=64, help="grid divisions in x (default 64)")
    ap.add_argument("--P", type=int, default=None, help="grid divisions in y (default: same as N)")
    ap.add_argument("--invert", action="store_true",
                     help="use (255 - grayscale) as density instead of raw grayscale -- "
                          "for photos where the subject is darker than the background")
    args = ap.parse_args()

    N = args.N
    P = args.P if args.P is not None else N
    if N < 1 or P < 1:
        print("error: N and P must both be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        grid = image_to_grid(args.image_path, N, P, args.invert)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    save_density_text(args.out_path, N, P, grid)
    mass = sum(sum(row) for row in grid) * (1.0 / N) * (1.0 / P)
    print(f"Wrote '{args.out_path}' ({N+1}x{P+1} grid, raw mass ~= {mass:.6f}, renormalized to 1 on load)")
    if mass < 1e-6:
        print("warning: raw mass is ~0 -- the image may be blank, or --invert may need to be "
              "toggled (currently " + ("on" if args.invert else "off") + ")", file=sys.stderr)


if __name__ == "__main__":
    main()
