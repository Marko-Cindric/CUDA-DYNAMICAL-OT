#!/usr/bin/env python3
"""Lays out evenly time-spaced density snapshots (initial, n in-between, and
final) side by side in a single labeled image, so a transport can be
inspected as a static contact sheet.

Reads the same binary frames file tools/export_density.cu writes and
tools/make_animation.py already animates -- see either file's header
comment for the exact format. This tool doesn't require export_density to
have exported exactly the requested times: it picks, from whatever frames
are actually present, the one closest in time to each of the n+2 evenly
spaced targets t = 0, 1/(n+1), 2/(n+1), ..., 1, and warns if the closest
match isn't a good one. For an exact match at every target time, export
with --stride = Q/(n+1) (Q must be evenly divisible by n+1).

Usage:
    python tools/density_grid.py density_frames.bin --n 3 --out grid.png
    python tools/density_grid.py density_frames.bin --n 5 --out grid.png --cmap inferno --labels below
    python tools/density_grid.py density_frames.bin --n 3 --out grid.png --style grayscale
"""

import argparse
import struct
import sys

import numpy as np


def read_frames(path):
    """Parses export_density's binary format: int32 N, int32 P, int32
    num_frames, then num_frames doubles of physical time t, then num_frames
    frames of (N+1)x(P+1) doubles each, row-major i*(P+1)+j."""
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"'{path}' is too short to be a valid density_frames file")
        N, P, num_frames = struct.unpack("<iii", header)
        times = np.frombuffer(f.read(8 * num_frames), dtype="<f8", count=num_frames)
        frame_size = (N + 1) * (P + 1)
        data = np.frombuffer(f.read(8 * frame_size * num_frames), dtype="<f8", count=frame_size * num_frames)
        if data.size != frame_size * num_frames:
            raise ValueError(f"'{path}' is truncated: expected {frame_size * num_frames} values, got {data.size}")
        frames = data.reshape(num_frames, N + 1, P + 1)
    return N, P, times, frames


def pick_evenly_spaced(times, frames, n):
    """Returns (picked_times, picked_frames, actual_times) for the n+2
    targets t=0,1/(n+1),...,1 -- picked_times are the *targets*,
    actual_times are the closest available frame's real time (which may
    differ slightly if the export's stride didn't land exactly on them)."""
    targets = [k / (n + 1) for k in range(n + 2)]
    idxs = [int(np.argmin(np.abs(times - t))) for t in targets]
    actual = times[idxs]
    for target, got in zip(targets, actual):
        if abs(target - got) > 1e-9:
            print(f"warning: no frame at exactly t={target:.4f}; using closest available, t={got:.4f}",
                  file=sys.stderr)
    return targets, frames[idxs], actual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="binary frames file produced by tools/export_density")
    ap.add_argument("--n", type=int, default=3, help="number of in-between frames (default 3); "
                                                       "total panels shown = n+2 (initial + n + final)")
    ap.add_argument("--out", default="density_grid.png", help="output image path (default density_grid.png)")
    ap.add_argument("--cmap", default="magma", help="matplotlib colormap (default magma); ignored if --style grayscale is given")
    ap.add_argument("--style", choices=["color", "grayscale"], default="color",
                     help="'color' (default) uses --cmap as a heatmap. 'grayscale' renders low density as "
                          "white and high density as black (matplotlib's 'Greys' colormap), overriding --cmap.")
    ap.add_argument("--dpi", type=int, default=150, help="output resolution (default 150)")
    ap.add_argument("--labels", choices=["above", "below"], default="above",
                     help="place each panel's t= label above or below it (default above)")
    ap.add_argument("--normalize", choices=["shared", "per-frame"], default="shared",
                     help="'shared' (default) uses one vmax (the max over all shown panels) for every "
                          "panel, with one shared colorbar -- panels stay physically comparable to each "
                          "other, but a panel whose own peak is well below the shared max never reaches "
                          "full color. 'per-frame' instead scales each panel 0..its own max independently "
                          "(no shared colorbar, since the scale differs per panel) -- useful when the "
                          "two endpoint densities have very different peak concentrations (e.g. two "
                          "photos/drawings with different ink density) and you want every panel to look "
                          "as crisp as its own content allows, at the cost of no longer being able to "
                          "visually compare absolute density between panels.")
    args = ap.parse_args()

    if args.n < 0:
        print("error: --n must be >= 0", file=sys.stderr)
        sys.exit(1)

    N, P, times, frames = read_frames(args.input)
    _, picked, actual_times = pick_evenly_spaced(times, frames, args.n)
    print(f"Loaded {len(times)} frames on a {N+1}x{P+1} grid; showing {len(picked)} panels "
          f"at t = {', '.join(f'{t:.3f}' for t in actual_times)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 'Greys' is matplotlib's sequential white(low)->black(high) grayscale ramp
    cmap = "Greys" if args.style == "grayscale" else args.cmap

    shared_vmax = picked.max()
    n_panels = len(picked)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.0), squeeze=False)
    axes = axes[0]

    im = None
    for ax, frame, t in zip(axes, picked, actual_times):
        panel_vmax = frame.max() if args.normalize == "per-frame" else shared_vmax
        im = ax.imshow(frame.T, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=panel_vmax, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        label = f"t = {t:.2f}"
        if args.normalize == "per-frame":
            label += f"\n(peak={panel_vmax:.3g})"
        if args.labels == "above":
            ax.set_title(label, fontsize=11)
        else:
            ax.set_xlabel(label, fontsize=11)

    if args.normalize == "shared":
        fig.colorbar(im, ax=axes.tolist(), label="density", fraction=0.025, pad=0.02)
        subtitle = "shared color scale"
    else:
        # Each panel has its own vmax, so one colorbar would misrepresent
        # every panel but the last -- the per-panel peak printed in each
        # panel's own label is the only honest way to convey scale here.
        subtitle = "each panel independently normalized to its own peak -- not directly comparable across panels"
    fig.suptitle(f"Density transport, t=0 → t=1 ({n_panels} frames, {subtitle})")
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote '{args.out}'")


if __name__ == "__main__":
    main()
