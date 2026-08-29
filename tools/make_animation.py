#!/usr/bin/env python3
"""Compile density frames exported by tools/export_density.cu into an
animation (GIF or MP4) so the mass transport can be watched over time.

Requires: numpy, matplotlib (with Pillow for GIF output; ffmpeg on PATH for
MP4 output).

Usage:
    python tools/make_animation.py density_frames.bin transport.gif
    python tools/make_animation.py density_frames.bin transport.mp4 --fps 20 --cmap inferno
"""

import argparse
import struct
import sys

import numpy as np


def read_frames(path):
    """Parse export_density's binary format.

    Layout: int32 N, int32 P, int32 num_frames, then num_frames doubles of
    physical time `t`, then num_frames frames of (N+1)x(P+1) doubles each,
    row-major i*(P+1)+j, in increasing time order.
    """
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="binary frames file produced by tools/export_density")
    ap.add_argument("output", help="output animation path (.gif or .mp4)")
    ap.add_argument("--fps", type=int, default=12, help="frames per second (default 12)")
    ap.add_argument("--cmap", default="magma", help="matplotlib colormap (default magma)")
    ap.add_argument("--dpi", type=int, default=100, help="output resolution (default 100)")
    args = ap.parse_args()

    N, P, times, frames = read_frames(args.input)
    print(f"Loaded {len(times)} frames on a {N+1}x{P+1} grid from '{args.input}'")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    vmax = frames.max()
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(frames[0].T, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=vmax, cmap=args.cmap)
    title = ax.set_title(f"t = {times[0]:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="density f")
    fig.tight_layout()

    def update(frame_idx):
        im.set_data(frames[frame_idx].T)
        title.set_text(f"t = {times[frame_idx]:.3f}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=1000 / args.fps, blit=False)

    out = args.output
    if out.lower().endswith(".mp4"):
        anim.save(out, writer=animation.FFMpegWriter(fps=args.fps), dpi=args.dpi)
    else:
        anim.save(out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote animation to '{out}'")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
