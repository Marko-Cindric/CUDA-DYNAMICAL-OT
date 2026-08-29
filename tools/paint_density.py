#!/usr/bin/env python3
"""Interactive canvas for hand-drawing a density and saving it in the text
format tools/export_density.cu's --f0/--f1 flags (and its load_density_text
loader) already accept, so a custom initial/final density no longer has to
be authored by hand or scripted externally.

File format written (and, via --load, read): a text file containing the
header 'N P' followed by (N+1)*(P+1) whitespace-separated doubles, row-major
i*(P+1)+j -- the exact format load_density_text (tools/export_density.cu)
parses. Values are written un-normalized; export_density's loader already
Riemann-sum-renormalizes to unit mass on load, so this tool doesn't need to.

Usage:
    python tools/paint_density.py --N 32 --out f0.txt
    python tools/paint_density.py --N 64 --P 48 --out f1.txt --cell-size 12
    python tools/paint_density.py --load f0.txt --out f0.txt   # keep editing

Controls:
    Left click/drag   paint (soft additive brush)
    Right click/drag   erase (same brush, subtractive, clamped at 0)
    [ / ]              shrink / grow brush radius
    - / =              weaken / strengthen brush
    c                  clear the canvas
    Ctrl+Z             undo the last stroke (one level)
    s / Ctrl+S         save
"""

import argparse
import math
import sys
import tkinter as tk


def hot_color(t):
    """Maps t in [0,1] to a hex color along a black -> red -> yellow -> white
    ramp (the standard 'hot' colormap)."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    if t < 1.0 / 3.0:
        r, g, b = t * 3.0, 0.0, 0.0
    elif t < 2.0 / 3.0:
        r, g, b = 1.0, (t - 1.0 / 3.0) * 3.0, 0.0
    else:
        r, g, b = 1.0, 1.0, (t - 2.0 / 3.0) * 3.0
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def load_density_file(path):
    """Parses the same 'N P' + (N+1)*(P+1)-values text format
    load_density_text (tools/export_density.cu) reads. Returns (N, P, grid)
    with grid[i][j] indexed the same way."""
    with open(path) as f:
        tokens = f.read().split()
    if len(tokens) < 2:
        raise ValueError(f"'{path}': expected an 'N P' header")
    N, P = int(tokens[0]), int(tokens[1])
    if N < 1 or P < 1:
        raise ValueError(f"'{path}': N and P must both be >= 1, got N={N} P={P}")
    expected = (N + 1) * (P + 1)
    vals = tokens[2:2 + expected]
    if len(vals) != expected:
        raise ValueError(f"'{path}': expected {expected} values after the header, found {len(vals)}")
    grid = [[0.0] * (P + 1) for _ in range(N + 1)]
    idx = 0
    for i in range(N + 1):
        for j in range(P + 1):
            grid[i][j] = float(vals[idx])
            idx += 1
    return N, P, grid


class PaintApp:
    def __init__(self, root, N, P, cell_size, out_path, grid=None):
        self.root = root
        self.N, self.P = N, P
        self.cell_size = cell_size
        self.out_path = out_path
        self.grid = grid if grid is not None else [[0.0] * (P + 1) for _ in range(N + 1)]

        self.radius = max(1, N // 20)
        self.strength = 1.0
        self.display_max = max(1e-9, max((max(row) for row in self.grid), default=0.0))
        self.undo_snapshot = None

        root.title(f"paint_density -- {N}x{P} grid" + (f" -- {out_path}" if out_path else ""))

        total_w, total_h = (N + 1) * cell_size, (P + 1) * cell_size
        view_w, view_h = min(800, total_w), min(800, total_h)
        self.canvas = tk.Canvas(root, width=view_w, height=view_h, bg="black",
                                 scrollregion=(0, 0, total_w, total_h))
        vbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        hbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # One rectangle per grid cell, created once; painting only ever
        # recolors existing items (itemconfig), never recreates them.
        self.rect_ids = [[None] * (P + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            x0, x1 = i * cell_size, (i + 1) * cell_size
            for j in range(P + 1):
                y0, y1 = (P - j) * cell_size, (P - j + 1) * cell_size
                color = hot_color(self.grid[i][j] / self.display_max)
                self.rect_ids[i][j] = self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew")
        btns = tk.Frame(root)
        btns.grid(row=3, column=0, columnspan=2, sticky="ew")
        tk.Button(btns, text="Clear (c)", command=self.clear).pack(side=tk.LEFT)
        tk.Button(btns, text="Undo (Ctrl+Z)", command=self.undo).pack(side=tk.LEFT)
        tk.Button(btns, text="Save (s)", command=self.save).pack(side=tk.LEFT)
        self._update_status()

        self.canvas.bind("<ButtonPress-1>", lambda e: self._on_press(e, +1))
        self.canvas.bind("<B1-Motion>", lambda e: self._on_drag(e, +1))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_press(e, -1))
        self.canvas.bind("<B3-Motion>", lambda e: self._on_drag(e, -1))
        root.bind("<Key-bracketright>", lambda e: self._adjust_radius(+1))
        root.bind("<Key-bracketleft>", lambda e: self._adjust_radius(-1))
        root.bind("<Key-equal>", lambda e: self._adjust_strength(1.25))
        root.bind("<Key-minus>", lambda e: self._adjust_strength(1 / 1.25))
        root.bind("c", lambda e: self.clear())
        root.bind("s", lambda e: self.save())
        root.bind("<Control-s>", lambda e: self.save())
        root.bind("<Control-z>", lambda e: self.undo())

    # -- coordinate/painting helpers ------------------------------------

    def _cell_at(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        i = int(cx // self.cell_size)
        j = self.P - int(cy // self.cell_size)
        return i, j

    def _on_press(self, event, sign):
        self.undo_snapshot = [row[:] for row in self.grid]
        i, j = self._cell_at(event)
        self.paint_stroke(i, j, sign)

    def _on_drag(self, event, sign):
        i, j = self._cell_at(event)
        self.paint_stroke(i, j, sign)

    def paint_stroke(self, ci, cj, sign):
        radius, strength = self.radius, self.strength
        r_int = max(1, math.ceil(radius))
        touched, rescale_needed = [], False
        for di in range(-r_int, r_int + 1):
            for dj in range(-r_int, r_int + 1):
                i, j = ci + di, cj + dj
                if not (0 <= i <= self.N and 0 <= j <= self.P):
                    continue
                dist = math.hypot(di, dj)
                if dist > radius:
                    continue
                weight = 1.0 - dist / radius if radius > 0 else 1.0
                old = self.grid[i][j]
                new = max(0.0, old + sign * strength * weight)
                if new == old:
                    continue
                self.grid[i][j] = new
                touched.append((i, j))
                if new > self.display_max:
                    self.display_max = new
                    rescale_needed = True

        if rescale_needed:
            self.recolor_all()
        else:
            for i, j in touched:
                self._recolor_cell(i, j)

    def _recolor_cell(self, i, j):
        self.canvas.itemconfig(self.rect_ids[i][j], fill=hot_color(self.grid[i][j] / self.display_max))

    def recolor_all(self):
        for i in range(self.N + 1):
            for j in range(self.P + 1):
                self._recolor_cell(i, j)

    # -- actions ----------------------------------------------------------

    def _adjust_radius(self, direction):
        self.radius = max(1, self.radius + direction)
        self._update_status()

    def _adjust_strength(self, factor):
        self.strength = max(0.01, self.strength * factor)
        self._update_status()

    def _update_status(self):
        self.status.set(f"brush radius={self.radius}  strength={self.strength:.2f}   "
                         f"[ / ] radius, - / = strength, c clear, Ctrl+Z undo, s save")

    def clear(self):
        self.undo_snapshot = [row[:] for row in self.grid]
        self.grid = [[0.0] * (self.P + 1) for _ in range(self.N + 1)]
        self.display_max = 1e-9
        self.recolor_all()

    def undo(self):
        if self.undo_snapshot is None:
            return
        self.grid, self.undo_snapshot = self.undo_snapshot, None
        self.display_max = max(1e-9, max((max(row) for row in self.grid), default=0.0))
        self.recolor_all()

    def save(self):
        path = self.out_path
        if not path:
            try:
                path = input("Output path to save density to: ").strip()
            except EOFError:
                path = ""
            if not path:
                print("Save cancelled (no path given).")
                return
            self.out_path = path
        with open(path, "w") as f:
            f.write(f"{self.N} {self.P}\n")
            for i in range(self.N + 1):
                f.write(" ".join(f"{v:.10g}" for v in self.grid[i]) + "\n")
        mass = sum(sum(row) for row in self.grid) * (1.0 / self.N) * (1.0 / self.P)
        msg = f"Saved to '{path}' (raw mass ~= {mass:.6f}, renormalized to 1 on load)"
        print(msg)
        self.status.set(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=64, help="grid divisions in x (default 64); ignored if --load is given")
    ap.add_argument("--P", type=int, default=None, help="grid divisions in y (default: same as N); ignored if --load is given")
    ap.add_argument("--out", default=None, help="output path to save to (prompted for on save if omitted)")
    ap.add_argument("--cell-size", type=int, default=None, help="pixels per grid cell (default: auto-scaled)")
    ap.add_argument("--load", default=None, help="pre-fill the canvas from an existing density file")
    args = ap.parse_args()

    if args.load:
        try:
            N, P, grid = load_density_file(args.load)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded '{args.load}' ({N}x{P} grid); --N/--P ignored")
    else:
        N = args.N
        P = args.P if args.P is not None else N
        if N < 1 or P < 1:
            print("error: N and P must both be >= 1", file=sys.stderr)
            sys.exit(1)
        grid = None

    cell_size = args.cell_size if args.cell_size else max(4, min(20, 700 // (N + 1)))

    root = tk.Tk()
    PaintApp(root, N, P, cell_size, args.out, grid=grid)
    root.mainloop()


if __name__ == "__main__":
    main()
