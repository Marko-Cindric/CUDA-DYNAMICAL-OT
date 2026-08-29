# Dynamic Optimal Transport (Benamou–Brenier) — CUDA Solver

A CUDA C++ implementation of the dynamic optimal
transport problem: computes the quadratic Wasserstein distance `W2(f0,f1)`
between two 2D probability densities on `[0,1]^2` via the Benamou–Brenier
formulation, solved by asymmetric Douglas–Rachford (A-DR) proximal splitting
on a centered/staggered space-time grid.

See paper.pdf (writen in Croatian) for the mathematical description of the problem 
and the derivation of the Proximal operators.

Rad je prijavlje na natječaj za Rektorovu nagradu za akademsku godinu 2025./2026. Svi primjeri iz rada dobiveni su korištenjem ovog koda.

## Requirements

**To build and run the solver:**
- CMake >= 3.24 (developed/tested with 3.31.5)
- CUDA Toolkit — developed/tested with **12.6.77** (provides `nvcc`, `cudart`, and `cuFFT`; cuFFT ships with the toolkit, no separate install needed). Get it from [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).
- A C++17 host compiler:
  - **Windows** (the only platform this has been built/tested on): Visual Studio 2022 with the **"Desktop development with C++"** workload (provides both the MSVC compiler and the Windows SDK). Tested with Visual Studio Community 2022 17.11.4, MSVC 19.41.34120 (toolset 14.41.34120).
  - Linux/macOS: should work with a C++17-capable gcc/clang and a matching CUDA Toolkit install, since nothing in the CMake setup is Windows-specific, but this hasn't been tried — treat it as untested, not unsupported.
- An NVIDIA GPU, with its CUDA compute capability passed as `CMAKE_CUDA_ARCHITECTURES` (see Build below). Developed/tested on an RTX 4080 (Ada Lovelace, compute capability `89`, the CMake default).

**Optional, only for the Python visualization scripts in `tools/`** (not needed to build or run the solver itself — see Visualization below):
- Python 3 (tested with 3.12)
- `pip install numpy matplotlib pillow` (tested versions: numpy 2.1.3, matplotlib 3.9.2, Pillow 11.0.0)
- `tkinter` — used only by `paint_density.py`. Bundled with a standard CPython install on Windows/macOS; on Linux it's sometimes a separate system package (e.g. `sudo apt install python3-tk` on Debian/Ubuntu).
- `ffmpeg` on `PATH` — only for `make_animation.py`'s `.mp4` output; `.gif` output doesn't need it.

## Build

Requires CUDA Toolkit (tested with 12.6) and CMake >= 3.24.

```
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = RTX 40-series (Ada); adjust for your GPU
cmake --build build --config Release
```

This produces several executables under `build/`:

- `validation_gaussians` — a sanity check: two Gaussians with
  equal covariance but different means, whose `W2` distance has the closed
  form `|mean1 - mean0|`. Reports the solver's estimate vs. the closed form
  across grid resolutions and iteration counts.
- `export_density` — optional visualization helper; see Visualization below.


```
./build/adjoint_tests
./build/validation_gaussians
```

## What's implemented

- **Discretization**: centered grid `G_c` ((N+1)x(P+1)x(Q+1) points) for the
  density/momentum field `V=(m,f)` used to evaluate the objective `J`, and
  three staggered grids (`m^x` staggered in x only, `m^y` in y only, `f`
  staggered in t only) for the flux field `U` used to enforce the continuity
  equation. See `include/ot/grid.cuh` for the exact array-storage convention.
- **Three proximal operators**, matching the problem's `J`, `iota_C`
  (continuity + boundary), `iota_{C_c,s}` (staggered/centered consensus):
  - `Prox_{gamma J}`: exact per-point cubic solve (Cardano's formula +
    Newton polish), fully parallel over grid points.
  - `Prox_{gamma iota_C}`: a **direct cuFFT DCT-II/DCT-III spectral
    solve**. See paper.pdf for the derivation.
  - `Prox_{gamma iota_{C_c,s}}`: exact tridiagonal (Thomas algorithm) solve
    per dimension-line.
- **A-DR iteration** combining the three proxes with weighted averaging and
  relaxation.
  
## Parameters (`ot::OTParams`)

| Field | Meaning |
|---|---|
| `N, P, Q` | Grid divisions in x, y, t (centered grid has `N+1, P+1, Q+1` points per axis) |
| `gamma` | A-DR proximal step size for `Prox_{gamma J}` (default `1.0`) |
| `alpha` | A-DR relaxation parameter, `(0,2)` (default `1.9`) |
| `omega1, omega2, omega3` | Weights on `Prox_{gamma J}`, `Prox_{gamma iota_C}`, `Prox_{gamma iota_{C_c,s}}` respectively; must sum to 1 (default equal `1/3` each) |
| `max_iter`, `tol` | Outer A-DR loop stops when `\|x_{n+1}-x_n\|_2 < tol` or `max_iter` is reached. This class of splitting method converges slowly — expect several thousand iterations for tight residuals on grids beyond `N~16`; see `validation_gaussians`'s own per-resolution budgets |
| `use_cuda_graph` | Capture one outer A-DR iteration into a CUDA Graph and replay it (default `false`; falls back to per-iteration launches automatically if capture is unsupported or fails — check `OTResult::used_cuda_graph`). 1.1x-2.5x faster, more at smaller grids |
| `cuda_graph_batch_size` | Number of graph replays between host-side residual checks when `use_cuda_graph` is active (default `100`). `iterations_run` can overshoot the true convergence point by up to this many extra (harmless) iterations before it's observed |
| `verbose_every` | If > 0, print the outer-loop residual every K iterations (batch-granularity, not exact, under `use_cuda_graph`) |

`OptimalTransportSolver::solve(f0, f1)` takes `f0,f1` as flat
`(N+1)*(P+1)`-length vectors (row-major `i*(P+1)+j`) and returns an
`OTResult` with the centered-grid `f, mx, my` trajectories (each
`(N+1)*(P+1)*(Q+1)` long, row-major `i*(P+1)*(Q+1)+j*(Q+1)+k`), the
estimated `W2_squared`, iteration count, final residual, and
`used_cuda_graph` (whether `use_cuda_graph` was requested and actually
engaged, vs. falling back).

## Notes on validation

`f0`/`f1` are normalized to unit mass under simple Riemann-sum quadrature
(`sum(f) * (1/N) * (1/P) == 1`) before being handed to the solver — the
solver itself does not renormalize its inputs. Grid truncation of the
Gaussians' tails near the domain boundary introduces some approximation
error relative to the true (untruncated) closed-form `W2`; this is expected
to shrink as resolution increases, which is exactly what
`validation_gaussians` checks for.

## Visualization (optional)

`OTResult::f` already contains the full density trajectory `f(x,y,t)`, so
watching the transport instead of only measuring it just needs the right
slices pulled out and rendered. This is entirely optional tooling, kept
deliberately separate from both the CUDA build's core deliverables and each
other:

1. **`export_density`** (built by CMake alongside the other executables,
   `tools/export_density.cu`) runs the solver — by default on the same
   two-Gaussians setup as `validation_gaussians`, or on your own `f0`/`f1`
   loaded from files via `--f0`/`--f1` — then writes a handful of `f(:,:,t)`
   time-slices to a flat binary file, not every slice, just enough to see the
   motion (see its `--help` for every option, including `--stride`, which
   controls how many slices are kept).
2. **`tools/make_animation.py`** (a separate, standalone Python script — not
   part of the CMake build or the C++/CUDA toolchain at all) reads that file
   and renders it as a GIF or MP4 with matplotlib.

```
./build/Release/export_density --N 64 --max-iter 20000 --out density_frames.bin --stride 2
python tools/make_animation.py density_frames.bin transport.gif
```

To transport between your own densities instead of the built-in Gaussians,
pass `--f0`/`--f1` pointing at text files each containing the header `N P`
followed by `(N+1)*(P+1)` whitespace-separated doubles, row-major
`i*(P+1)+j` (see `export_density --help`); both files must agree on `N` and
`P`, and are Riemann-sum-renormalized to unit mass after loading, same as
the built-in Gaussians:

```
./build/Release/export_density --f0 my_f0.txt --f1 my_f1.txt --max-iter 20000 --out density_frames.bin
```

**`tools/paint_density.py`** — a third, equally optional piece — hand-draws
a density in exactly that file format instead of requiring one to be
authored by hand or scripted externally. It's a standalone script using
only `tkinter` (bundled with a standard CPython install, so no new
dependency at all): give it a grid size, get a small paint canvas, click
and drag to add density with a soft brush (right-click to erase), then save.

```
python tools/paint_density.py --N 32 --out my_f0.txt
python tools/paint_density.py --N 32 --out my_f1.txt
./build/Release/export_density --f0 my_f0.txt --f1 my_f1.txt --max-iter 20000 --out density_frames.bin
```

Run `python tools/paint_density.py --help` for the full control list
(brush radius/strength, undo, clear) and for `--load`, which reopens an
existing density file for further editing.

**`tools/image_to_density.py`** — a fourth optional piece, for transporting
one actual photo onto another instead of a synthetic or hand-drawn density.
Converts an image (grayscale or color, auto-desaturated) into the same text
format via Pillow: desaturate, stretch-resize to the target grid (ignoring
the original aspect ratio, same as everywhere else in this project treats
the domain as a plain grid), write it out. `--invert` flips which end of
the brightness range counts as mass, for photos where the subject is darker
than the background.

```
python tools/image_to_density.py photo0.jpg f0.txt --N 64
python tools/image_to_density.py photo1.jpg f1.txt --N 64
./build/Release/export_density --f0 f0.txt --f1 f1.txt --max-iter 20000 --out density_frames.bin
python tools/make_animation.py density_frames.bin transport.gif
```

**`tools/density_grid.py`** — a fifth optional piece, for a static look at a
transport instead of an animation: lays out the initial density, `n` evenly
time-spaced in-between frames, and the final density side by side in one
labeled image (each panel captioned with its `t` value), reading the same
binary frames file `make_animation.py` animates.

```
./build/Release/export_density --N 64 --max-iter 20000 --out density_frames.bin --stride 16
python tools/density_grid.py density_frames.bin --n 3 --out grid.png
```

`--n 3` shows 5 panels (`t=0, 0.25, 0.5, 0.75, 1`); the tool picks, from
whichever frames are actually in the file, the ones closest to those evenly
spaced times, and warns if the closest available frame isn't an exact
match — export with `--stride = Q/(n+1)` (must divide evenly) to guarantee
exact matches instead.

By default each panel is colored with `--cmap`'s heatmap (any matplotlib
colormap name, default `magma`). Pass `--style grayscale` for plain
low-density-white/high-density-black shading instead (matplotlib's `Greys`
colormap) — overrides `--cmap` when given.

By default all panels also share one color scale (one colorbar, `vmax` =
the largest value across every shown panel), so density is physically
comparable panel to panel — but if the two endpoint densities have very
different peak concentrations (e.g. two photos converted with
`image_to_density.py`, one with sparser linework than the other), the
lower-peak one can never reach full color under a shared scale and looks
washed out. Pass `--normalize per-frame` to instead scale each panel 0 to
its *own* peak (each panel's peak value printed in its label, no shared
colorbar) — every panel looks as crisp as its own content allows, at the
cost of no longer being comparable to the others in absolute terms.

Nothing else in the project reads `export_density`'s output or depends on
`make_animation.py` running — both steps are purely for visual inspection,
and the Python script's dependencies (`numpy`, `matplotlib`; `Pillow` for GIF
output or `ffmpeg` on `PATH` for MP4) are not required to build or run the
solver itself.
