// Optional visualization helper: runs the solver -- either on the same two-
// Gaussians setup as test/validation_gaussians.cu, or on user-supplied f0/f1
// loaded from files -- then dumps the resulting density f(x,y,t) as a
// sequence of time-frames to a flat binary file that tools/make_animation.py
// (a separate, optional Python script) turns into a GIF/MP4 of the mass
// transport. See README.md's Visualization section.
//
// Usage: export_density [options]
//   --N <int>         grid divisions in x=y for the built-in synthetic
//                      Gaussians (default 64); ignored if --f0/--f1 are
//                      given, since the grid is then taken from those files
//   --Q <int>         grid divisions in t (default: same as N, or as the
//                      loaded f0/f1's N if --f0/--f1 are given)
//   --max-iter <int>  outer A-DR iteration cap (default 20000)
//   --tol <float>     outer A-DR stopping tolerance (default 1e-6)
//   --out <path>      output binary frames file (default density_frames.bin)
//   --stride <int>    export every stride-th time slice, always including
//                      t=0 and t=1 (default 1, i.e. every slice)
//   --f0 <path>       load the initial density from a text file instead of
//                      the built-in synthetic Gaussian (requires --f1 too)
//   --f1 <path>       load the final density from a text file (requires --f0)
//
// --f0/--f1 file format: a text file containing 'N P' followed by
// (N+1)*(P+1) whitespace-separated doubles, row-major i*(P+1)+j. Both files
// must agree on N and P; the loaded values are Riemann-sum-renormalized to
// unit mass (the solver itself does not renormalize its inputs).
//
// Output format (all little-endian, native double/int32 layout -- this
// machine and the Python reader are assumed to agree, which holds for any
// x86_64 host):
//   int32   N            (grid divisions in x; centered grid has N+1 points)
//   int32   P            (grid divisions in y)
//   int32   num_frames
//   double  times[num_frames]              -- physical t in [0,1] per frame
//   double  frames[num_frames][N+1][P+1]   -- row-major i*(P+1)+j per frame,
//                                              frames stored back-to-back in
//                                              increasing time order

#include "ot/solver.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace ot;

namespace {

std::vector<double> make_gaussian(int N, int P, double mux, double muy, double sigma) {
    std::vector<double> f((size_t)(N + 1) * (P + 1));
    double dx = 1.0 / N, dy = 1.0 / P;
    double sum = 0.0;
    for (int i = 0; i <= N; ++i) {
        double x = (double)i / N;
        for (int j = 0; j <= P; ++j) {
            double y = (double)j / P;
            double ddx = x - mux, ddy = y - muy;
            double val = std::exp(-(ddx * ddx + ddy * ddy) / (2.0 * sigma * sigma));
            f[(size_t)i * (P + 1) + j] = val;
            sum += val;
        }
    }
    double mass = sum * dx * dy; // Riemann-sum normalization to unit probability mass
    for (auto& v : f) v /= mass;
    return f;
}

// Riemann-sum mass normalization.
void normalize_mass(std::vector<double>& f, int N, int P) {
    double dx = 1.0 / N, dy = 1.0 / P;
    double sum = 0.0;
    for (double v : f) sum += v;
    double mass = sum * dx * dy;
    if (mass > 1e-300) {
        for (auto& v : f) v /= mass;
    }
}

// Loads a density from a simple whitespace-separated text file:
//   N P
//   value(i=0,j=0) value(i=0,j=1) ... value(i=0,j=P)
//   value(i=1,j=0) ...
//   ...
std::vector<double> load_density_text(const std::string& path, int& N, int& P) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open '" + path + "'");
    if (!(in >> N >> P) || N < 1 || P < 1) {
        throw std::runtime_error("'" + path + "': expected an 'N P' header with N,P >= 1");
    }
    std::vector<double> f((size_t)(N + 1) * (P + 1));
    for (size_t idx = 0; idx < f.size(); ++idx) {
        if (!(in >> f[idx])) {
            throw std::runtime_error("'" + path + "': expected " + std::to_string(f.size()) +
                                      " values after the header, ran out at index " + std::to_string(idx));
        }
    }
    normalize_mass(f, N, P);
    return f;
}

struct Args {
    int N = 64, Q = -1, max_iter = 20000, stride = 1;
    double tol = 1e-6;
    std::string out_path = "density_frames.bin";
    std::string f0_path, f1_path;
};

void print_usage() {
    std::printf(
        "Usage: export_density [options]\n"
        "  --N <int>         grid divisions in x=y for the built-in synthetic\n"
        "                     Gaussians (default 64); ignored if --f0/--f1 are\n"
        "                     given, since the grid is then taken from those files\n"
        "  --Q <int>         grid divisions in t (default: same as N, or as the\n"
        "                     loaded f0/f1's N if --f0/--f1 are given)\n"
        "  --max-iter <int>  outer A-DR iteration cap (default 20000)\n"
        "  --tol <float>     outer A-DR stopping tolerance (default 1e-6)\n"
        "  --out <path>      output binary frames file (default density_frames.bin)\n"
        "  --stride <int>    export every stride-th time slice, always including\n"
        "                     t=0 and t=1 (default 1, i.e. every slice)\n"
        "  --f0 <path>       load the initial density from a text file instead of\n"
        "                     the built-in synthetic Gaussian (requires --f1 too)\n"
        "  --f1 <path>       load the final density from a text file (requires --f0)\n"
        "\n"
        "--f0/--f1 file format: 'N P' followed by (N+1)*(P+1) whitespace-separated\n"
        "doubles, row-major i*(P+1)+j.\n");
}

// Throws std::runtime_error with a message on any parse problem.
bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " needs a value");
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") { print_usage(); return false; }
        else if (arg == "--N") a.N = std::atoi(next("--N").c_str());
        else if (arg == "--Q") a.Q = std::atoi(next("--Q").c_str());
        else if (arg == "--max-iter") a.max_iter = std::atoi(next("--max-iter").c_str());
        else if (arg == "--tol") a.tol = std::atof(next("--tol").c_str());
        else if (arg == "--out") a.out_path = next("--out");
        else if (arg == "--stride") a.stride = std::atoi(next("--stride").c_str());
        else if (arg == "--f0") a.f0_path = next("--f0");
        else if (arg == "--f1") a.f1_path = next("--f1");
        else throw std::runtime_error("unrecognized argument '" + arg + "' (see --help)");
    }
    if (a.f0_path.empty() != a.f1_path.empty()) {
        throw std::runtime_error("--f0 and --f1 must be given together");
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse_args(argc, argv, args)) return 0; // --help
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage();
        return 1;
    }

    const double mux0 = 0.3, muy0 = 0.5, mux1 = 0.7, muy1 = 0.5, sigma = 0.05;

    std::vector<double> f0, f1;
    int N, P;
    try {
        if (!args.f0_path.empty()) {
            int N0, P0, N1, P1;
            f0 = load_density_text(args.f0_path, N0, P0);
            f1 = load_density_text(args.f1_path, N1, P1);
            if (N0 != N1 || P0 != P1) {
                throw std::runtime_error("--f0 and --f1 grids disagree: " + std::to_string(N0) + "x" +
                                          std::to_string(P0) + " vs " + std::to_string(N1) + "x" +
                                          std::to_string(P1));
            }
            N = N0; P = P0;
            std::printf("Loaded f0 from '%s' and f1 from '%s' (%dx%d grid)\n", args.f0_path.c_str(),
                        args.f1_path.c_str(), N + 1, P + 1);
        } else {
            N = P = args.N;
            f0 = make_gaussian(N, P, mux0, muy0, sigma);
            f1 = make_gaussian(N, P, mux1, muy1, sigma);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const int Q = args.Q > 0 ? args.Q : N;
    const int stride = std::max(1, args.stride);

    OTParams p;
    p.N = N; p.P = P; p.Q = Q;
    p.max_iter = args.max_iter;
    p.tol = args.tol;
    p.verbose_every = std::max(1, args.max_iter / 10);

    std::printf("Solving OT on a %dx%dx%d grid (max_iter=%d, tol=%.1e)...\n", N, P, Q, args.max_iter, args.tol);
    OptimalTransportSolver solver(p);
    auto res = solver.solve(f0, f1);
    std::printf("Done: iters=%d residual=%.3e W2=%.5f\n", res.iterations_run, res.final_residual,
                std::sqrt(std::max(0.0, res.W2_squared)));

    std::vector<int> ks;
    for (int k = 0; k < Q; k += stride) ks.push_back(k);
    if (ks.empty() || ks.back() != Q) ks.push_back(Q); // always include the final slice

    FILE* fp = std::fopen(args.out_path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "Failed to open '%s' for writing\n", args.out_path.c_str());
        return 1;
    }
    int32_t hN = N, hP = P, hFrames = (int32_t)ks.size();
    std::fwrite(&hN, sizeof(hN), 1, fp);
    std::fwrite(&hP, sizeof(hP), 1, fp);
    std::fwrite(&hFrames, sizeof(hFrames), 1, fp);

    std::vector<double> times(ks.size());
    for (size_t f = 0; f < ks.size(); ++f) times[f] = (double)ks[f] / Q;
    std::fwrite(times.data(), sizeof(double), times.size(), fp);

    std::vector<double> frame((size_t)(N + 1) * (P + 1));
    for (int k : ks) {
        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= P; ++j) {
                frame[(size_t)i * (P + 1) + j] = res.f[(size_t)(((long long)i * (P + 1) + j) * (Q + 1) + k)];
            }
        }
        std::fwrite(frame.data(), sizeof(double), frame.size(), fp);
    }
    std::fclose(fp);

    std::printf("Wrote %zu frames to '%s'\n", ks.size(), args.out_path.c_str());
    std::printf("Animate with: python tools/make_animation.py %s <output.gif|output.mp4>\n", args.out_path.c_str());
    return 0;
}
