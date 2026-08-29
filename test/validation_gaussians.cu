// Validation driver: transport between two Gaussians with the same
// covariance but different means has closed-form W2 equal to the
// Euclidean distance between the means. Builds f0,f1 on the grid, runs the
// solver, and reports the estimated W2 vs. the closed-form value across
// resolutions and iteration counts, plus a mass-center-over-time check that
// the interpolation is a near-pure translation.

#include "ot/solver.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>

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

void print_mass_center_trajectory(const OTResult& res, int N, int P, int Q) {
    std::printf("  mass-center trajectory (should move ~linearly, mass should stay ~1):\n");
    double dx = 1.0 / N, dy = 1.0 / P;
    int stride = std::max(1, Q / 8);
    for (int k = 0; k <= Q; k += stride) {
        double sumf = 0, sumx = 0, sumy = 0;
        for (int i = 0; i <= N; ++i) {
            double x = (double)i / N;
            for (int j = 0; j <= P; ++j) {
                double y = (double)j / P;
                double val = res.f[(size_t)(((long long)i * (P + 1) + j) * (Q + 1) + k)];
                sumf += val; sumx += val * x; sumy += val * y;
            }
        }
        sumf *= dx * dy; sumx *= dx * dy; sumy *= dx * dy;
        std::printf("    t=%.3f  mass=%.4f  center=(%.4f, %.4f)\n", (double)k / Q, sumf,
                    sumf > 1e-12 ? sumx / sumf : 0.0, sumf > 1e-12 ? sumy / sumf : 0.0);
    }
}

} // namespace

int main() {
    const double mux0 = 0.3, muy0 = 0.5, mux1 = 0.7, muy1 = 0.5, sigma = 0.05;
    const double closed_form_W2 = std::sqrt((mux1 - mux0) * (mux1 - mux0) + (muy1 - muy0) * (muy1 - muy0));
    std::printf("Closed-form W2 (equal-covariance Gaussians, means (%.2f,%.2f) -> (%.2f,%.2f)) = %.6f\n\n",
                mux0, muy0, mux1, muy1, closed_form_W2);

    std::printf("=== Error vs. resolution (N=P=Q) ===\n");
    std::printf("(A-DR/PPXA-style splitting converges slowly -- larger grids genuinely need\n");
    std::printf(" more outer iterations, not just more wall-clock per iteration; budgets below\n");
    std::printf(" are scaled accordingly. Pushed further than before thanks to the persistent-\n");
    std::printf(" cooperative-kernel CG solve in IMPLEMENTATION.md section 15.)\n");

    struct Cfg { int res; int max_iter; double tol; };
    std::vector<Cfg> cfgs = {{16, 10000, 1e-7}, {32, 16000, 1e-6}, {64, 40000, 1e-6},
                              {96, 40000, 1e-6}, {128, 30000, 1e-6}};
    for (auto& cfg : cfgs) {
        OTParams p;
        p.N = p.P = p.Q = cfg.res;
        p.max_iter = cfg.max_iter;
        p.tol = cfg.tol;
        OptimalTransportSolver solver(p);
        auto f0 = make_gaussian(p.N, p.P, mux0, muy0, sigma);
        auto f1 = make_gaussian(p.N, p.P, mux1, muy1, sigma);
        auto t0 = std::chrono::steady_clock::now();
        auto res = solver.solve(f0, f1);
        auto t1 = std::chrono::steady_clock::now();
        double w2 = std::sqrt(std::max(0.0, res.W2_squared));
        double relerr = std::fabs(w2 - closed_form_W2) / closed_form_W2;
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("N=P=Q=%3d  iters=%4d  residual=%.3e  W2_est=%.5f  relerr=%.4f  time=%.2fs\n",
                    cfg.res, res.iterations_run, res.final_residual, w2, relerr, secs);
        if (cfg.res == 32 || cfg.res == 128) print_mass_center_trajectory(res, p.N, p.P, p.Q);
    }

    std::printf("\n=== Error vs. iteration count (N=P=Q=24, forced to run exactly max_iter) ===\n");
    std::vector<int> iters_list = {50, 100, 200, 400, 800, 1600, 3200, 6400, 9600};
    for (int mi : iters_list) {
        OTParams p;
        p.N = p.P = p.Q = 24;
        p.max_iter = mi;
        p.tol = 0.0;
        OptimalTransportSolver solver(p);
        auto f0 = make_gaussian(p.N, p.P, mux0, muy0, sigma);
        auto f1 = make_gaussian(p.N, p.P, mux1, muy1, sigma);
        auto res = solver.solve(f0, f1);
        double w2 = std::sqrt(std::max(0.0, res.W2_squared));
        double relerr = std::fabs(w2 - closed_form_W2) / closed_form_W2;
        std::printf("max_iter=%4d  residual=%.3e  W2_est=%.5f  relerr=%.4f\n", mi, res.final_residual, w2, relerr);
    }

    return 0;
}
