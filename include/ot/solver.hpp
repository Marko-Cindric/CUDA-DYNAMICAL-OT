#pragma once
// Public API for the classical dynamic optimal transport
// solver: quadratic Wasserstein distance between two 2D densities on
// [0,1]^2 via Benamou-Brenier + asymmetric Douglas-Rachford proximal
// splitting on a centered/staggered space-time grid.

#include <vector>

namespace ot {

struct OTParams {
  int N = 0, P = 0, Q = 0; // grid divisions in x, y, t (centered grid has N+1,P+1,Q+1 points)

  // A-DR proximal step size for Prox_{gamma J} (the other two proxes are
  // projections, scale-invariant in gamma). Tune per problem if needed.
  double gamma = 1.0;

  // A-DR relaxation, in (0,2).
  double alpha = 1.9;

  // Weights on the three simple functionals (J, iota_C, iota_{C_c,s}); must sum to 1.
  double omega1 = 1.0 / 3.0, omega2 = 1.0 / 3.0, omega3 = 1.0 / 3.0;

  int max_iter = 20000; // outer A-DR iteration cap
  double tol = 1e-6;   // stop when ||x_{n+1}-x_n||_2 < tol

  // Capture one outer A-DR iteration into a CUDA Graph (on first use) and
  // replay it cuda_graph_batch_size times between host-side residual
  // checks.
  bool use_cuda_graph = false;
  // Number of graph replays (K) between host-side residual reads when
  // use_cuda_graph is active.
  int cuda_graph_batch_size = 100;

  // 0 = silent; else print residual roughly every K outer iterations.
  int verbose_every = 0;
};

struct OTResult {
  // Centered-grid trajectories, size (N+1)*(P+1)*(Q+1) each, row-major i*(P+1)*(Q+1)+j*(Q+1)+k.
  std::vector<double> f, mx, my;
  // Estimated squared W2 distance = 2 * J(V*) at convergence
  double W2_squared = 0.0;
  int iterations_run = 0;
  double final_residual = 0.0;
  // True iff use_cuda_graph was requested AND graph capture actually
  // succeeded and was used for this solve.
  bool used_cuda_graph = false;
};

class OptimalTransportSolver {
public:
  explicit OptimalTransportSolver(const OTParams& params);
  ~OptimalTransportSolver();
  OptimalTransportSolver(const OptimalTransportSolver&) = delete;
  OptimalTransportSolver& operator=(const OptimalTransportSolver&) = delete;

  // f0, f1: size (N+1)*(P+1), row-major i*(P+1)+j -- the two densities on
  // the centered (x,y) grid. Not renormalized internally; the caller should
  // ensure they represent consistent probability mass under whatever grid
  // quadrature convention is intended (see README).
  OTResult solve(const std::vector<double>& f0, const std::vector<double>& f1);

private:
  struct Impl;
  Impl* impl_;
};

} // namespace ot
