#include "ot/solver.hpp"
#include "ot/grid.cuh"
#include "ot/joint_field.hpp"
#include "ot/kernels.hpp"
#include "ot/prox_c.hpp"
#include "ot/prox_consensus.hpp"
#include "ot/dr_iteration.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace ot {

struct OptimalTransportSolver::Impl {
  GridDims g;
  JointLayout L;
  OTParams params;
  ProxC proxC;
  ProxConsensus proxCS;

  thrust::device_vector<double> z1, z2, z3, x, p, pJ, pC, pCS;
  thrust::device_vector<double> f0d, f1d;

  // CUDA-Graph capture state for use_cuda_graph. graph_/graph_exec_ are 
  // captured lazily on first use inside one solve() call and reused for 
  // the rest of this Impl's lifetime.
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_ready_ = false;    // capture succeeded, exec instantiated
  bool graph_disabled_ = false; // capture failed or unsupported config -- never retried
  thrust::device_vector<double> residual_sq_dev;

  explicit Impl(const OTParams& prm)
    : g{prm.N, prm.P, prm.Q}, L(JointLayout::make(g)), params(prm),
      proxC(g), proxCS(g),
      z1(L.total, 0.0), z2(L.total, 0.0), z3(L.total, 0.0), x(L.total, 0.0),
      p(L.total, 0.0), pJ(L.total, 0.0), pC(L.total, 0.0), pCS(L.total, 0.0),
      f0d((size_t)g.nx_c() * g.ny_c(), 0.0), f1d((size_t)g.nx_c() * g.ny_c(), 0.0),
      residual_sq_dev(1, 0.0) {}

  ~Impl() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
  }

  // Idempotent: returns graph_ready_ immediately on repeat calls. Attempts
  // capture exactly once per Impl; any failure sets graph_disabled_ and is
  // never retried.
  bool ensure_graph_ready(const DRWeights& w) {
    if (graph_ready_) return true;
    if (graph_disabled_) return false;

    if (!proxC.bind_cuda_graph_stream(cudaStreamPerThread)) {
      std::fprintf(stderr, "[ot] use_cuda_graph requested but ProxC could not bind a "
                           "capturable stream; falling back to per-iteration launches.\n");
      graph_disabled_ = true;
      return false;
    }

    double* z1p = thrust::raw_pointer_cast(z1.data());
    double* z2p = thrust::raw_pointer_cast(z2.data());
    double* z3p = thrust::raw_pointer_cast(z3.data());
    double* xp = thrust::raw_pointer_cast(x.data());
    double* pp = thrust::raw_pointer_cast(p.data());
    double* pJp = thrust::raw_pointer_cast(pJ.data());
    double* pCp = thrust::raw_pointer_cast(pC.data());
    double* pCSp = thrust::raw_pointer_cast(pCS.data());
    const double* f0dp = thrust::raw_pointer_cast(f0d.data());
    const double* f1dp = thrust::raw_pointer_cast(f1d.data());
    double* rsq = thrust::raw_pointer_cast(residual_sq_dev.data());

    cudaError_t begin_err = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    if (begin_err != cudaSuccess) {
      std::fprintf(stderr, "[ot] cudaStreamBeginCapture failed: %s; "
                           "falling back to per-iteration launches.\n",
                   cudaGetErrorString(begin_err));
      graph_disabled_ = true;
      return false;
    }

    bool threw = false;
    try {
      dr_iteration_step_capturable(L, w, proxC, proxCS, z1p, z2p, z3p, xp,
                                    pp, pJp, pCp, pCSp, f0dp, f1dp, rsq);
    } catch (...) {
      // Must still end capture, or cudaStreamPerThread is left stuck
      // in capturing mode permanently
      threw = true;
    }

    cudaGraph_t g_captured = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(cudaStreamPerThread, &g_captured);
    if (threw || end_err != cudaSuccess) {
      if (g_captured) cudaGraphDestroy(g_captured);
      std::fprintf(stderr, "[ot] CUDA graph capture failed (likely a cuFFT/driver error); "
                           "falling back to per-iteration launches.\n");
      graph_disabled_ = true;
      return false;
    }

    cudaError_t inst_err = cudaGraphInstantiate(&graph_exec_, g_captured, 0);
    if (inst_err != cudaSuccess) {
      cudaGraphDestroy(g_captured);
      std::fprintf(stderr, "[ot] CUDA graph instantiate failed: %s; "
                           "falling back to per-iteration launches.\n",
                   cudaGetErrorString(inst_err));
      graph_disabled_ = true;
      return false;
    }
    
    graph_ = g_captured;
    graph_ready_ = true;
    return true;
  }
};

OptimalTransportSolver::OptimalTransportSolver(const OTParams& params) {
  if (params.N < 1 || params.P < 1 || params.Q < 1)
    throw std::invalid_argument("OTParams: N,P,Q must all be >= 1");
  double wsum = params.omega1 + params.omega2 + params.omega3;
  if (std::fabs(wsum - 1.0) > 1e-9)
    throw std::invalid_argument("OTParams: omega1+omega2+omega3 must equal 1");
  if (!(params.alpha > 0.0 && params.alpha < 2.0))
    throw std::invalid_argument("OTParams: alpha must be in (0,2)");
  if (!(params.gamma > 0.0))
    throw std::invalid_argument("OTParams: gamma must be > 0");
  impl_ = new Impl(params);
}

OptimalTransportSolver::~OptimalTransportSolver() { delete impl_; }

OTResult OptimalTransportSolver::solve(const std::vector<double>& f0, const std::vector<double>& f1) {
  Impl& im = *impl_;
  const GridDims& g = im.g;
  const long long npts_xy = (long long)g.nx_c() * g.ny_c();
  if ((long long)f0.size() != npts_xy || (long long)f1.size() != npts_xy)
    throw std::invalid_argument("solve(): f0,f1 must have size (N+1)*(P+1)");

  thrust::copy(f0.begin(), f0.end(), im.f0d.begin());
  thrust::copy(f1.begin(), f1.end(), im.f1d.begin());
  const double* f0d = thrust::raw_pointer_cast(im.f0d.data());
  const double* f1d = thrust::raw_pointer_cast(im.f1d.data());

  // --- warm start: zero momentum everywhere; f linearly interpolated in
  // time on both the centered and staggered grids. ---
  std::vector<double> host_x(im.L.total, 0.0);
  for (int i = 0; i <= g.N; ++i) {
    for (int j = 0; j <= g.P; ++j) {
      long long ij = (long long)i * g.ny_c() + j;
      double v0 = f0[ij], v1 = f1[ij];
      for (int k = 0; k <= g.Q; ++k) {
        double t = (double)k / g.Q;
        host_x[im.L.off_f_c + idx_c(g, i, j, k)] = (1.0 - t) * v0 + t * v1;
      }
      for (int c = 0; c <= g.Q + 1; ++c) {
        double t = (c == 0) ? 0.0 : (c == g.Q + 1 ? 1.0 : ((double)c - 0.5) / g.Q);
        host_x[im.L.off_f_s + idx_f(g, i, j, c)] = (1.0 - t) * v0 + t * v1;
      }
    }
  }
  thrust::copy(host_x.begin(), host_x.end(), im.x.begin());
  im.z1 = im.x;
  im.z2 = im.x;
  im.z3 = im.x;

  DRWeights w{im.params.omega1, im.params.omega2, im.params.omega3, im.params.alpha, im.params.gamma};
  double* z1 = thrust::raw_pointer_cast(im.z1.data());
  double* z2 = thrust::raw_pointer_cast(im.z2.data());
  double* z3 = thrust::raw_pointer_cast(im.z3.data());
  double* xp = thrust::raw_pointer_cast(im.x.data());
  double* pp = thrust::raw_pointer_cast(im.p.data());
  double* pJ = thrust::raw_pointer_cast(im.pJ.data());
  double* pC = thrust::raw_pointer_cast(im.pC.data());
  double* pCS = thrust::raw_pointer_cast(im.pCS.data());

  int iterations_run = 0;
  double residual = 1e300;
  bool used_cuda_graph = false;

  if (im.params.max_iter > 0) {
    // Iteration 0 always runs through the ordinary, non-graph path --
    residual = dr_iteration_step(im.L, w, im.proxC, im.proxCS, z1, z2, z3, xp,
                                  pp, pJ, pC, pCS, f0d, f1d);
    iterations_run = 1;

    if (im.params.verbose_every > 0)
      std::printf("[ot] iter 0 residual %.6e\n", residual);

    if (residual >= im.params.tol && im.params.max_iter > 1) {
      if (im.params.use_cuda_graph && im.ensure_graph_ready(w)) {
        used_cuda_graph = true;
        const int K = std::max(1, im.params.cuda_graph_batch_size);
        double* rsq = thrust::raw_pointer_cast(im.residual_sq_dev.data());
        int it = 1;
        int last_print = 0;

          while (it < im.params.max_iter) {
            int batch = std::min(K, im.params.max_iter - it);
            for (int r = 0; r < batch; ++r)
              cudaGraphLaunch(im.graph_exec_, cudaStreamPerThread);

            double sumsq = 0.0;
            cudaMemcpyAsync(&sumsq, rsq, sizeof(double), cudaMemcpyDeviceToHost, cudaStreamPerThread);
            cudaStreamSynchronize(cudaStreamPerThread);
            residual = im.params.alpha * std::sqrt(sumsq);
            it += batch;
            iterations_run = it;

            if (im.params.verbose_every > 0 && it - last_print >= im.params.verbose_every) {
              std::printf("[ot] iter %d (batch=%d, graphed) residual %.6e\n", it, batch, residual);
              last_print = it;
            }
            if (residual < im.params.tol) break;
        }
      } else {
        for (int it = 1; it < im.params.max_iter; ++it) {
          residual = dr_iteration_step(im.L, w, im.proxC, im.proxCS, z1, z2, z3, xp,
                                       pp, pJ, pC, pCS, f0d, f1d);
          iterations_run = it + 1;
          if (im.params.verbose_every > 0 && (it % im.params.verbose_every == 0))
            std::printf("[ot] iter %d residual %.6e\n", it, residual);
          if (residual < im.params.tol) break;
        }
      }
    }
  }

  OTResult result;
  result.f.resize((size_t)g.size_c());
  result.mx.resize((size_t)g.size_c());
  result.my.resize((size_t)g.size_c());
  thrust::copy_n(im.x.begin() + im.L.off_f_c, g.size_c(), result.f.begin());
  thrust::copy_n(im.x.begin() + im.L.off_mx_c, g.size_c(), result.mx.begin());
  thrust::copy_n(im.x.begin() + im.L.off_my_c, g.size_c(), result.my.begin());

  // W2^2 = 2 * J(V*).
  result.W2_squared = 2.0 * eval_J(g, xp + im.L.off_mx_c, xp + im.L.off_my_c, xp + im.L.off_f_c);
  result.iterations_run = iterations_run;
  result.final_residual = residual;
  result.used_cuda_graph = used_cuda_graph;
  
  return result;
}

} // namespace ot
