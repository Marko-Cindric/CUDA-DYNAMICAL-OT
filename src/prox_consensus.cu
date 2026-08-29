#include "ot/prox_consensus.hpp"
#include "ot/kernels.hpp"
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vector>

namespace ot {

namespace {
constexpr int kBlock = 256;
inline int grid_for(long long n) { return (int)((n + kBlock - 1) / kBlock); }

constexpr double kA = 0.25, kB = 1.5, kC = 0.25; // tridiag(0.25, 1.5, 0.25), constant coefficients

// Precompute the Thomas forward-elimination coefficients c'_i for a
// constant-coefficient tridiagonal system of length L.
std::vector<double> precompute_cprime(int L) {
  std::vector<double> cprime(L, 0.0);
  cprime[0] = kC / kB;

  for (int i = 1; i < L - 1; ++i) {
    double denom = kB - kA * cprime[i - 1];
    cprime[i] = kC / denom;
  }

  return cprime;
}

// One thread per line; sequential forward+backward substitution along the
// line, using `out` as scratch for the forward pass's d' values.
__global__ void thomas_solve_x_kernel(GridDims g, const double* cprime, const double* rhs, double* out) {
  long long lines = (long long)g.ny_c() * g.nt_c();
  long long lid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (lid >= lines) return;

  int k = (int)(lid % g.nt_c());
  int j = (int)(lid / g.nt_c());
  int L = g.nx_c();

  double dprev = rhs[idx_c(g, 0, j, k)] / kB;
  out[idx_c(g, 0, j, k)] = dprev;
  for (int i = 1; i < L; ++i) {
    double denom = kB - kA * cprime[i - 1];
    double di = (rhs[idx_c(g, i, j, k)] - kA * dprev) / denom;
    out[idx_c(g, i, j, k)] = di;
    dprev = di;
  }

  double xnext = out[idx_c(g, L - 1, j, k)];
  for (int i = L - 2; i >= 0; --i) {
    double xi = out[idx_c(g, i, j, k)] - cprime[i] * xnext;
    out[idx_c(g, i, j, k)] = xi;
    xnext = xi;
  }
}

__global__ void thomas_solve_y_kernel(GridDims g, const double* cprime, const double* rhs, double* out) {
  long long lines = (long long)g.nx_c() * g.nt_c();
  long long lid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (lid >= lines) return;

  int k = (int)(lid % g.nt_c());
  int i = (int)(lid / g.nt_c());
  int L = g.ny_c();

  double dprev = rhs[idx_c(g, i, 0, k)] / kB;
  out[idx_c(g, i, 0, k)] = dprev;
  for (int j = 1; j < L; ++j) {
    double denom = kB - kA * cprime[j - 1];
    double dj = (rhs[idx_c(g, i, j, k)] - kA * dprev) / denom;
    out[idx_c(g, i, j, k)] = dj;
    dprev = dj;
  }

  double xnext = out[idx_c(g, i, L - 1, k)];
  for (int j = L - 2; j >= 0; --j) {
    double xj = out[idx_c(g, i, j, k)] - cprime[j] * xnext;
    out[idx_c(g, i, j, k)] = xj;
    xnext = xj;
  }
}

__global__ void thomas_solve_t_kernel(GridDims g, const double* cprime, const double* rhs, double* out) {
  long long lines = (long long)g.nx_c() * g.ny_c();
  long long lid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (lid >= lines) return;

  int j = (int)(lid % g.ny_c());
  int i = (int)(lid / g.ny_c());
  int L = g.nt_c();

  double dprev = rhs[idx_c(g, i, j, 0)] / kB;
  out[idx_c(g, i, j, 0)] = dprev;
  for (int k = 1; k < L; ++k) {
    double denom = kB - kA * cprime[k - 1];
    double dk = (rhs[idx_c(g, i, j, k)] - kA * dprev) / denom;
    out[idx_c(g, i, j, k)] = dk;
    dprev = dk;
  }

  double xnext = out[idx_c(g, i, j, L - 1)];
  for (int k = L - 2; k >= 0; --k) {
    double xk = out[idx_c(g, i, j, k)] - cprime[k] * xnext;
    out[idx_c(g, i, j, k)] = xk;
    xnext = xk;
  }
}

} // namespace

ProxConsensus::ProxConsensus(GridDims g)
  : g_(g),
    cprime_x_(g.nx_c(), 0.0), cprime_y_(g.ny_c(), 0.0), cprime_t_(g.nt_c(), 0.0),
    w_mx_(g.size_c(), 0.0), w_my_(g.size_c(), 0.0), w_f_(g.size_c(), 0.0),
    corr_mx_(g.size_mx(), 0.0), corr_my_(g.size_my(), 0.0), corr_f_(g.size_f(), 0.0) {

  thrust::host_vector<double> hx(precompute_cprime(g.nx_c()));
  thrust::host_vector<double> hy(precompute_cprime(g.ny_c()));
  thrust::host_vector<double> ht(precompute_cprime(g.nt_c()));
  cprime_x_ = hx;
  cprime_y_ = hy;
  cprime_t_ = ht;
}

void ProxConsensus::apply(const double* src_mx_s, const double* src_my_s, const double* src_f_s,
                          const double* src_mx_c, const double* src_my_c, const double* src_f_c,
                          double* dst_mx_s, double* dst_my_s, double* dst_f_s,
                          double* dst_mx_c, double* dst_my_c, double* dst_f_c) {

  double* w_mx = thrust::raw_pointer_cast(w_mx_.data());
  double* w_my = thrust::raw_pointer_cast(w_my_.data());
  double* w_f  = thrust::raw_pointer_cast(w_f_.data());

  // w = I(U~) - V~
  launch_interp(g_, src_mx_s, src_my_s, src_f_s, w_mx, w_my, w_f);
  launch_sub_inplace(w_mx, src_mx_c, g_.size_c());
  launch_sub_inplace(w_my, src_my_c, g_.size_c());
  launch_sub_inplace(w_f,  src_f_c,  g_.size_c());

  // p = (BB*)^{-1} w, solved in place into w_mx,w_my,w_f
  const double* cx = thrust::raw_pointer_cast(cprime_x_.data());
  const double* cy = thrust::raw_pointer_cast(cprime_y_.data());
  const double* ct = thrust::raw_pointer_cast(cprime_t_.data());
  thomas_solve_x_kernel<<<grid_for((long long)g_.ny_c() * g_.nt_c()), kBlock>>>(g_, cx, w_mx, w_mx);
  thomas_solve_y_kernel<<<grid_for((long long)g_.nx_c() * g_.nt_c()), kBlock>>>(g_, cy, w_my, w_my);
  thomas_solve_t_kernel<<<grid_for((long long)g_.nx_c() * g_.ny_c()), kBlock>>>(g_, ct, w_f, w_f);

  // V_proj = V~ + p
  launch_add(dst_mx_c, src_mx_c, w_mx, g_.size_c());
  launch_add(dst_my_c, src_my_c, w_my, g_.size_c());
  launch_add(dst_f_c,  src_f_c,  w_f,  g_.size_c());

  // U_proj = U~ - I*(p)
  double* cmx = thrust::raw_pointer_cast(corr_mx_.data());
  double* cmy = thrust::raw_pointer_cast(corr_my_.data());
  double* cf  = thrust::raw_pointer_cast(corr_f_.data());
  launch_interp_adjoint(g_, w_mx, w_my, w_f, cmx, cmy, cf);
  launch_sub(dst_mx_s, src_mx_s, cmx, g_.size_mx());
  launch_sub(dst_my_s, src_my_s, cmy, g_.size_my());
  launch_sub(dst_f_s,  src_f_s,  cf,  g_.size_f());
}

} // namespace ot
