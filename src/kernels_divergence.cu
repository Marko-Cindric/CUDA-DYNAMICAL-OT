// Space-time divergence div : E_s -> R^{D_c} and its full adjoint div*, plus the
// boundary utilities used to build Prox_{gamma * iota_C}.
//
// div(U)_{i,j,k} = N*(mx[i+1,j,k]-mx[i,j,k]) + P*(my[i,j+1,k]-my[i,j,k])
//                + Q*(f[i,j,k+1]-f[i,j,k])                      (i=0..N etc.)

#include "ot/kernels.hpp"

namespace ot {

namespace {
constexpr int kBlock = 256;
inline int grid_for(long long n) { return (int)((n + kBlock - 1) / kBlock); }

__global__ void div_kernel(GridDims g, const double* mx_s, const double* my_s, const double* f_s,
                           double* out_c) {
  long long n = g.size_c();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int i, j, k;
  decode_c(g, tid, i, j, k);
  double dx = g.N * (mx_s[idx_mx(g, i + 1, j, k)] - mx_s[idx_mx(g, i, j, k)]);
  double dy = g.P * (my_s[idx_my(g, i, j + 1, k)] - my_s[idx_my(g, i, j, k)]);
  double dt = g.Q * (f_s[idx_f(g, i, j, k + 1)]  - f_s[idx_f(g, i, j, k)]);
  out_c[tid] = dx + dy + dt;
}

__global__ void div_adjoint_x_kernel(GridDims g, const double* phi_c, double* out_mx_s) {
  long long n = g.size_mx();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int k = (int)(tid % g.nt_mx());
  long long tmp = tid / g.nt_mx();
  int j = (int)(tmp % g.ny_mx());
  int a = (int)(tmp / g.ny_mx());
  double val = 0.0;
  if (a - 1 >= 0 && a - 1 <= g.N) val += g.N * phi_c[idx_c(g, a - 1, j, k)];
  if (a >= 0 && a <= g.N)         val -= g.N * phi_c[idx_c(g, a, j, k)];
  out_mx_s[tid] = val;
}

__global__ void div_adjoint_y_kernel(GridDims g, const double* phi_c, double* out_my_s) {
  long long n = g.size_my();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int k = (int)(tid % g.nt_my());
  long long tmp = tid / g.nt_my();
  int b = (int)(tmp % g.ny_my());
  int i = (int)(tmp / g.ny_my());
  double val = 0.0;
  if (b - 1 >= 0 && b - 1 <= g.P) val += g.P * phi_c[idx_c(g, i, b - 1, k)];
  if (b >= 0 && b <= g.P)         val -= g.P * phi_c[idx_c(g, i, b, k)];
  out_my_s[tid] = val;
}

__global__ void div_adjoint_t_kernel(GridDims g, const double* phi_c, double* out_f_s) {
  long long n = g.size_f();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int c = (int)(tid % g.nt_f());
  long long tmp = tid / g.nt_f();
  int j = (int)(tmp % g.ny_f());
  int i = (int)(tmp / g.ny_f());
  double val = 0.0;
  if (c - 1 >= 0 && c - 1 <= g.Q) val += g.Q * phi_c[idx_c(g, i, j, c - 1)];
  if (c >= 0 && c <= g.Q)         val -= g.Q * phi_c[idx_c(g, i, j, c)];
  out_f_s[tid] = val;
}

__global__ void zero_boundary_mx_kernel(GridDims g, double* mx_s) {
  // a=0 and a=N+1 slots, for all (j,k)
  long long n = (long long)g.ny_mx() * g.nt_mx();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int k = (int)(tid % g.nt_mx());
  int j = (int)(tid / g.nt_mx());
  mx_s[idx_mx(g, 0, j, k)] = 0.0;
  mx_s[idx_mx(g, g.N + 1, j, k)] = 0.0;
}

__global__ void zero_boundary_my_kernel(GridDims g, double* my_s) {
  long long n = (long long)g.nx_my() * g.nt_my();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int k = (int)(tid % g.nt_my());
  int i = (int)(tid / g.nt_my());
  my_s[idx_my(g, i, 0, k)] = 0.0;
  my_s[idx_my(g, i, g.P + 1, k)] = 0.0;
}

__global__ void zero_boundary_f_kernel(GridDims g, double* f_s) {
  long long n = (long long)g.nx_f() * g.ny_f();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int j = (int)(tid % g.ny_f());
  int i = (int)(tid / g.ny_f());
  f_s[idx_f(g, i, j, 0)] = 0.0;
  f_s[idx_f(g, i, j, g.Q + 1)] = 0.0;
}

__global__ void clamp_boundary_f_kernel(GridDims g, double* f_s, const double* f0, const double* f1) {
  long long n = (long long)g.nx_f() * g.ny_f();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;
  
  int j = (int)(tid % g.ny_f());
  int i = (int)(tid / g.ny_f());
  long long ij = (long long)i * g.ny_f() + j; // f0,f1 are (N+1)x(P+1) row-major
  f_s[idx_f(g, i, j, 0)] = f0[ij];
  f_s[idx_f(g, i, j, g.Q + 1)] = f1[ij];
}

} // namespace

void launch_div(const GridDims& g, const double* mx_s, const double* my_s, const double* f_s,
                double* out_c) {
  long long n = g.size_c();
  if (n <= 0) return;
  div_kernel<<<grid_for(n), kBlock>>>(g, mx_s, my_s, f_s, out_c);
}

void launch_div_adjoint(const GridDims& g, const double* phi_c, double* mx_s, double* my_s, double* f_s) {
  div_adjoint_x_kernel<<<grid_for(g.size_mx()), kBlock>>>(g, phi_c, mx_s);
  div_adjoint_y_kernel<<<grid_for(g.size_my()), kBlock>>>(g, phi_c, my_s);
  div_adjoint_t_kernel<<<grid_for(g.size_f()), kBlock>>>(g, phi_c, f_s);
}

void launch_zero_all_boundary(const GridDims& g, double* mx_s, double* my_s, double* f_s) {
  zero_boundary_mx_kernel<<<grid_for((long long)g.ny_mx() * g.nt_mx()), kBlock>>>(g, mx_s);
  zero_boundary_my_kernel<<<grid_for((long long)g.nx_my() * g.nt_my()), kBlock>>>(g, my_s);
  zero_boundary_f_kernel<<<grid_for((long long)g.nx_f() * g.ny_f()), kBlock>>>(g, f_s);
}

void launch_clamp_input_boundary(const GridDims& g, double* mx_s, double* my_s, double* f_s,
                                 const double* f0, const double* f1) {
  zero_boundary_mx_kernel<<<grid_for((long long)g.ny_mx() * g.nt_mx()), kBlock>>>(g, mx_s);
  zero_boundary_my_kernel<<<grid_for((long long)g.nx_my() * g.nt_my()), kBlock>>>(g, my_s);
  clamp_boundary_f_kernel<<<grid_for((long long)g.nx_f() * g.ny_f()), kBlock>>>(g, f_s, f0, f1);
}

} // namespace ot
