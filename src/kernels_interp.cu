// Interpolation I : E_s -> E_c (average staggered values onto the centered
// grid) and its exact adjoint I* : E_c -> E_s.
//
// I(U)^mx_{i,j,k} = (mx[i,j,k] + mx[i+1,j,k]) / 2,  i=0..N   (storage indices)
// I(U)^my_{i,j,k} = (my[i,j,k] + my[i,j+1,k]) / 2,  j=0..P
// I(U)^f_{i,j,k}  = (f[i,j,k]  + f[i,j,k+1])  / 2,  k=0..Q

#include "ot/kernels.hpp"

namespace ot {

namespace {
constexpr int kBlock = 256;
inline int grid_for(long long n) { return (int)((n + kBlock - 1) / kBlock); }

__global__ void interp_kernel(GridDims g, const double* mx_s, const double* my_s, const double* f_s,
                              double* mx_c, double* my_c, double* f_c) {
  long long n = g.size_c();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int i, j, k;
  decode_c(g, tid, i, j, k);
  mx_c[tid] = 0.5 * (mx_s[idx_mx(g, i, j, k)] + mx_s[idx_mx(g, i + 1, j, k)]);
  my_c[tid] = 0.5 * (my_s[idx_my(g, i, j, k)] + my_s[idx_my(g, i, j + 1, k)]);
  f_c[tid]  = 0.5 * (f_s[idx_f(g, i, j, k)]  + f_s[idx_f(g, i, j, k + 1)]);
}

__global__ void interp_adjoint_x_kernel(GridDims g, const double* psi_mx_c, double* out_mx_s) {
  long long n = g.size_mx();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int k = (int)(tid % g.nt_mx());
  long long tmp = tid / g.nt_mx();
  int j = (int)(tmp % g.ny_mx());
  int a = (int)(tmp / g.ny_mx());
  double val = 0.0;
  if (a - 1 >= 0 && a - 1 <= g.N) val += 0.5 * psi_mx_c[idx_c(g, a - 1, j, k)];
  if (a >= 0 && a <= g.N)         val += 0.5 * psi_mx_c[idx_c(g, a, j, k)];
  out_mx_s[tid] = val;
}

__global__ void interp_adjoint_y_kernel(GridDims g, const double* psi_my_c, double* out_my_s) {
  long long n = g.size_my();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int k = (int)(tid % g.nt_my());
  long long tmp = tid / g.nt_my();
  int b = (int)(tmp % g.ny_my());
  int i = (int)(tmp / g.ny_my());
  double val = 0.0;
  if (b - 1 >= 0 && b - 1 <= g.P) val += 0.5 * psi_my_c[idx_c(g, i, b - 1, k)];
  if (b >= 0 && b <= g.P)         val += 0.5 * psi_my_c[idx_c(g, i, b, k)];
  out_my_s[tid] = val;
}

__global__ void interp_adjoint_t_kernel(GridDims g, const double* psi_f_c, double* out_f_s) {
  long long n = g.size_f();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;
  
  int c = (int)(tid % g.nt_f());
  long long tmp = tid / g.nt_f();
  int j = (int)(tmp % g.ny_f());
  int i = (int)(tmp / g.ny_f());
  double val = 0.0;
  if (c - 1 >= 0 && c - 1 <= g.Q) val += 0.5 * psi_f_c[idx_c(g, i, j, c - 1)];
  if (c >= 0 && c <= g.Q)         val += 0.5 * psi_f_c[idx_c(g, i, j, c)];
  out_f_s[tid] = val;
}

} // namespace

void launch_interp(const GridDims& g, const double* mx_s, const double* my_s, const double* f_s,
                   double* mx_c, double* my_c, double* f_c) {
  long long n = g.size_c();
  if (n <= 0) return;
  interp_kernel<<<grid_for(n), kBlock>>>(g, mx_s, my_s, f_s, mx_c, my_c, f_c);
}

void launch_interp_adjoint(const GridDims& g, const double* mx_c, const double* my_c, const double* f_c,
                           double* mx_s, double* my_s, double* f_s) {
  interp_adjoint_x_kernel<<<grid_for(g.size_mx()), kBlock>>>(g, mx_c, mx_s);
  interp_adjoint_y_kernel<<<grid_for(g.size_my()), kBlock>>>(g, my_c, my_s);
  interp_adjoint_t_kernel<<<grid_for(g.size_f()), kBlock>>>(g, f_c, f_s);
}

} // namespace ot
