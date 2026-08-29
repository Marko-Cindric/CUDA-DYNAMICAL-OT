#pragma once
// Prox_{gamma * iota_{C_c,s}}: projection of the joint pair (U,V) onto
// C_{c,s} = { (U,V) : V = I(U) }.
//
// With B(U,V) = I(U) - V, BB* = I*I* + Id is separable into three
// independent, constant-coefficient tridiagonal systems (one per component,
// applied along that component's own dimension: mx along x, my along y, f
// along t), each tridiag(0.25, 1.5, 0.25) of length nx_c()/ny_c()/nt_c().
// Solved exactly per line via a precomputed Thomas-algorithm forward sweep
// plus a per-line forward/backward substitution kernel.
//
//   w = I(U~) - V~              
//   p = (BB*)^{-1} w
//   U_proj = U~ - I*(p) ,   V_proj = V~ + p
//
// apply() takes separate src/dst pointers (rather than operating in place)
// so the caller never has to pre-copy its input into the output buffer --
// every output slot is written by this function from the src data directly.
// See dr_iteration.cu, which relies on this to skip a full state-vector
// copy every A-DR iteration.

#include "ot/grid.cuh"
#include <thrust/device_vector.h>

namespace ot {

class ProxConsensus {
public:
  explicit ProxConsensus(GridDims g);

  void apply(const double* src_mx_s, const double* src_my_s, const double* src_f_s,
             const double* src_mx_c, const double* src_my_c, const double* src_f_c,
             double* dst_mx_s, double* dst_my_s, double* dst_f_s,
             double* dst_mx_c, double* dst_my_c, double* dst_f_c);

private:
  GridDims g_;
  thrust::device_vector<double> cprime_x_, cprime_y_, cprime_t_; // precomputed Thomas coefficients
  thrust::device_vector<double> w_mx_, w_my_, w_f_;              // rhs / solution buffers (size_c())
  thrust::device_vector<double> corr_mx_, corr_my_, corr_f_;     // I*(p) scratch (staggered sizes)
};

} // namespace ot
