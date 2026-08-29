#include "ot/dr_iteration.hpp"
#include "ot/kernels.hpp"
#include <cuda_runtime.h>

namespace ot {

double dr_iteration_step(const JointLayout& L, const DRWeights& w,
                          ProxC& proxC, ProxConsensus& proxCS,
                          double* z1, double* z2, double* z3, double* x,
                          double* p, double* pJ, double* pC, double* pCS,
                          const double* f0, const double* f1) {
  const GridDims& g = L.g;
  const size_t bytes = (size_t)L.total * sizeof(double);
  // U and V are contiguous blocks in JointLayout ([U][V]), so off_mx_c is
  // exactly the U/V split point.
  const size_t u_bytes = (size_t)L.off_mx_c * sizeof(double);

  // p_J = (z1.U, Prox_{gamma J}(z1.V)). Only U needs copying -- V is
  // computed directly from z1's V into pJ's V.
  cudaMemcpy(pJ, z1, u_bytes, cudaMemcpyDeviceToDevice);
  launch_cubic_prox(g, L.mx_c(z1), L.my_c(z1), L.f_c(z1), w.gamma,
                     L.mx_c(pJ), L.my_c(pJ), L.f_c(pJ));

  // p_C = (Prox_{gamma iota_C}(z2.U), z2.V). Unlike p_J, this needs a full
  // copy: ProxC::apply's clamp+div sequence reads z2's *interior* U values
  // before it has any other way to see them and V must be carried through 
  // for the combine step below even though ProxC never touches it.
  cudaMemcpy(pC, z2, bytes, cudaMemcpyDeviceToDevice);
  proxC.apply(L.mx_s(pC), L.my_s(pC), L.f_s(pC), f0, f1);

  // p_cs = Prox_{gamma iota_{C_c,s}}(z3.U, z3.V), joint. No copy at all:
  // ProxConsensus::apply reads z3 directly and writes pCS directly.
  proxCS.apply(L.mx_s(z3), L.my_s(z3), L.f_s(z3), L.mx_c(z3), L.my_c(z3), L.f_c(z3),
               L.mx_s(pCS), L.my_s(pCS), L.f_s(pCS), L.mx_c(pCS), L.my_c(pCS), L.f_c(pCS));

  // p = w1*p_J + w2*p_C + w3*p_cs
  launch_lincomb3(p, w.omega1, pJ, w.omega2, pC, w.omega3, pCS, L.total);

  // z_i += alpha*(2p - x - p_i), using the *old* x.
  launch_dr_zupdate(z1, p, x, pJ, w.alpha, L.total);
  launch_dr_zupdate(z2, p, x, pC, w.alpha, L.total);
  launch_dr_zupdate(z3, p, x, pCS, w.alpha, L.total);

  // residual = ||x_new - x_old|| = alpha*||p - x_old||, then x += alpha*(p - x).
  double residual = residual_p_minus_x(p, x, w.alpha, L.total);
  launch_dr_xupdate(x, p, w.alpha, L.total);
  return residual;
}

void dr_iteration_step_capturable(const JointLayout& L, const DRWeights& w,
                                   ProxC& proxC, ProxConsensus& proxCS,
                                   double* z1, double* z2, double* z3, double* x,
                                   double* p, double* pJ, double* pC, double* pCS,
                                   const double* f0, const double* f1,
                                   double* residual_sq_dev) {
  const GridDims& g = L.g;
  const size_t bytes = (size_t)L.total * sizeof(double);
  const size_t u_bytes = (size_t)L.off_mx_c * sizeof(double);

  // Same sequence as dr_iteration_step except cudaMemcpy ->
  // cudaMemcpyAsync (a blocking cudaMemcpy is not capture-safe) and the
  // residual is accumulated device-side instead of returned.
  cudaMemsetAsync(residual_sq_dev, 0, sizeof(double));
  cudaMemcpyAsync(pJ, z1, u_bytes, cudaMemcpyDeviceToDevice);
  launch_cubic_prox(g, L.mx_c(z1), L.my_c(z1), L.f_c(z1), w.gamma,
                     L.mx_c(pJ), L.my_c(pJ), L.f_c(pJ));

  cudaMemcpyAsync(pC, z2, bytes, cudaMemcpyDeviceToDevice);
  proxC.apply(L.mx_s(pC), L.my_s(pC), L.f_s(pC), f0, f1);

  proxCS.apply(L.mx_s(z3), L.my_s(z3), L.f_s(z3), L.mx_c(z3), L.my_c(z3), L.f_c(z3),
               L.mx_s(pCS), L.my_s(pCS), L.f_s(pCS), L.mx_c(pCS), L.my_c(pCS), L.f_c(pCS));

  launch_lincomb3(p, w.omega1, pJ, w.omega2, pC, w.omega3, pCS, L.total);

  launch_dr_zupdate(z1, p, x, pJ, w.alpha, L.total);
  launch_dr_zupdate(z2, p, x, pC, w.alpha, L.total);
  launch_dr_zupdate(z3, p, x, pCS, w.alpha, L.total);

  launch_residual_sq_to_device(p, x, L.total, residual_sq_dev);
  launch_dr_xupdate(x, p, w.alpha, L.total);
}

} // namespace ot
