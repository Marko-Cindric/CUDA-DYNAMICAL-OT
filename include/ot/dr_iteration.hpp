#pragma once
// One iteration of the asymmetric Douglas-Rachford (A-DR)
// splitting over the three simple functionals:
//   p_J  = (z.U, Prox_{gamma J}(z.V))                  -- leaves U untouched
//   p_C  = (Prox_{gamma iota_C}(z.U), z.V)              -- leaves V untouched
//   p_cs = Prox_{gamma iota_{C_c,s}}(z.U, z.V)           -- joint update
//   p = w1*p_J + w2*p_C + w3*p_cs
//   z_i += alpha*(2p - x - p_i),  i=J,C,cs
//   x   += alpha*(p - x)

#include "ot/grid.cuh"
#include "ot/joint_field.hpp"
#include "ot/prox_c.hpp"
#include "ot/prox_consensus.hpp"

namespace ot {

struct DRWeights {
  double omega1, omega2, omega3; // weight on p_J, p_C, p_cs
  double alpha;
  double gamma;
};

// z1,z2,z3,x and the scratch buffers p,pJ,pC,pCS are all JointLayout-shaped
// device buffers of length layout.total. f0,f1 are device pointers of size
// (N+1)*(P+1) (row-major i*(P+1)+j).
double dr_iteration_step(const JointLayout& layout, const DRWeights& weights,
                          ProxC& proxC, ProxConsensus& proxCS,
                          double* z1, double* z2, double* z3, double* x,
                          double* p, double* pJ, double* pC, double* pCS,
                          const double* f0, const double* f1);

// Capture-safe variant of dr_iteration_step
void dr_iteration_step_capturable(const JointLayout& layout, const DRWeights& weights,
                                   ProxC& proxC, ProxConsensus& proxCS,
                                   double* z1, double* z2, double* z3, double* x,
                                   double* p, double* pJ, double* pC, double* pCS,
                                   const double* f0, const double* f1,
                                   double* residual_sq_dev);

} // namespace ot
