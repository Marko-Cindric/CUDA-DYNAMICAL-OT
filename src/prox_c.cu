// ProxC::apply

#include "ot/prox_c.hpp"
#include "ot/kernels.hpp"
#include <thrust/device_ptr.h>

namespace ot {

ProxC::ProxC(GridDims g)
    : g_(g), fft_solver_(std::make_unique<PoissonFFTSolver>(g)),
      phi_(g.size_c(), 0.0), r_(g.size_c(), 0.0),
      tmp_mx_(g.size_mx(), 0.0), tmp_my_(g.size_my(), 0.0), tmp_f_(g.size_f(), 0.0) {}

void ProxC::apply(double* mx_s, double* my_s, double* f_s, const double* f0, const double* f1) {
  // Step 1: clamp boundary slots to the prescribed data.
  launch_clamp_input_boundary(g_, mx_s, my_s, f_s, f0, f1);

  // Step 2: rhs = div(U_clamped); solve AA* phi = rhs directly via the
  // spectral (DCT-II/III) Poisson solve.
  double* phi = thrust::raw_pointer_cast(phi_.data());
  double* r = thrust::raw_pointer_cast(r_.data());
  launch_div(g_, mx_s, my_s, f_s, r);
  fft_solver_->solve(r, phi);

  // Step 3: U_proj = U_clamped - zero_boundary(div*(phi)). Runs once per
  // apply() call.
  double* tmx = thrust::raw_pointer_cast(tmp_mx_.data());
  double* tmy = thrust::raw_pointer_cast(tmp_my_.data());
  double* tf  = thrust::raw_pointer_cast(tmp_f_.data());

  launch_div_adjoint(g_, phi, tmx, tmy, tf);
  launch_zero_all_boundary(g_, tmx, tmy, tf);
  launch_sub_inplace(mx_s, tmx, g_.size_mx());
  launch_sub_inplace(my_s, tmy, g_.size_my());
  launch_sub_inplace(f_s, tf, g_.size_f());
}

bool ProxC::bind_cuda_graph_stream(cudaStream_t stream) {
  fft_solver_->set_stream(stream);
  
  return true;
}

} // namespace ot
