#pragma once
// Prox_{gamma * iota_C}: projection of the staggered field U=(mx,my,f) onto
// C = { div U = 0, b(U) = (0,0,f0,f1) }.
//
// Boundary slots are hard-clamped, so the projection reduces to:
//   1) clamp U's boundary slots to 0/0/f0/f1
//   2) solve  AA* phi = div(U_clamped)  on the centered grid, where
//      AA*(phi) := div( zero_boundary( div*(phi) ) ) -- the operator
//      of div* restricted to interior (free) staggered dofs. This is
//      symmetric positive definite by construction, and separable --
//      a discrete Neumann Laplacian -- so it's solved directly via a 
//      3D DCT-II/DCT-III spectral transform (PoissonFFTSolver, see prox_c_fft.hpp).
//      Exact to floating-point precision, O(N^3 log N).
//   3) U_proj = U_clamped - zero_boundary(div*(phi))
//
// V is untouched by this operator (only the U-half of the DR joint state is
// updated).

#include "ot/grid.cuh"
#include "ot/prox_c_fft.hpp"
#include <thrust/device_vector.h>
#include <memory>

namespace ot {

class ProxC {
public:
  explicit ProxC(GridDims g);

  // In-place: mx_s,my_s,f_s hold U~ on entry, U_proj on exit.
  // f0,f1 are device pointers of size (N+1)*(P+1), row-major i*(P+1)+j.
  void apply(double* mx_s, double* my_s, double* f_s, const double* f0, const double* f1);

  // Binds the FFT solve's cuFFT plans to `stream` so apply() becomes 
  // CUDA-Graph-capturable on that stream.
  bool bind_cuda_graph_stream(cudaStream_t stream);

private:
  GridDims g_;
  std::unique_ptr<PoissonFFTSolver> fft_solver_;

  // rhs (div(U_clamped)) / solution (phi) scratch on the centered grid (size_c())
  thrust::device_vector<double> phi_, r_;
  // scratch staggered buffers, needed only for the final correction step
  thrust::device_vector<double> tmp_mx_, tmp_my_, tmp_f_;
};

} // namespace ot
