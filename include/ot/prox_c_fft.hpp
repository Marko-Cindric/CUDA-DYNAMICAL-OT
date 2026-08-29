#pragma once
// Direct spectral solve of AA*(phi) = rhs, where
//   AA*(phi) := div( zero_boundary( div*(phi) ) )

// --- How the solve is computed ---
// A 3D DCT-II forward transform (applied one axis at a time; each 1D pass
// mirror-extends the M samples to a length-2M signal, real-FFTs it, and
// extracts C[k] = 0.5*Re(exp(-i*pi*k/(2M))*Y[k]) -- which turns rhs 
// into spectral coefficients; dividing elementwise by Lambda(l,m,n) 
// (except the zero mode, left at 0) and transforming back with the 
// matching DCT-III inverse (again one axis at a time) gives phi directly, 
// in O(N^3 log N).

#include "ot/grid.cuh"
#include <cufft.h>
#include <thrust/device_vector.h>

namespace ot {

class PoissonFFTSolver {
public:
  explicit PoissonFFTSolver(GridDims g);
  ~PoissonFFTSolver();
  PoissonFFTSolver(const PoissonFFTSolver&) = delete;
  PoissonFFTSolver& operator=(const PoissonFFTSolver&) = delete;

  // Solve AA*(phi) = rhs for phi.
  void solve(const double* rhs_c, double* phi_c);

  // Binds all 6 plans to `stream` via cufftSetStream. 
  void set_stream(cudaStream_t stream);

  // Debug/test-only hooks exposing the x-axis forward/inverse transform in
  // isolation, used by fft_poisson_tests.cu to validate the FFT-based
  // DCT-II/DCT-III formulas against a brute-force reference before
  // trusting the full 3D solve.
  void debug_forward_x(const double* in, double* out) { run_forward_axis(plan_x_fwd_, 1, Mx_, My_ * Mt_, in, out); }
  void debug_inverse_x(const double* in, double* out) { run_inverse_axis(plan_x_inv_, 1, Mx_, My_ * Mt_, in, out); }

private:
  GridDims g_;
  long long Mx_, My_, Mt_;

  // 6 plans: forward (D2Z) and inverse (Z2D) for each of the 3 axes.
  cufftHandle plan_x_fwd_, plan_x_inv_;
  cufftHandle plan_y_fwd_, plan_y_inv_;
  cufftHandle plan_t_fwd_, plan_t_inv_;

  thrust::device_vector<double> inv_denom_;     // precomputed 1/(norm*Lambda), size_c(), 0 at the null mode
  thrust::device_vector<double> buf_a_, buf_b_; // real coefficient arrays, size_c() each
  thrust::device_vector<cufftDoubleComplex> cplx_scratch_; // reused complex (M+1-bin) scratch, sized for the largest axis pass
  thrust::device_vector<double> real_scratch_;             // reused real (2M-element, mirror-padded) scratch, sized for the largest axis pass
  // Transposed-layout scratch for the y-axis pass (y made contiguous)
  thrust::device_vector<double> y_trans_a_, y_trans_b_;

  void run_forward_axis(cufftHandle plan, long long outer, long long M, long long inner,
                        const double* in, double* out);
  void run_inverse_axis(cufftHandle plan, long long outer, long long M, long long inner,
                        const double* in, double* out);
  // y-axis passes: unlike x and t, the y-axis's batch structure can't be 
  // expressed as a single cufftPlanMany call. These wrappers instead 
  // transpose (x,y,t) -> (x,t,y) first making y contiguous so the transform 
  // becomes the same shape x and t already use, then transpose the result back.
  void run_forward_axis_y(const double* in, double* out);
  void run_inverse_axis_y(const double* in, double* out);
};

} // namespace ot
