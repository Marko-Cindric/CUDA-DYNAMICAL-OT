#include "ot/prox_c_fft.hpp"
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ot {

namespace {
constexpr int kBlock = 256;
constexpr double kPi = 3.14159265358979323846;
inline int grid_for(long long n) { return (int)((n + kBlock - 1) / kBlock); }

void cufft_check(cufftResult r, const char* what) {
  if (r != CUFFT_SUCCESS)
    throw std::runtime_error(std::string("cufft error in ") + what + ": code " + std::to_string((int)r));
}

// ---- forward pass, step 1: mirror-extend x[0..M-1] to y[0..2M-1] via
// y[n] = x[n] for n<M, y[n] = x[2M-1-n] for n>=M -- the "half-sample symmetric" 
// extension whose length-2M DFT carries the DCT-II coefficients.
__global__ void dct2_mirror_pad_kernel(const double* x, double* y,
                                       long long outer, long long M, long long inner) {
  long long twoM = 2 * M;
  long long n = outer * twoM * inner;
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  long long in_ = tid % inner;
  long long tmp = tid / inner;
  long long p = tmp % twoM;
  long long o = tmp / twoM;
  long long src = (p < M) ? p : (twoM - 1 - p);

  y[tid] = x[(o * M + src) * inner + in_];
}

// ---- forward pass, step 2: extract the DCT-II coefficients from the
// (M+1)-bin half-spectrum Y produced by cufftExecD2Z(length 2M) of the
// mirror-extended signal, using
//   C[k] = 0.5 * Re( exp(-i*pi*k/(2M)) * Y[k] ),  k=0..M-1
// (all needed bins k=0..M-1 are directly available in Y[0..M], no
// conjugate-symmetry extension needed).
__global__ void dct2_extract_kernel(const cufftDoubleComplex* Y, double* C,
                                    long long outer, long long M, long long inner) {
  long long n = outer * M * inner;
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  long long in_ = tid % inner;
  long long tmp = tid / inner;
  long long k = tmp % M;
  long long o = tmp / M;
  long long Mp1 = M + 1;

  cufftDoubleComplex y = Y[(o * Mp1 + k) * inner + in_];
  double theta = -kPi * (double)k / (2.0 * (double)M);
  double c = cos(theta), s = sin(theta);

  C[tid] = 0.5 * (c * y.x - s * y.y);
}

// ---- inverse pass, step 1: embed real DCT-space coefficients c[0..M-1]
// into the (M+1)-bin half-spectrum Z that a length-2M cufftExecZ2D expects,
// via
//   Z[0] = c[0],  Z[l] = c[l]*exp(i*pi*l/(2M))/2  (1<=l<=M-1),  Z[M] = 0.
__global__ void dct3_embed_kernel(const double* c, cufftDoubleComplex* Z,
                                  long long outer, long long M, long long inner) {
  long long Mp1 = M + 1;
  long long n = outer * Mp1 * inner;
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  long long in_ = tid % inner;
  long long tmp = tid / inner;
  long long l = tmp % Mp1;
  long long o = tmp / Mp1;

  cufftDoubleComplex z;

  if (l == 0) {
    z.x = c[(o * M + 0) * inner + in_];
    z.y = 0.0;
  } else if (l == M) {
    z.x = 0.0;
    z.y = 0.0;
  } else {
    double cv = c[(o * M + l) * inner + in_];
    double theta = kPi * (double)l / (2.0 * (double)M);
    z.x = 0.5 * cv * cos(theta);
    z.y = 0.5 * cv * sin(theta);
  }

  Z[tid] = z;
}

// ---- inverse pass, step 2: cufftExecZ2D produces 2M real samples per
// transform; only the first M are the desired DCT-III output (the rest are
// the implicit mirror image and are discarded).
__global__ void dct3_truncate_kernel(const double* R, double* out,
                                     long long outer, long long M, long long inner) {
  long long n = outer * M * inner;
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  long long in_ = tid % inner;
  long long tmp = tid / inner;
  long long i = tmp % M;
  long long o = tmp / M;

  out[tid] = R[(o * (2 * M) + i) * inner + in_];
}

__global__ void multiply_inplace_kernel(double* a, const double* b, long long n) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;

  if (tid < n) a[tid] *= b[tid];
}

// ---- y-axis transpose helpers: (x,y,t) <-> (x,t,y). Swapping y and t makes
// y contiguous, so its transform can use the same single-cufft-call
// "inner==1" shape as the t-axis.
__global__ void transpose_y_to_fast_kernel(const double* in, double* out,
                                           long long Mx, long long My, long long Mt) {
  long long n = Mx * My * Mt;
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  long long k = tid % Mt;
  long long tmp = tid / Mt;
  long long j = tmp % My;
  long long i = tmp / My;

  out[(i * Mt + k) * My + j] = in[tid];
}

__global__ void transpose_y_from_fast_kernel(const double* in, double* out,
                                              long long Mx, long long My, long long Mt) {
  long long n = Mx * My * Mt;
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  long long j = tid % My;
  long long tmp = tid / My;
  long long k = tmp % Mt;
  long long i = tmp / Mt;

  out[(i * My + j) * Mt + k] = in[tid];
}

} // namespace

PoissonFFTSolver::PoissonFFTSolver(GridDims g) : g_(g) {
  Mx_ = g.N + 1;
  My_ = g.P + 1;
  Mt_ = g.Q + 1;
  long long size_c = Mx_ * My_ * Mt_;

  // ---- eigenvalues + normalization, precomputed once on the host ----
  // lambda_axis(l) = axisDivisions^2 * 4*sin^2(l*pi/(2*M_axis)); norm(l) =
  // M_axis for l=0, M_axis/2 otherwise (||DCT-II basis vector||^2).
  auto lambda_norm = [](int divisions, long long M, long long l, double& lam, double& norm) {
    double s = sin((double)l * kPi / (2.0 * (double)M));
    lam = (double)divisions * (double)divisions * 4.0 * s * s;
    norm = (l == 0) ? (double)M : (double)M / 2.0;
  };

  std::vector<double> lam_x(Mx_), norm_x(Mx_), lam_y(My_), norm_y(My_), lam_t(Mt_), norm_t(Mt_);
  for (long long l = 0; l < Mx_; ++l) lambda_norm(g.N, Mx_, l, lam_x[l], norm_x[l]);
  for (long long m = 0; m < My_; ++m) lambda_norm(g.P, My_, m, lam_y[m], norm_y[m]);
  for (long long n = 0; n < Mt_; ++n) lambda_norm(g.Q, Mt_, n, lam_t[n], norm_t[n]);

  thrust::host_vector<double> h_inv_denom(size_c);
  for (long long l = 0; l < Mx_; ++l)
    for (long long m = 0; m < My_; ++m)
      for (long long n = 0; n < Mt_; ++n) {
        long long idx = (l * My_ + m) * Mt_ + n;
        double lam = lam_x[l] + lam_y[m] + lam_t[n];
        if (l == 0 && m == 0 && n == 0) {
          h_inv_denom[idx] = 0.0; // null mode, defined away
        } else {
          double denom = norm_x[l] * norm_y[m] * norm_t[n] * lam;
          h_inv_denom[idx] = 1.0 / denom;
        }
      }
  inv_denom_ = h_inv_denom;

  buf_a_.resize(size_c);
  buf_b_.resize(size_c);
  y_trans_a_.resize(size_c);
  y_trans_b_.resize(size_c);

  // ---- scratch sizing: largest of the 3 axes' complex (M+1 bins) and
  // real-padded (2M elements) per-line requirements -- both forward
  // (mirror-padded input / half-spectrum output) and inverse
  // (half-spectrum input / padded output) passes use the same shape.
  long long inner_x = My_ * Mt_, inner_y = Mt_, inner_t = 1;
  long long outer_x = 1, outer_y = Mx_, outer_t = Mx_ * My_;
  long long cplx_x = outer_x * (Mx_ + 1) * inner_x;
  long long cplx_y = outer_y * (My_ + 1) * inner_y;
  long long cplx_t = outer_t * (Mt_ + 1) * inner_t;
  long long cplx_max = std::max({cplx_x, cplx_y, cplx_t});
  long long real_x = outer_x * (2 * Mx_) * inner_x;
  long long real_y = outer_y * (2 * My_) * inner_y;
  long long real_t = outer_t * (2 * Mt_) * inner_t;
  long long real_max = std::max({real_x, real_y, real_t});
  cplx_scratch_.resize(cplx_max);
  real_scratch_.resize(real_max);

  // ---- cuFFT plans. Two "shapes":
  //  - inner==1 (t-axis, and now y-axis too via the transpose below): a
  //    single call covers the whole batch, batch=outer, istride=1,
  //    idist=M (fwd) / 2M (inv).
  //  - inner!=1 (x-axis; has outer=1 so its "loop" trivially runs once):
  //    looped over outer, batch=inner, istride=inner, idist=1.
  auto make_plan_inner1 = [&](cufftHandle& plan, long long M, long long outer, bool forward) {
    int n0 = (int)(2 * M);
    int real_n = (int)(2 * M), cplx_n = (int)(M + 1);
    int in_n = forward ? real_n : cplx_n;
    int out_n = forward ? cplx_n : real_n;
    int istride = 1, idist = forward ? real_n : cplx_n;
    int ostride = 1, odist = forward ? cplx_n : real_n;

    cufftType type = forward ? CUFFT_D2Z : CUFFT_Z2D;
    cufft_check(cufftPlanMany(&plan, 1, &n0, &in_n, istride, idist, &out_n, ostride, odist, type, (int)outer),
                "plan(inner=1)");
  };
  auto make_plan_looped = [&](cufftHandle& plan, long long M, long long inner, bool forward) {
    int n0 = (int)(2 * M);
    int real_n = (int)(2 * M), cplx_n = (int)(M + 1);
    int in_n = forward ? real_n : cplx_n;
    int out_n = forward ? cplx_n : real_n;
    int istride = (int)inner, idist = 1;
    int ostride = (int)inner, odist = 1;

    cufftType type = forward ? CUFFT_D2Z : CUFFT_Z2D;
    cufft_check(cufftPlanMany(&plan, 1, &n0, &in_n, istride, idist, &out_n, ostride, odist,
                              type, (int)inner),
                "plan(looped)");
  };

  make_plan_looped(plan_x_fwd_, Mx_, inner_x, true);
  make_plan_looped(plan_x_inv_, Mx_, inner_x, false);
  make_plan_inner1(plan_y_fwd_, My_, Mx_ * Mt_, true);
  make_plan_inner1(plan_y_inv_, My_, Mx_ * Mt_, false);
  make_plan_inner1(plan_t_fwd_, Mt_, outer_t, true);
  make_plan_inner1(plan_t_inv_, Mt_, outer_t, false);
}

void PoissonFFTSolver::set_stream(cudaStream_t stream) {
  cufft_check(cufftSetStream(plan_x_fwd_, stream), "setStream(x_fwd)");
  cufft_check(cufftSetStream(plan_x_inv_, stream), "setStream(x_inv)");
  cufft_check(cufftSetStream(plan_y_fwd_, stream), "setStream(y_fwd)");
  cufft_check(cufftSetStream(plan_y_inv_, stream), "setStream(y_inv)");
  cufft_check(cufftSetStream(plan_t_fwd_, stream), "setStream(t_fwd)");
  cufft_check(cufftSetStream(plan_t_inv_, stream), "setStream(t_inv)");
}

PoissonFFTSolver::~PoissonFFTSolver() {
  cufftDestroy(plan_x_fwd_);
  cufftDestroy(plan_x_inv_);
  cufftDestroy(plan_y_fwd_);
  cufftDestroy(plan_y_inv_);
  cufftDestroy(plan_t_fwd_);
  cufftDestroy(plan_t_inv_);
}

void PoissonFFTSolver::run_forward_axis(cufftHandle plan, long long outer, long long M, long long inner,
                                        const double* in, double* out) {
  cufftDoubleComplex* cplx = thrust::raw_pointer_cast(cplx_scratch_.data());
  double* real_pad = thrust::raw_pointer_cast(real_scratch_.data());
  dct2_mirror_pad_kernel<<<grid_for(outer * 2 * M * inner), kBlock>>>(in, real_pad, outer, M, inner);
  long long Mp1 = M + 1;

  if (inner == 1) {
    cufft_check(cufftExecD2Z(plan, real_pad, cplx), "execD2Z(inner=1)");
  } else {
    for (long long o = 0; o < outer; ++o) {
      double* r_o = real_pad + o * (2 * M) * inner;
      cufftDoubleComplex* c_o = cplx + o * Mp1 * inner;
      cufft_check(cufftExecD2Z(plan, r_o, c_o), "execD2Z(looped)");
    }
  }

  dct2_extract_kernel<<<grid_for(outer * M * inner), kBlock>>>(cplx, out, outer, M, inner);
}

void PoissonFFTSolver::run_inverse_axis(cufftHandle plan, long long outer, long long M, long long inner,
                                         const double* in, double* out) {
  cufftDoubleComplex* cplx = thrust::raw_pointer_cast(cplx_scratch_.data());
  double* real_pad = thrust::raw_pointer_cast(real_scratch_.data());
  long long Mp1 = M + 1;
  dct3_embed_kernel<<<grid_for(outer * Mp1 * inner), kBlock>>>(in, cplx, outer, M, inner);

  if (inner == 1) {
    cufft_check(cufftExecZ2D(plan, cplx, real_pad), "execZ2D(inner=1)");
  } else {
    for (long long o = 0; o < outer; ++o) {
      cufftDoubleComplex* c_o = cplx + o * Mp1 * inner;
      double* r_o = real_pad + o * (2 * M) * inner;
      cufft_check(cufftExecZ2D(plan, c_o, r_o), "execZ2D(looped)");
    }
  }

  dct3_truncate_kernel<<<grid_for(outer * M * inner), kBlock>>>(real_pad, out, outer, M, inner);
}

void PoissonFFTSolver::run_forward_axis_y(const double* in, double* out) {
  long long n = Mx_ * My_ * Mt_;
  double* ta = thrust::raw_pointer_cast(y_trans_a_.data());
  double* tb = thrust::raw_pointer_cast(y_trans_b_.data());

  transpose_y_to_fast_kernel<<<grid_for(n), kBlock>>>(in, ta, Mx_, My_, Mt_);
  run_forward_axis(plan_y_fwd_, Mx_ * Mt_, My_, 1, ta, tb);
  transpose_y_from_fast_kernel<<<grid_for(n), kBlock>>>(tb, out, Mx_, My_, Mt_);
}

void PoissonFFTSolver::run_inverse_axis_y(const double* in, double* out) {
  long long n = Mx_ * My_ * Mt_;
  double* ta = thrust::raw_pointer_cast(y_trans_a_.data());
  double* tb = thrust::raw_pointer_cast(y_trans_b_.data());

  transpose_y_to_fast_kernel<<<grid_for(n), kBlock>>>(in, ta, Mx_, My_, Mt_);
  run_inverse_axis(plan_y_inv_, Mx_ * Mt_, My_, 1, ta, tb);
  transpose_y_from_fast_kernel<<<grid_for(n), kBlock>>>(tb, out, Mx_, My_, Mt_);
}

void PoissonFFTSolver::solve(const double* rhs_c, double* phi_c) {
  long long inner_x = My_ * Mt_, inner_t = 1;
  long long outer_x = 1, outer_t = Mx_ * My_;

  double* a = thrust::raw_pointer_cast(buf_a_.data());
  double* b = thrust::raw_pointer_cast(buf_b_.data());

  // forward: x, then y, then t
  run_forward_axis(plan_x_fwd_, outer_x, Mx_, inner_x, rhs_c, a);
  run_forward_axis_y(a, b);
  run_forward_axis(plan_t_fwd_, outer_t, Mt_, inner_t, b, a);
  // a now holds B(l,m,n); divide by norm*Lambda (0 at the null mode)
  long long n = Mx_ * My_ * Mt_;
  multiply_inplace_kernel<<<grid_for(n), kBlock>>>(a, thrust::raw_pointer_cast(inv_denom_.data()), n);
  // inverse: t, then y, then x
  run_inverse_axis(plan_t_inv_, outer_t, Mt_, inner_t, a, b);
  run_inverse_axis_y(b, a);
  run_inverse_axis(plan_x_inv_, outer_x, Mx_, inner_x, a, phi_c);
}

} // namespace ot
