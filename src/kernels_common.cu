// Generic elementwise / reduction utilities operating on flat double buffers.
// Used for the JointLayout-shaped DR state (z1,z2,z3,x,p,...) and for
// centered-grid vectors elsewhere in the codebase and its tests.

#include "ot/kernels.hpp"
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>

namespace ot {

namespace {

constexpr int kBlock = 256;
inline int grid_for(long long n) { return (int)((n + kBlock - 1) / kBlock); }

// Standard shared-memory tree reduction within a block.
__device__ inline double block_reduce_sum(double val) {
  __shared__ double sdata[kBlock];
  int t = threadIdx.x;
  sdata[t] = val;
  __syncthreads();

  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (t < s) sdata[t] += sdata[t + s];
    __syncthreads();
  }

  return sdata[0];
}

// Device-resident version of residual_p_minus_x's sum((p-x)^2).
__global__ void residual_sq_kernel(const double* p, const double* x, long long n, double* out_dev) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  double contrib = 0.0;

  if (tid < n) {
      double d = p[tid] - x[tid];
      contrib = d * d;
  }

  double partial = block_reduce_sum(contrib);
  if (threadIdx.x == 0) atomicAdd(out_dev, partial);
}

__global__ void lincomb3_kernel(double* out, double w1, const double* a, double w2, const double* b,
                                double w3, const double* c, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;

  out[i] = w1 * a[i] + w2 * b[i] + w3 * c[i];
}

__global__ void sub_kernel(double* out, const double* a, const double* b, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = a[i] - b[i];
}

__global__ void add_kernel(double* out, const double* a, const double* b, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = a[i] + b[i];
}

__global__ void add_inplace_kernel(double* a, const double* b, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] += b[i];
}

__global__ void sub_inplace_kernel(double* a, const double* b, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] -= b[i];
}

__global__ void dr_zupdate_kernel(double* z, const double* p, const double* x, const double* pi,
                                   double alpha, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  
  z[i] += alpha * (2.0 * p[i] - x[i] - pi[i]);
}

__global__ void dr_xupdate_kernel(double* x, const double* p, double alpha, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += alpha * (p[i] - x[i]);
}

struct SquareDiffPX {

  __host__ __device__ double operator()(const thrust::tuple<double,double>& t) const {
    double val = thrust::get<0>(t) - thrust::get<1>(t);

    return val * val;
  }
};

} // namespace

void launch_lincomb3(double* out, double w1, const double* a, double w2, const double* b,
                     double w3, const double* c, long long n) {
  if (n <= 0) return;
  lincomb3_kernel<<<grid_for(n), kBlock>>>(out, w1, a, w2, b, w3, c, n);
}

void launch_sub(double* out, const double* a, const double* b, long long n) {
  if (n <= 0) return;
  sub_kernel<<<grid_for(n), kBlock>>>(out, a, b, n);
}

void launch_add(double* out, const double* a, const double* b, long long n) {
  if (n <= 0) return;
  add_kernel<<<grid_for(n), kBlock>>>(out, a, b, n);
}

void launch_add_inplace(double* a, const double* b, long long n) {
  if (n <= 0) return;
  add_inplace_kernel<<<grid_for(n), kBlock>>>(a, b, n);
}

void launch_sub_inplace(double* a, const double* b, long long n) {
  if (n <= 0) return;
  sub_inplace_kernel<<<grid_for(n), kBlock>>>(a, b, n);
}

void launch_dr_zupdate(double* z, const double* p, const double* x, const double* pi,
                       double alpha, long long n) {
  if (n <= 0) return;

  dr_zupdate_kernel<<<grid_for(n), kBlock>>>(z, p, x, pi, alpha, n);
}

void launch_dr_xupdate(double* x, const double* p, double alpha, long long n) {
  if (n <= 0) return;
  dr_xupdate_kernel<<<grid_for(n), kBlock>>>(x, p, alpha, n);
}

double dot(const double* a, const double* b, long long n) {
  if (n <= 0) return 0.0;
  thrust::device_ptr<const double> pa(a), pb(b);

  return thrust::inner_product(pa, pa + n, pb, 0.0);
}

double norm2(const double* a, long long n) {
  return std::sqrt(dot(a, a, n));
}

double residual_p_minus_x(const double* p, const double* x, double alpha, long long n) {
  if (n <= 0) return 0.0;
  thrust::device_ptr<const double> pp(p), px(x);
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(pp, px));
  auto end = thrust::make_zip_iterator(thrust::make_tuple(pp + n, px + n));
  double sumsq = thrust::transform_reduce(begin, end, SquareDiffPX{}, 0.0, thrust::plus<double>());

  return alpha * std::sqrt(sumsq);
}

void launch_residual_sq_to_device(const double* p, const double* x, long long n, double* residual_sq_dev) {
  if (n <= 0) return;
  residual_sq_kernel<<<grid_for(n), kBlock>>>(p, x, n, residual_sq_dev);
}

} // namespace ot
