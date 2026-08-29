// Prox_{gamma J}: for each centered grid point, given target (mx~,my~,f~),
// solve
//   f^3 + (2*gamma - f~) f^2 + (gamma^2 - 2*gamma*f~) f - gamma^2*f~
//       - (gamma/2)*|m~|^2 = 0
// for the largest positive real root f*, then m = f* * m~ / (f* + gamma).
// Fully pointwise / separable -> one thread per grid point.
//
// Solved via Cardano's formula on the depressed cubic (trig form when there
// are three real roots, direct form otherwise), each candidate root polished
// with a few Newton iterations for robustness against cancellation.

#include "ot/kernels.hpp"
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <cmath>

namespace ot {

namespace {
constexpr int kBlock = 256;
inline int grid_for(long long n) { return (int)((n + kBlock - 1) / kBlock); }

__device__ inline double cubic_eval(double f, double b, double c, double d) {
  return ((f + b) * f + c) * f + d;
}
__device__ inline double cubic_deriv(double f, double b, double c) {
  return (3.0 * f + 2.0 * b) * f + c;
}
__device__ inline double newton_polish(double f, double b, double c, double d) {
#pragma unroll
  for (int it = 0; it < 3; ++it) {
    double fp = cubic_deriv(f, b, c);
    if (fabs(fp) < 1e-14) break;
    f -= cubic_eval(f, b, c, d) / fp;
  }

  return f;
}

__device__ double solve_prox_f(double ftil, double msq, double gamma) {
  double b = 2.0 * gamma - ftil;
  double c = gamma * gamma - 2.0 * gamma * ftil;
  double d = -gamma * gamma * ftil - 0.5 * gamma * msq;

  double p = c - b * b / 3.0;
  double q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
  double disc = (q * q) / 4.0 + (p * p * p) / 27.0;

  double roots[3];
  int nroots = 0;

  if (fabs(p) < 1e-12 && fabs(q) < 1e-12) {
    roots[0] = 0.0; nroots = 1;
  } else if (disc > 1e-14) {
    double sq = sqrt(disc);
    double u = cbrt(-q / 2.0 + sq);
    double v = cbrt(-q / 2.0 - sq);
    roots[0] = u + v; nroots = 1;
  } else if (fabs(p) < 1e-12) {
    roots[0] = cbrt(-q); nroots = 1;
  } else {
    double A = (3.0 * q) / (2.0 * p) * sqrt(-3.0 / p);
    A = fmax(-1.0, fmin(1.0, A));
    double phi = acos(A) / 3.0;
    double r = 2.0 * sqrt(-p / 3.0);
    const double two_pi_3 = 2.0943951023931953;
    roots[0] = r * cos(phi);
    roots[1] = r * cos(phi - two_pi_3);
    roots[2] = r * cos(phi - 2.0 * two_pi_3);
    nroots = 3;
  }

  double best = -1e300;
  for (int i = 0; i < nroots; ++i) {
    double f = newton_polish(roots[i] - b / 3.0, b, c, d);
    if (f > best) best = f;
  }

  return best > 0.0 ? best : 0.0;
}

__global__ void cubic_prox_kernel(GridDims g, const double* mx_t, const double* my_t, const double* f_t,
                                  double gamma, double* mx_out, double* my_out, double* f_out) {
  long long n = g.size_c();
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  double mxv = mx_t[tid], myv = my_t[tid], ftv = f_t[tid];
  double msq = mxv * mxv + myv * myv;
  double fstar = solve_prox_f(ftv, msq, gamma);

  if (fstar > 0.0) {
    double scale = fstar / (fstar + gamma);
    mx_out[tid] = mxv * scale;
    my_out[tid] = myv * scale;
    f_out[tid] = fstar;
  } else {
    mx_out[tid] = 0.0;
    my_out[tid] = 0.0;
    f_out[tid] = 0.0;
  }
}

struct JFunctor {

  __host__ __device__ double operator()(const thrust::tuple<double,double,double>& t) const {
    double mx = thrust::get<0>(t), my = thrust::get<1>(t), f = thrust::get<2>(t);
    if (f > 1e-14) return (mx * mx + my * my) / (2.0 * f);
    return 0.0;
  }
};

} // namespace

void launch_cubic_prox(const GridDims& g, const double* mx_t, const double* my_t, const double* f_t,
                       double gamma, double* mx_out, double* my_out, double* f_out) {
  long long n = g.size_c();
  if (n <= 0) return;
  cubic_prox_kernel<<<grid_for(n), kBlock>>>(g, mx_t, my_t, f_t, gamma, mx_out, my_out, f_out);
}

double eval_J(const GridDims& g, const double* mx_c, const double* my_c, const double* f_c) {
  long long n = g.size_c();
  if (n <= 0) return 0.0;
  
  thrust::device_ptr<const double> pmx(mx_c), pmy(my_c), pf(f_c);
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(pmx, pmy, pf));
  auto end = thrust::make_zip_iterator(thrust::make_tuple(pmx + n, pmy + n, pf + n));
  double raw_sum = thrust::transform_reduce(begin, end, JFunctor{}, 0.0, thrust::plus<double>());
  
  double cell_volume = 1.0 / ((double)g.N * (double)g.P * (double)g.Q);
  return raw_sum * cell_volume;
}

} // namespace ot
