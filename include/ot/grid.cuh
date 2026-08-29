#pragma once
// Grid dimensions and flat-index helpers for the dynamic optimal transport solver.
//
// Domain [0,1]^2 x [0,1] (x,y,t), N,P,Q divisions per axis.
//
//  - Centered grid G_c: (N+1) x (P+1) x (Q+1) points, x_i=i/N, y_j=j/P, t_k=k/Q.
//
//  - Staggered arrays:
//      mx : (N+2) x (P+1) x (Q+1), storage a in [0,N+1]. a=0,N+1 are the
//           x-boundary flux slots, hard-clamped to 0.
//      my : (N+1) x (P+2) x (Q+1), storage b in [0,P+1]. b=0,P+1 are the
//           y-boundary flux slots, hard-clamped to 0.
//      f  : (N+1) x (P+1) x (Q+2), storage c in [0,Q+1]. c=0 is hard-clamped
//           to f0, c=Q+1 is hard-clamped to f1.


#include <cuda_runtime.h>

namespace ot {

struct GridDims {
  int N, P, Q;

  __host__ __device__ int nx_c() const { return N + 1; }
  __host__ __device__ int ny_c() const { return P + 1; }
  __host__ __device__ int nt_c() const { return Q + 1; }

  __host__ __device__ int nx_mx() const { return N + 2; }
  __host__ __device__ int ny_mx() const { return P + 1; }
  __host__ __device__ int nt_mx() const { return Q + 1; }

  __host__ __device__ int nx_my() const { return N + 1; }
  __host__ __device__ int ny_my() const { return P + 2; }
  __host__ __device__ int nt_my() const { return Q + 1; }

  __host__ __device__ int nx_f() const { return N + 1; }
  __host__ __device__ int ny_f() const { return P + 1; }
  __host__ __device__ int nt_f() const { return Q + 2; }

  __host__ __device__ long long size_c()  const { return (long long)nx_c()  * ny_c()  * nt_c(); }
  __host__ __device__ long long size_mx() const { return (long long)nx_mx() * ny_mx() * nt_mx(); }
  __host__ __device__ long long size_my() const { return (long long)nx_my() * ny_my() * nt_my(); }
  __host__ __device__ long long size_f()  const { return (long long)nx_f()  * ny_f()  * nt_f(); }
};

__host__ __device__ inline long long idx_c(const GridDims& g, int i, int j, int k) {
  return ((long long)i * g.ny_c() + j) * g.nt_c() + k;
}
__host__ __device__ inline long long idx_mx(const GridDims& g, int a, int j, int k) {
  return ((long long)a * g.ny_mx() + j) * g.nt_mx() + k;
}
__host__ __device__ inline long long idx_my(const GridDims& g, int i, int b, int k) {
  return ((long long)i * g.ny_my() + b) * g.nt_my() + k;
}
__host__ __device__ inline long long idx_f(const GridDims& g, int i, int j, int c) {
  return ((long long)i * g.ny_f() + j) * g.nt_f() + c;
}

// Decode a flat centered-grid thread id (0..size_c()-1) back into (i,j,k)
__host__ __device__ inline void decode_c(const GridDims& g, long long tid, int& i, int& j, int& k) {
  k = (int)(tid % g.nt_c());
  long long tmp = tid / g.nt_c();
  j = (int)(tmp % g.ny_c());
  i = (int)(tmp / g.ny_c());
}

} // namespace ot
