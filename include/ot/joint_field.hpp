#pragma once
// A "joint field" holds one full (U,V) pair -- U = staggered (mx,my,f), V =
// centered (mx,my,f) - concatenated into a single flat buffer.

#include "ot/grid.cuh"

namespace ot {

struct JointLayout {
  GridDims g;
  long long off_mx_s, off_my_s, off_f_s;   // U block
  long long off_mx_c, off_my_c, off_f_c;   // V block
  long long total;

  static JointLayout make(GridDims g) {
    JointLayout L{};
    L.g = g;
    long long off = 0;
    L.off_mx_s = off; off += g.size_mx();
    L.off_my_s = off; off += g.size_my();
    L.off_f_s  = off; off += g.size_f();
    L.off_mx_c = off; off += g.size_c();
    L.off_my_c = off; off += g.size_c();
    L.off_f_c  = off; off += g.size_c();
    L.total = off;
    
    return L;
  }

  __host__ __device__ double* mx_s(double* base) const { return base + off_mx_s; }
  __host__ __device__ double* my_s(double* base) const { return base + off_my_s; }
  __host__ __device__ double* f_s(double* base)  const { return base + off_f_s; }
  __host__ __device__ double* mx_c(double* base) const { return base + off_mx_c; }
  __host__ __device__ double* my_c(double* base) const { return base + off_my_c; }
  __host__ __device__ double* f_c(double* base)  const { return base + off_f_c; }

  __host__ __device__ const double* mx_s(const double* base) const { return base + off_mx_s; }
  __host__ __device__ const double* my_s(const double* base) const { return base + off_my_s; }
  __host__ __device__ const double* f_s(const double* base)  const { return base + off_f_s; }
  __host__ __device__ const double* mx_c(const double* base) const { return base + off_mx_c; }
  __host__ __device__ const double* my_c(const double* base) const { return base + off_my_c; }
  __host__ __device__ const double* f_c(const double* base)  const { return base + off_f_c; }
};

} // namespace ot
