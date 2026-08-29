#pragma once
// Host-callable launcher declarations for all device kernels.

#include "ot/grid.cuh"

namespace ot {

// ---- generic elementwise / reduction utilities on flat double buffers ----
void launch_lincomb3(double* out, double w1, const double* a, double w2, const double* b,
                    double w3, const double* c, long long n);
void launch_sub(double* out, const double* a, const double* b, long long n);      // out = a - b
void launch_add(double* out, const double* a, const double* b, long long n);      // out = a + b
void launch_add_inplace(double* a, const double* b, long long n);                 // a += b
void launch_sub_inplace(double* a, const double* b, long long n);                 // a -= b
void launch_dr_zupdate(double* z, const double* p, const double* x, const double* pi,
                        double alpha, long long n);                               // z += alpha*(2p - x - pi)
void launch_dr_xupdate(double* x, const double* p, double alpha, long long n);    // x += alpha*(p - x).
double dot(const double* a, const double* b, long long n);
double norm2(const double* a, long long n);
// alpha * || p - x ||_2.
double residual_p_minus_x(const double* p, const double* x, double alpha, long long n);
// Device-resident version of residual_p_minus_x's sum((p-x)^2)
void launch_residual_sq_to_device(const double* p, const double* x, long long n, double* residual_sq_dev);

// ---- interpolation I : E_s -> E_c and its adjoint I* : E_c -> E_s ----
void launch_interp(const GridDims& g, const double* mx_s, const double* my_s, const double* f_s,
                    double* mx_c, double* my_c, double* f_c);
void launch_interp_adjoint(const GridDims& g, const double* mx_c, const double* my_c, const double* f_c,
                            double* mx_s, double* my_s, double* f_s);

// ---- space-time divergence div : E_s -> R^{D_c} and its full adjoint div* ----
void launch_div(const GridDims& g, const double* mx_s, const double* my_s, const double* f_s,
                 double* out_c);
void launch_div_adjoint(const GridDims& g, const double* phi_c,
                         double* mx_s, double* my_s, double* f_s);

// Zero every boundary slot (mx a=0,N+1 ; my b=0,P+1 ; f c=0,Q+1).
void launch_zero_all_boundary(const GridDims& g, double* mx_s, double* my_s, double* f_s);

// Overwrite boundary slots with the prescribed data: mx,my boundary -> 0,
// f boundary -> (f0, f1) (each of size (N+1)*(P+1), row-major i*ny+j).
void launch_clamp_input_boundary(const GridDims& g, double* mx_s, double* my_s, double* f_s,
                                  const double* f0, const double* f1);

// ---- Prox_{gamma J}: per-centered-point cubic solve ----
void launch_cubic_prox(const GridDims& g, const double* mx_t, const double* my_t, const double* f_t,
                        double gamma, double* mx_out, double* my_out, double* f_out);

// J(V) = (dx*dy*dt) * sum_{grid pts} |m|^2/(2f) [f>0], 0 at (0,0)
double eval_J(const GridDims& g, const double* mx_c, const double* my_c, const double* f_c);

} // namespace ot
