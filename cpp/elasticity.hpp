// C++ port of the two numerical kernels in src/elasticity_utils.py.
//
// This header deliberately knows nothing about Python or NumPy. Every array is
// a flat, row-major (C-order) block of N*N doubles, the same layout NumPy uses
// by default, so the bindings can hand us NumPy's own memory without copying.
// Element (i, j) -- row i, column j -- lives at index i*N + j.
#pragma once

#include <cstddef>

namespace elasticity {

// Port of build_greens_function(grid_size, mu, nu).
// Writes the three N x N Green's tensor components into caller-owned buffers.
void build_greens_function(std::size_t N, double mu, double nu,
                           double* Gxx, double* Gxy, double* Gyy);

// Every output of compute_strain_and_energy, as pointers to caller-owned
// N x N buffers. The caller (bindings.cpp) allocates these as NumPy arrays,
// so the kernel writes its results straight into the arrays Python receives.
struct StrainEnergyOutputs {
    double* eps_xx;
    double* eps_yy;
    double* eps_xy;
    double* sig_xx;
    double* sig_yy;
    double* sig_xy;
    double* energy_density;
    double* ux;
    double* uy;
};

// Port of compute_strain_and_energy(exx, eyy, exy, Gxx, Gxy, Gyy, mu, nu).
// Fills every buffer in `out` and returns the total strain energy.
double compute_strain_and_energy(std::size_t N,
                                 const double* exx, const double* eyy, const double* exy,
                                 const double* Gxx, const double* Gxy, const double* Gyy,
                                 double mu, double nu,
                                 const StrainEnergyOutputs& out);

}  // namespace elasticity
