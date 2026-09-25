// C++ port of build_greens_function and compute_strain_and_energy.
//
// The goal is a faithful port: every expression below mirrors a line of
// src/elasticity_utils.py, in the same order of operations, so the two paths
// agree to floating-point round-off. Nothing about the physics is changed.
#include "elasticity.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

// pocketfft is vendored in third_party/ (see VENDORED.txt). These two options
// are set in CMakeLists.txt, not here, so every file sees the same settings:
//   POCKETFFT_NO_MULTITHREADING -- single-threaded, like numpy.fft
//   POCKETFFT_CACHE_SIZE=16     -- reuse FFT plans (twiddle factors) between
//                                  calls instead of rebuilding them each time
#include "pocketfft_hdronly.h"

namespace elasticity {

namespace {  // anonymous namespace: helpers visible only inside this file

using cplx = std::complex<double>;

// Port of np.fft.fftfreq(N) * 2*pi for index i.
// NumPy computes fftfreq as integer_index * (1.0 / N), then multiplies by
// 2*pi, so we do exactly the same two roundings.
double angular_frequency(std::size_t i, std::size_t N) {
    const double val = 1.0 / static_cast<double>(N);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(N);
    std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(i);
    if (idx > (n - 1) / 2) idx -= n;  // upper half of the grid holds negative frequencies
    return (static_cast<double>(idx) * val) * (2.0 * 3.141592653589793);
}

// Port of scipy.ndimage.sobel(f, axis) / 8.0 with SciPy's default
// mode='reflect' boundary. SciPy's 'reflect' repeats the edge sample
// (d c b a | a b c d | d c b a), so for a 3-point stencil the neighbour of
// index 0 is index 0 itself: that is exactly clamping the index to [0, N-1].
//
// Like SciPy, this is two 1D passes: a central difference [-1, 0, 1] along
// `axis`, then a [1, 2, 1] smoothing along the other axis. `tmp` is scratch.
void sobel_div8(std::size_t N, const double* f, int axis, double* out, double* tmp) {
    // Row and column indices are signed (std::ptrdiff_t) because we step to
    // i - 1 and j - 1: with unsigned std::size_t, 0 - 1 wraps around to a huge
    // number instead of becoming -1.
    using idx = std::ptrdiff_t;
    const idx n = static_cast<idx>(N);
    // Value of grid g at (row r, column c), with out-of-range indices clamped
    // to the edge: SciPy's 'reflect' boundary for a 3-point stencil.
    auto at = [n](const double* g, idx r, idx c) {
        return g[std::clamp<idx>(r, 0, n - 1) * n + std::clamp<idx>(c, 0, n - 1)];
    };

    // Pass 1: central difference along `axis`.
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < n; ++j)
            tmp[i * n + j] = (axis == 1) ? at(f, i, j + 1) - at(f, i, j - 1)
                                         : at(f, i + 1, j) - at(f, i - 1, j);

    // Pass 2: [1, 2, 1] smoothing along the other axis, then / 8.
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < n; ++j) {
            const double sides = (axis == 1) ? at(tmp, i - 1, j) + at(tmp, i + 1, j)
                                             : at(tmp, i, j - 1) + at(tmp, i, j + 1);
            out[i * n + j] = (tmp[i * n + j] * 2.0 + sides) / 8.0;
        }
}

// Pairwise summation, the same scheme np.sum uses. A plain running sum loses
// accuracy like O(n * eps); pairwise keeps it near O(log n * eps), which
// matters because notebook 03 picks moves by comparing total energies.
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += a[i];
        return s;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        std::size_t i = 8;
        for (; i + 8 <= n; i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) s += a[i];
        return s;
    }
    std::size_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(a, half) + pairwise_sum(a + half, n - half);
}

}  // namespace

void build_greens_function(std::size_t N, double mu, double nu,
                           double* Gxx, double* Gxy, double* Gyy) {
    const double prefactor = (1 - nu) / mu;
    for (std::size_t i = 0; i < N; ++i) {
        const double kx = angular_frequency(i, N);  // meshgrid(..., indexing="ij"): kx varies along rows
        for (std::size_t j = 0; j < N; ++j) {
            const double ky = angular_frequency(j, N);
            double k2 = kx * kx + ky * ky;
            if (i == 0 && j == 0) k2 = 1e-10;  // avoid division by zero; zeroed below
            const double a = prefactor / k2;
            const std::size_t p = i * N + j;
            Gxx[p] = a * (1 - ky * ky / k2);
            Gxy[p] = a * (-kx * ky / k2);
            Gyy[p] = a * (1 - kx * kx / k2);
        }
    }
    Gxx[0] = Gxy[0] = Gyy[0] = 0.0;  // zero DC component
}

double compute_strain_and_energy(std::size_t N,
                                 const double* exx, const double* eyy, const double* exy,
                                 const double* Gxx, const double* Gxy, const double* Gyy,
                                 double mu, double nu,
                                 const StrainEnergyOutputs& out) {
    const double lambda = 2 * mu * nu / (1 - 2 * nu);
    const std::size_t NN = N * N;

    // --- Fourier transform eigenstrains (real-to-complex) ---
    // The inputs are real, so their spectrum is Hermitian: F(-k) = conj(F(k)).
    // r2c stores only the non-redundant half, columns 0..N/2, which is about
    // half the work and memory of NumPy's full complex fft2.
    const std::size_t H = N / 2 + 1;
    const pocketfft::shape_t shape{N, N};
    const pocketfft::shape_t axes{0, 1};
    // pocketfft strides are in bytes: how far to jump to reach the next row / column.
    const pocketfft::stride_t stride_r{static_cast<std::ptrdiff_t>(N * sizeof(double)),
                                       static_cast<std::ptrdiff_t>(sizeof(double))};
    const pocketfft::stride_t stride_c{static_cast<std::ptrdiff_t>(H * sizeof(cplx)),
                                       static_cast<std::ptrdiff_t>(sizeof(cplx))};

    // std::vector owns its memory and frees it automatically when it goes out
    // of scope (RAII), so there is no manual new/delete anywhere in this file.
    std::vector<cplx> exx_k(N * H), eyy_k(N * H), exy_k(N * H);
    pocketfft::r2c(shape, stride_r, stride_c, axes, pocketfft::FORWARD, exx, exx_k.data(), 1.0);
    pocketfft::r2c(shape, stride_r, stride_c, axes, pocketfft::FORWARD, eyy, eyy_k.data(), 1.0);
    pocketfft::r2c(shape, stride_r, stride_c, axes, pocketfft::FORWARD, exy, exy_k.data(), 1.0);

    // --- Displacement in Fourier space: u_i(k) = G_ij(k) * eps*_jk(k) ---
    // Half-spectrum column j has the same wavevector as full-grid column j,
    // so we read G from the full N x N arrays at (i, j) for j <= N/2.
    //
    // Why G is symmetrised: the Python takes np.fft.ifft2(...).real, which is
    // mathematically the same as using the even part (G(k) + G(-k)) / 2. For
    // Gxx and Gyy that is exactly G. Gxy ~ -kx*ky is not even on the Nyquist
    // row/column of an even grid (index N/2 is its own mirror), so there .real
    // discards a non-Hermitian component. c2r below assumes a Hermitian
    // spectrum, so we make that symmetrisation explicit to match the Python.
    //
    // ux_k and uy_k overwrite exx_k and eyy_k in place, since those are no
    // longer needed afterwards.
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t i_mirror = (N - i) % N;  // index of -kx
        for (std::size_t j = 0; j < H; ++j) {
            const std::size_t g = i * N + j;
            const std::size_t g_mirror = i_mirror * N + (N - j) % N;  // index of -k
            const std::size_t h = i * H + j;
            const double gxx = 0.5 * (Gxx[g] + Gxx[g_mirror]);
            const double gxy = 0.5 * (Gxy[g] + Gxy[g_mirror]);
            const double gyy = 0.5 * (Gyy[g] + Gyy[g_mirror]);
            const cplx ux_k = gxx * exx_k[h] + gxy * exy_k[h];
            const cplx uy_k = gyy * eyy_k[h] + gxy * exy_k[h];
            exx_k[h] = ux_k;
            eyy_k[h] = uy_k;
        }
    }

    // --- Real-space displacement ---
    // Complex-to-real inverse; it returns the real part directly, which is what
    // np.fft.ifft2(...).real does. The 1/N^2 factor is ifft2's normalisation.
    const double inv_nn = 1.0 / static_cast<double>(NN);
    pocketfft::c2r(shape, stride_c, stride_r, axes, pocketfft::BACKWARD, exx_k.data(), out.ux, inv_nn);
    pocketfft::c2r(shape, stride_c, stride_r, axes, pocketfft::BACKWARD, eyy_k.data(), out.uy, inv_nn);

    // --- Strain from displacement gradients (Sobel finite difference) ---
    std::vector<double> tmp(NN), dux_d0(NN), duy_d1(NN);
    sobel_div8(N, out.ux, 1, out.eps_xx, tmp.data());   // sobel(ux, axis=1) / 8
    sobel_div8(N, out.uy, 0, out.eps_yy, tmp.data());   // sobel(uy, axis=0) / 8
    sobel_div8(N, out.ux, 0, dux_d0.data(), tmp.data());
    sobel_div8(N, out.uy, 1, duy_d1.data(), tmp.data());

    // --- Shear strain, stress (isotropic law) and energy density, in one pass ---
    // NumPy evaluates each of these lines as a separate full-grid array
    // operation with its own temporary; here each point is done once.
    for (std::size_t p = 0; p < NN; ++p) {
        const double e_xx = out.eps_xx[p];
        const double e_yy = out.eps_yy[p];
        const double e_xy = 0.5 * (dux_d0[p] + duy_d1[p]);
        const double e_kk = e_xx + e_yy;
        const double s_xx = lambda * e_kk + 2 * mu * e_xx;
        const double s_yy = lambda * e_kk + 2 * mu * e_yy;
        const double s_xy = 2 * mu * e_xy;
        out.eps_xy[p] = e_xy;
        out.sig_xx[p] = s_xx;
        out.sig_yy[p] = s_yy;
        out.sig_xy[p] = s_xy;
        // W = 0.5 * sigma_ij * eps_ij
        out.energy_density[p] = 0.5 * (s_xx * e_xx + s_yy * e_yy + 2 * s_xy * e_xy);
    }

    return pairwise_sum(out.energy_density, NN);
}

}  // namespace elasticity
