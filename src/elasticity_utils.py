import numpy as np
from scipy.ndimage import sobel

def build_greens_function(grid_size, mu, nu):
    """
    Constructs 2D isotropic Green's function tensors in Fourier space.

    Solves the Navier equilibrium equations for a 2D plane-strain isotropic
    elastic medium. The DC component (k=0) is zeroed to enforce zero mean
    displacement.

    Parameters
    ----------
    grid_size : int
        Number of grid points along each axis (square grid assumed).
    mu : float
        Shear modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    Gxx, Gxy, Gyy : np.ndarray
        Green's function tensor components in Fourier space, shape (grid_size, grid_size).
    """
    kx, ky = np.meshgrid(*[np.fft.fftfreq(grid_size)] * 2, indexing="ij")
    kx *= 2 * np.pi
    ky *= 2 * np.pi
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1e-10  # avoid division by zero; zeroed below

    Gxx = (1 - nu) / mu / k_squared * (1 - ky**2 / k_squared)
    Gxy = (1 - nu) / mu / k_squared * (-kx * ky / k_squared)
    Gyy = (1 - nu) / mu / k_squared * (1 - kx**2 / k_squared)

    # Zero DC component
    Gxx[0, 0] = Gxy[0, 0] = Gyy[0, 0] = 0

    return Gxx, Gxy, Gyy


def compute_strain_and_energy(exx, eyy, exy, Gxx, Gxy, Gyy, mu, nu):
    """
    Computes elastic strain fields, stress fields, energy density, and
    total strain energy from a prescribed eigenstrain field.

    Uses Fourier-space convolution with the Green's function to obtain
    displacement fields, then recovers strain via Sobel finite differences.
    Stress follows from the isotropic constitutive law.

    Parameters
    ----------
    exx, eyy, exy : np.ndarray
        Eigenstrain field components, shape (grid_size, grid_size).
    Gxx, Gxy, Gyy : np.ndarray
        Green's function tensor components from build_greens_function().
    mu : float
        Shear modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    strain : dict
        Keys 'xx', 'yy', 'xy' — elastic strain tensor components.
    stress : dict
        Keys 'xx', 'yy', 'xy' — stress tensor components.
    energy_density : np.ndarray
        Pointwise elastic strain energy density W = 0.5 * sigma_ij * eps_ij.
    total_energy : float
        Total elastic strain energy (sum of energy_density over grid).
    displacement : tuple
        (ux, uy) real-space displacement fields.
    """
    lambda_ = 2 * mu * nu / (1 - 2 * nu)

    # Fourier transform eigenstrains
    exx_k = np.fft.fft2(exx)
    eyy_k = np.fft.fft2(eyy)
    exy_k = np.fft.fft2(exy)

    # Displacement in Fourier space: u_i(k) = G_ij(k) * eps*_jk(k)
    ux_k = Gxx * exx_k + Gxy * exy_k
    uy_k = Gyy * eyy_k + Gxy * exy_k

    # Real-space displacement
    ux = np.fft.ifft2(ux_k).real
    uy = np.fft.ifft2(uy_k).real

    # Strain from displacement gradients (Sobel finite difference)
    eps_xx = sobel(ux, axis=1) / 8.0
    eps_yy = sobel(uy, axis=0) / 8.0
    eps_xy = 0.5 * (sobel(ux, axis=0) / 8.0 + sobel(uy, axis=1) / 8.0)

    # Stress from isotropic constitutive law
    eps_kk = eps_xx + eps_yy
    sig_xx = lambda_ * eps_kk + 2 * mu * eps_xx
    sig_yy = lambda_ * eps_kk + 2 * mu * eps_yy
    sig_xy = 2 * mu * eps_xy

    # Strain energy density: W = 0.5 * sigma_ij * eps_ij
    energy_density = 0.5 * (sig_xx * eps_xx + sig_yy * eps_yy + 2 * sig_xy * eps_xy)
    total_energy = np.sum(energy_density)

    strain = {'xx': eps_xx, 'yy': eps_yy, 'xy': eps_xy}
    stress = {'xx': sig_xx, 'yy': sig_yy, 'xy': sig_xy}

    return strain, stress, energy_density, total_energy, (ux, uy)


def generate_inclusion_mask(positions, grid_size, inclusion_radius, eigenstrain):
    """
    Generates 2D eigenstrain field arrays for a set of circular inclusions.

    Each inclusion is a circular region of uniform eigenstrain. Inclusions
    are not permitted to overlap; overlapping placements are silently skipped.

    Parameters
    ----------
    positions : list of (int, int)
        List of (row, col) center coordinates for each inclusion.
    grid_size : int
        Number of grid points along each axis.
    inclusion_radius : int
        Radius of each circular inclusion in grid points.
    eigenstrain : tuple of (float, float, float)
        (exx, eyy, exy) eigenstrain components to assign inside each inclusion.

    Returns
    -------
    exx, eyy, exy : np.ndarray
        Eigenstrain field arrays, shape (grid_size, grid_size).
    """
    exx_val, eyy_val, exy_val = eigenstrain
    exx = np.zeros((grid_size, grid_size))
    eyy = np.zeros((grid_size, grid_size))
    exy = np.zeros((grid_size, grid_size))

    # Precompute circular footprint
    y_grid, x_grid = np.ogrid[:grid_size, :grid_size]

    for (row, col) in positions:
        mask = (x_grid - col)**2 + (y_grid - row)**2 <= inclusion_radius**2
        exx[mask] = exx_val
        eyy[mask] = eyy_val
        exy[mask] = exy_val

    return exx, eyy, exy