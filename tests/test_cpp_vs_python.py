"""
Correctness tests for the C++ port.

Stated tolerances
-----------------
* Green's function: bit-identical. Same arithmetic, same order, and FMA
  contraction disabled (-ffp-contract=off), so there is no round-off to allow for.
* Fields (u, strain, stress, energy density):
  max|cpp - py| <= FIELD_RTOL * max|py|, component by component. The FFTs
  and sums differ in operation order (r2c vs full complex FFT, fused loops),
  so agreement is to round-off, not bitwise. Observed: ~1e-14.
* Total energy: relative difference <= ENERGY_RTOL. Observed: <~1e-15.
* Inclusion mask: bit-identical to the original full-grid implementation.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import elasticity_utils as eu
import notebook_configs as nb

FIELD_RTOL = 1e-12
ENERGY_RTOL = 1e-12

requires_cpp = pytest.mark.skipif(eu._cpp is None, reason="C++ extension not built")


def generate_inclusion_mask_original(positions, grid_size, inclusion_radius, eigenstrain):
    """generate_inclusion_mask exactly as committed in 4d62cbe (full-grid mask)."""
    exx_val, eyy_val, exy_val = eigenstrain
    exx = np.zeros((grid_size, grid_size))
    eyy = np.zeros((grid_size, grid_size))
    exy = np.zeros((grid_size, grid_size))
    y_grid, x_grid = np.ogrid[:grid_size, :grid_size]
    for (row, col) in positions:
        mask = (x_grid - col)**2 + (y_grid - row)**2 <= inclusion_radius**2
        exx[mask] = exx_val
        eyy[mask] = eyy_val
        exy[mask] = exy_val
    return exx, eyy, exy


def cpp_compute(*args):
    assert eu._cpp is not None
    return eu.compute_strain_and_energy(*args)


py_compute = eu._compute_strain_and_energy_py

CONFIGS = nb.all_mask_configs()
CONFIG_IDS = [c[0] for c in CONFIGS]
_G_CACHE = {}


def greens_py(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = eu._build_greens_function_py(N, nb.MU, nb.NU)
    return _G_CACHE[N]


def assert_fields_close(a, b, label):
    scale = np.abs(b).max()
    err = np.abs(a - b).max()
    assert err <= FIELD_RTOL * scale, f"{label}: max|diff| = {err:.3e}, max|py| = {scale:.3e}"


# --- Inclusion mask -----------------------------------------------------------

@pytest.mark.parametrize("label,positions,N,radius,eig", CONFIGS, ids=CONFIG_IDS)
def test_mask_identical_to_original(label, positions, N, radius, eig):
    new = eu.generate_inclusion_mask(positions, N, radius, eig)
    old = generate_inclusion_mask_original(positions, N, radius, eig)
    for a, b in zip(new, old):
        np.testing.assert_array_equal(a, b)


def test_mask_identical_at_edges():
    # Discs clipped by every edge and corner, plus overlapping discs.
    positions = [(0, 0), (0, 63), (63, 0), (63, 63), (2, 30), (30, 61), (30, 30), (32, 33)]
    new = eu.generate_inclusion_mask(positions, 64, 5, (1.0, 2.0, 3.0))
    old = generate_inclusion_mask_original(positions, 64, 5, (1.0, 2.0, 3.0))
    for a, b in zip(new, old):
        np.testing.assert_array_equal(a, b)


# --- Green's function ---------------------------------------------------------

@requires_cpp
@pytest.mark.parametrize("N", [127, 128, 1024])
def test_greens_function_bit_identical(N):
    cpp = eu._cpp.build_greens_function(N, nb.MU, nb.NU)
    py = eu._build_greens_function_py(N, nb.MU, nb.NU)
    for a, b in zip(cpp, py):
        np.testing.assert_array_equal(a, b)


# --- Strain / stress / energy kernel on every notebook workload ---------------

@requires_cpp
@pytest.mark.parametrize("label,positions,N,radius,eig", CONFIGS, ids=CONFIG_IDS)
def test_kernel_matches_python(label, positions, N, radius, eig):
    fields = eu.generate_inclusion_mask(positions, N, radius, eig)
    G = greens_py(N)
    c_strain, c_stress, c_W, c_E, c_u = cpp_compute(*fields, *G, nb.MU, nb.NU)
    p_strain, p_stress, p_W, p_E, p_u = py_compute(*fields, *G, nb.MU, nb.NU)

    for k in ("xx", "yy", "xy"):
        assert_fields_close(c_strain[k], p_strain[k], f"{label} strain {k}")
        assert_fields_close(c_stress[k], p_stress[k], f"{label} stress {k}")
    assert_fields_close(c_W, p_W, f"{label} energy density")
    assert_fields_close(c_u[0], p_u[0], f"{label} ux")
    assert_fields_close(c_u[1], p_u[1], f"{label} uy")
    assert abs(c_E - p_E) <= ENERGY_RTOL * abs(p_E), f"{label}: E cpp={c_E!r} py={p_E!r}"


@requires_cpp
def test_return_types_match_python():
    fields = eu.generate_inclusion_mask([(64, 64)], 128, 12, nb.NB03_EIGENSTRAIN)
    G = greens_py(128)
    c = cpp_compute(*fields, *G, nb.MU, nb.NU)
    p = py_compute(*fields, *G, nb.MU, nb.NU)
    assert type(c[0]) is type(p[0]) is dict and c[0].keys() == p[0].keys()
    assert type(c[1]) is type(p[1]) is dict and c[1].keys() == p[1].keys()
    assert type(c[3]) is type(p[3]) is np.float64
    assert type(c[4]) is type(p[4]) is tuple and len(c[4]) == 2
    for a, b in [(c[2], p[2]), (c[4][0], p[4][0]), *[(c[0][k], p[0][k]) for k in "xx yy xy".split()]]:
        assert a.shape == b.shape and a.dtype == b.dtype


@requires_cpp
def test_non_contiguous_input_is_converted():
    # Fortran-ordered inputs go through pybind11's forcecast copy; result unchanged.
    fields = eu.generate_inclusion_mask([(64, 64)], 128, 12, nb.NB03_EIGENSTRAIN)
    G = greens_py(128)
    ref = cpp_compute(*fields, *G, nb.MU, nb.NU)
    fortran = [np.asfortranarray(f) for f in fields]
    got = cpp_compute(*fortran, *G, nb.MU, nb.NU)
    np.testing.assert_array_equal(got[2], ref[2])


@requires_cpp
def test_shape_mismatch_raises():
    G = greens_py(128)
    small = np.zeros((64, 64))
    with pytest.raises(ValueError):
        eu._cpp.compute_strain_and_energy(small, small, small, *G, nb.MU, nb.NU)


# --- Committed notebook outputs (golden values) --------------------------------
# Printed values from the committed notebook 04 outputs. Both backends must
# print exactly the same strings.

NB04_E0 = "1.549607e-02"
NB04_GRID_EINT = ["-0.0159", "-0.0704", "-0.3148", "-1.0104", "-1.8655", "-3.6936"]
NB04_DIR_EINT = {
    ("horizontal", 3): "2.9559e-03", ("horizontal", 5): "1.7912e-02", ("horizontal", 9): "8.4582e-02",
    ("diagonal1", 3): "-4.7302e-03", ("diagonal1", 5): "-8.6004e-03", ("diagonal1", 9): "-1.1117e-02",
    ("diagonal2", 3): "2.0937e-04", ("diagonal2", 5): "8.4477e-03", ("diagonal2", 9): "4.9336e-02",
}

BACKENDS = [pytest.param(py_compute, id="python"), pytest.param("cpp", id="cpp", marks=requires_cpp)]


def _resolve(compute):
    return cpp_compute if compute == "cpp" else compute


def _energy(compute, positions, N):
    fields = eu.generate_inclusion_mask(positions, N, nb.NB04_RADIUS, nb.NB04_EIGENSTRAIN)
    return compute(*fields, *greens_py(N), nb.MU, nb.NU)[3]


@pytest.mark.parametrize("compute", BACKENDS)
def test_nb04_grid_scaling_golden(compute):
    compute = _resolve(compute)
    N = nb.NB04_GRID
    E0 = _energy(compute, [(N // 2, N // 2)], N)
    assert f"{E0:.6e}" == NB04_E0
    for n, expected in zip(nb.NB04_ARRAY_SIZES, NB04_GRID_EINT):
        positions = nb.make_grid_positions(n, N, nb.NB04_RADIUS)
        E_int = _energy(compute, positions, N) - len(positions) * E0
        assert f"{E_int:.4f}" == expected, f"{n}x{n}"


@pytest.mark.parametrize("compute", BACKENDS)
def test_nb04_directional_golden(compute):
    compute = _resolve(compute)
    N = nb.NB04_GRID_DIR
    E0 = _energy(compute, [(N // 2, N // 2)], N)
    for (orient, n), expected in NB04_DIR_EINT.items():
        positions = nb.make_line_positions(n, orient, N, nb.NB04_RADIUS)
        E_int = _energy(compute, positions, N) - n * E0
        assert f"{E_int:.4e}" == expected, f"{orient} n={n}"


# --- Notebook 03: the greedy search must take the same path --------------------

@requires_cpp
def test_nb03_minimization_same_trajectory():
    # The search picks moves by comparing energies, so a round-off difference
    # at a near-tie could send it down a different path. Two full sweeps
    # (2 x 64 inclusions x 9 trials = 1152 energy evaluations per backend).
    G = greens_py(nb.NB03_GRID)
    start = nb.initialize_positions(nb.NB03_NUM, nb.NB03_GRID, nb.NB03_RADIUS)
    args = (start, nb.NB03_GRID, nb.NB03_RADIUS, nb.NB03_EIGENSTRAIN, *G, nb.MU, nb.NU, 2)
    pos_c, hist_c = nb.minimize_strain_energy(*args, compute=cpp_compute)
    pos_p, hist_p = nb.minimize_strain_energy(*args, compute=py_compute)
    assert pos_c == pos_p
    np.testing.assert_allclose(hist_c, hist_p, rtol=ENERGY_RTOL, atol=0)


# --- Backend selection / fallback ----------------------------------------------

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _backend_in_subprocess(env_value, block_extension=False):
    code = (
        "import sys\n"
        + ("sys.modules['_elasticity_cpp'] = None\n" if block_extension else "")
        + f"sys.path.insert(0, {SRC!r})\n"
        "import elasticity_utils as eu\n"
        "print(eu.BACKEND)\n"
    )
    env = dict(os.environ, ELASTICITY_BACKEND=env_value)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_forced_python_backend():
    assert _backend_in_subprocess("python").stdout.strip() == "python"


def test_fallback_when_extension_missing():
    assert _backend_in_subprocess("auto", block_extension=True).stdout.strip() == "python"


def test_forced_cpp_fails_loudly_when_missing():
    result = _backend_in_subprocess("cpp", block_extension=True)
    assert result.returncode != 0 and "_elasticity_cpp" in result.stderr
