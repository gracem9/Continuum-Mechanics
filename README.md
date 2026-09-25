# Continuum-Mechanics

## Overview

This repository collects computational micromechanics work from a graduate continuum mechanics course at UIUC, developed alongside PhD research in computational materials science. It currently contains the course final project: an eigenstrain study of point defects in Fe-doped SrTiO$_3$, from DFT-relaxed atomic structures through elastic field solutions and energetics. All code is written from scratch in Python without FEA packages (the solver's two numerical kernels also have a C++ port; see [C++ kernel](#c-kernel)), implementing the underlying mathematical frameworks directly (Green's functions, eigenstrain theory, variational energy methods).

These works all predict how heterogeneity at the micro-scale (defects, inclusions, grains) drives mechanical response at the macro-scale.

Coming soon — Eshelby inhomogeneity fields, self-consistent homogenization, and viscoplastic polycrystal modeling

---

## Physical & Mathematical Background

### Eigenstrain Theory

When a defect or inclusion undergoes a stress-free transformation strain, it introduces an *eigenstrain* $\varepsilon^*_{ij}$ that drives elastic relaxation in the surrounding matrix. Here, eigenstrains are extracted directly from DFT-relaxed atomic structures:

$$\varepsilon^*_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$$

where $u_i(\mathbf{r})$ are displacement vectors from subtracting reference from defective atomic positions.

### Green's Function Elasticity

The elastic response is solved in Fourier space, where convolution with the eigenstrain reduces to pointwise multiplication. Displacement in Fourier space:

$$\tilde{u}_i(\mathbf{k}) = G_{ij}(\mathbf{k})\,\tilde{\varepsilon}^*_{jk}(\mathbf{k})$$

with Green's tensor components from the isotropic Navier equations:

$$G_{xx} = \frac{1-\nu}{\mu k^2}\left(1 - \frac{k_y^2}{k^2}\right), \quad G_{yy} = \frac{1-\nu}{\mu k^2}\left(1 - \frac{k_x^2}{k^2}\right), \quad G_{xy} = -\frac{1-\nu}{\mu k^2}\frac{k_x k_y}{k^2}$$

Stresses follow from the isotropic constitutive law $\sigma_{ij} = \lambda\delta_{ij}\varepsilon_{kk} + 2\mu\varepsilon_{ij}$, and strain energy density is computed pointwise as $W = \frac{1}{2}\sigma_{ij}\varepsilon_{ij}$.

---

## Repository Structure

```
continuum-mechanics/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eigenstrain_extraction.ipynb
│   ├── 02_greens_function_solver.ipynb
│   ├── 03_strain_energy_minimization.ipynb
│   └── 04_interaction_energy_scaling.ipynb
├── pyproject.toml          # pip → scikit-build-core → CMake
├── CMakeLists.txt
├── src/
│   └── elasticity_utils.py   # public API; uses C++ if built, else Python
├── cpp/
│   ├── elasticity.hpp/.cpp   # pure C++ kernels (no Python dependency)
│   ├── bindings.cpp          # pybind11 / NumPy boundary
│   └── build_info.hpp.in
├── third_party/pocketfft/    # vendored FFT header (BSD-3)
├── tests/                    # C++ vs Python on notebook workloads
├── benchmarks/               # bench.py, results.md, results.json
├── figures/
│   ├── strain_field_decomposition.png
│   ├── energy_minimization_convergence.png
│   ├── interaction_energy_grid_scaling.png
│   └── interaction_energy_directional.png
└── data/
    ├── README.md
    └── eigenstrain_results.csv
```

---

## Notebooks

| # | Notebook | Topic | Key Methods |
|---|---|---|---|
| 01 | `eigenstrain_extraction` | Extract eigenstrains from DFT structures | Linear regression on displacement gradients, pymatgen POSCAR parsing |
| 02 | `greens_function_solver` | 2D Fourier-space elasticity solver | FFT, Navier Green's functions, Sobel differentiation |
| 03 | `strain_energy_minimization` | Optimal inclusion placement via energy minimization | Iterative stochastic local search, strain energy landscape |
| 04 | `interaction_energy_scaling` | Inclusion-inclusion elastic interaction vs. geometry | Interaction energy, array scaling, directional anisotropy |

---

## Key Results

Five defect configurations in Fe-doped SrTiO₃ exhibit dipole-like strain fields. The 2FeTi + V_O cluster shows reduced total strain relative to either constituent, suggesting the vacancy compensates mechanical as well as electrostatic strain. Iterative energy minimization drives self-organization into staggered anti-aligned configurations purely from elastic cooperativity — no chemical or lattice constraints imposed. Interaction energy scales strongly with inclusion geometry: diagonal arrangements yield attractive interactions while horizontal arrangements are repulsive.

---

## Dependencies

```
numpy
scipy
matplotlib
pandas
scikit-learn
pymatgen
seaborn
```

Install with: `pip install -r requirements.txt`

---

## C++ kernel

`build_greens_function` and `compute_strain_and_energy` have a C++17 implementation (`cpp/`), exposed to Python with pybind11 and built with CMake. `src/elasticity_utils.py` keeps the same function signatures and return types. It calls the C++ version when the extension is installed and otherwise falls back to the original Python, which is still in the file as `_build_greens_function_py` and `_compute_strain_and_energy_py`. The physics, equations and results are unchanged: this is a port, not a rewrite.

**What moved, and why.** Profiling the notebooks showed that `compute_strain_and_energy` is the kernel: notebook 03 calls it 28,800 times. `build_greens_function` runs once per notebook. It was ported so the k-grid convention lives next to the code that consumes it, not for speed. Profiling also found that `generate_inclusion_mask` cost more than the kernel, because it tested every inclusion against the whole grid. It stays in Python, but now tests only each inclusion's bounding box. Its output is bit-identical to the original, which the tests check.

**Design.**
- Arrays cross the boundary as `py::array_t<double, c_style | forcecast>`. A C-contiguous float64 NumPy array is used in place, with no copy. The kernel writes into NumPy arrays allocated by the bindings.
- The FFT is [pocketfft](https://github.com/mreineck/pocketfft), the library behind `numpy.fft`, vendored as one BSD-licensed header. The inputs are real, so the kernel uses real-to-complex transforms, which do half the work of NumPy's complex `fft2`.
- The Sobel differences, stress and energy are computed in single passes over the grid, instead of one full-grid NumPy array per expression.
- It is built with `-O3 -ffp-contract=off` and without `-ffast-math`, so every floating-point operation rounds as NumPy's does.
- One subtlety: the Python takes `ifft2(...).real`, and on an even grid `Gxy` is not symmetric under k → −k on the Nyquist row and column. `.real` silently drops that part. The C++ symmetrises G explicitly to reproduce the same result.

**Build.** You need a C++17 compiler (Xcode Command Line Tools on macOS). pip fetches CMake, pybind11 and scikit-build-core automatically.

```bash
pip install -e .                      # builds cpp/ with CMake and installs _elasticity_cpp
python -m pytest tests                # C++ vs Python agreement
python benchmarks/bench.py            # writes benchmarks/results.md
ELASTICITY_BACKEND=python jupyter lab # force the original Python path
```

To use CMake directly: `cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir) && cmake --build build`.

**Correctness.** The tests run both paths on every configuration the notebooks use: the five DFT-derived defects from notebook 02, notebook 03's seed-42 starting configuration, and notebook 04's 1024² arrays and directional lines.
- The Green's tensor is bit-identical.
- Fields and total energy agree to within 1e-12 of the field's maximum. The observed difference is about 1e-14.
- Both backends reproduce every value printed in notebook 04's committed output.
- The full 50-sweep notebook 03 minimisation reaches the same final configuration with either backend.
- Notebooks 02–04 run unchanged with either backend. Three of the four figures are pixel-identical between backends. In `strain_field_decomposition.png`, 0.1% of pixels differ by one colour level, all at points where the plotted strain is zero to round-off (|value| < 1e-16), which sits exactly on the colormap's centre.

**Measured speedup.** One call of `compute_strain_and_energy`, median of 9 repeats, both paths single-threaded. Apple M1 Pro, macOS 14.5, AppleClang 16, `-O3 -ffp-contract=off`, Python 3.12, NumPy 2.3.5, SciPy 1.18.1:

| grid | Python (ms) | C++ (ms) | speedup |
|---:|---:|---:|---:|
| 128² | 1.35 | 0.52 | 2.6× |
| 256² | 7.36 | 2.48 | 3.0× |
| 512² | 39.3 | 12.9 | 3.0× |
| 1024² | 192 | 56.2 | 3.4× |

Notebook 03 end to end (28,800 evaluations): 151.5 s for the committed code, 73.5 s with the mask fix alone, and 47.6 s with the mask fix plus the C++ kernel, 3.2× overall. NumPy already runs compiled FFTs, so the gain comes from halving the FFT work and removing temporary arrays and Python overhead. About two thirds of the C++ time is still FFT, which is why the speedup levels off near 3×. Full output is in `benchmarks/results.md`. No threading is used.

---

## Notes on Data

DFT input files (CONTCAR/POSCAR format) used in notebook 01 are from ab initio calculations on Fe-doped SrTiO$_3$ and are available upon request. The extracted eigenstrains (`data/eigenstrain_results.csv`) are committed, so notebooks 02–04 are self-contained.

---

## References

1. Mura, T., *Micromechanics of Defects in Solids*, Martinus Nijhoff (1987).
2. de Jong et al., "Charting the complete elastic properties of inorganic crystalline compounds," *Scientific Data* **2**, 150009 (2015).
3. Perry et al., "Origins and control of optical absorption in a nondilute oxide solid solution: Sr(Ti,Fe)O$_{3-x}$," *Chemistry of Materials* **31**, 1030 (2019).
