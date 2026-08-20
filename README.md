# Continuum-Mechanics

## Overview

This repository collects computational micromechanics work from a graduate continuum mechanics course at UIUC, developed alongside PhD research in computational materials science. It currently contains the course final project: an eigenstrain study of point defects in Fe-doped SrTiO$_3$, from DFT-relaxed atomic structures through elastic field solutions and energetics. All code is written from scratch in Python without FEA packages, implementing the underlying mathematical frameworks directly (Green's functions, eigenstrain theory, variational energy methods).

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
├── src/
│   └── elasticity_utils.py
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

## Notes on Data

DFT input files (CONTCAR/POSCAR format) used in notebook 01 are from ab initio calculations on Fe-doped SrTiO$_3$ and are available upon request. The extracted eigenstrains (`data/eigenstrain_results.csv`) are committed, so notebooks 02–04 are self-contained.

---

## References

1. Mura, T., *Micromechanics of Defects in Solids*, Martinus Nijhoff (1987).
2. de Jong et al., "Charting the complete elastic properties of inorganic crystalline compounds," *Scientific Data* **2**, 150009 (2015).
3. Perry et al., "Origins and control of optical absorption in a nondilute oxide solid solution: Sr(Ti,Fe)O$_{3-x}$," *Chemistry of Materials* **31**, 1030 (2019).
