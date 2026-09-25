"""
The inputs the notebooks actually use, so tests and benchmarks run on real
workloads rather than synthetic arrays. Helper functions are copied verbatim
from the notebooks (only `compute` was added as a parameter to
minimize_strain_energy, so it can be run against either backend).
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from elasticity_utils import generate_inclusion_mask

REPO = Path(__file__).resolve().parents[1]
MU, NU = 1.0, 0.25

# --- Notebook 02: five DFT-derived defects, one inclusion each, 128^2, r=12 ---
NB02_GRID, NB02_RADIUS = 128, 12
NB02_DEFECTS = ["Fe_Ti_V_O_0_3_+U", "2Fe_Ti_V_O_0_3_+U", "Fe_Ti_0_3_+U", "VO_0_3_+U", "2Fe_Ti_0_3_+U"]


def nb02_eigenstrains():
    df = pd.read_csv(REPO / "data" / "eigenstrain_results.csv")
    out = []
    for defect in NB02_DEFECTS:
        row = df[df["defect"] == defect].iloc[0]
        out.append((defect, (row["strain_xx"], row["strain_yy"], row["strain_xy"])))
    return out


# --- Notebook 03: 64 FeTi + V_O inclusions, 128^2, r=4, seed 42 ---
NB03_GRID, NB03_RADIUS, NB03_NUM = 128, 4, 64
NB03_EIGENSTRAIN = (0.006373, 0.006373, 7.435702e-04)


def initialize_positions(num_inclusions, grid_size, inclusion_radius, seed=42):
    """Generate random non-overlapping inclusion center positions."""
    random.seed(seed)
    positions = []
    min_dist = 2 * inclusion_radius + 1
    max_attempts = 10000

    while len(positions) < num_inclusions:
        for _ in range(max_attempts):
            r = random.randint(inclusion_radius, grid_size - inclusion_radius - 1)
            c = random.randint(inclusion_radius, grid_size - inclusion_radius - 1)
            if all(abs(r - pr) + abs(c - pc) >= min_dist for pr, pc in positions):
                positions.append((r, c))
                break
        else:
            raise RuntimeError("Could not place all inclusions without overlap.")
    return positions


def minimize_strain_energy(positions, grid_size, inclusion_radius, eigenstrain,
                           Gxx, Gxy, Gyy, mu, nu, max_iters, compute):
    """Notebook 03's greedy local search, with the energy function passed in."""
    positions = [list(p) for p in positions]
    energy_history = []
    for iteration in range(max_iters):
        for idx in range(len(positions)):
            original = positions[idx].copy()
            best_energy = None
            best_pos = original.copy()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = original[0] + dr, original[1] + dc
                    if not (inclusion_radius <= nr < grid_size - inclusion_radius and
                            inclusion_radius <= nc < grid_size - inclusion_radius):
                        continue
                    trial = [p.copy() for p in positions]
                    trial[idx] = [nr, nc]
                    exx, eyy, exy = generate_inclusion_mask(trial, grid_size, inclusion_radius, eigenstrain)
                    _, _, _, energy, _ = compute(exx, eyy, exy, Gxx, Gxy, Gyy, mu, nu)
                    if best_energy is None or energy < best_energy:
                        best_energy = energy
                        best_pos = [nr, nc]
            positions[idx] = best_pos
        energy_history.append(best_energy)
    return positions, energy_history


# --- Notebook 04: 2FeTi + V_O, r=4; NxN arrays on 1024^2, lines on 128^2 ---
NB04_GRID, NB04_GRID_DIR, NB04_RADIUS = 1024, 128, 4
NB04_EIGENSTRAIN = (0.005691, 0.005691, -2.92e-07)
NB04_ARRAY_SIZES = [3, 5, 9, 15, 20, 25]
NB04_COUNTS = [3, 5, 9]
NB04_ORIENTATIONS = ["horizontal", "diagonal1", "diagonal2"]


def make_grid_positions(n, grid_size, inclusion_radius):
    """Place N×N inclusions in a uniform grid centered on the domain."""
    spacing = (grid_size - 2 * inclusion_radius) // (n + 1)
    start = inclusion_radius + spacing
    positions = []
    for i in range(n):
        for j in range(n):
            r = start + i * spacing
            c = start + j * spacing
            positions.append((r, c))
    return positions


def make_line_positions(n, orientation, grid_size, inclusion_radius):
    """Place n inclusions in a line across the grid center."""
    center = grid_size // 2
    spacing = (grid_size - 4 * inclusion_radius) // (n + 1)
    offsets = np.arange(n) - (n - 1) / 2
    offsets = (offsets * spacing).astype(int)

    positions = []
    for d in offsets:
        if orientation == 'horizontal':
            positions.append((center, center + d))
        elif orientation == 'diagonal1':
            positions.append((center + d, center + d))
        elif orientation == 'diagonal2':
            positions.append((center - d, center + d))
    return positions


def all_mask_configs():
    """(label, positions, grid_size, radius, eigenstrain) for every notebook workload."""
    configs = []
    for defect, eig in nb02_eigenstrains():
        configs.append((f"nb02-{defect}", [(NB02_GRID // 2, NB02_GRID // 2)], NB02_GRID, NB02_RADIUS, eig))
    configs.append(("nb03-initial",
                    initialize_positions(NB03_NUM, NB03_GRID, NB03_RADIUS),
                    NB03_GRID, NB03_RADIUS, NB03_EIGENSTRAIN))
    for n in NB04_ARRAY_SIZES:
        configs.append((f"nb04-grid-{n}x{n}", make_grid_positions(n, NB04_GRID, NB04_RADIUS),
                        NB04_GRID, NB04_RADIUS, NB04_EIGENSTRAIN))
    for orient in NB04_ORIENTATIONS:
        for n in NB04_COUNTS:
            configs.append((f"nb04-{orient}-{n}", make_line_positions(n, orient, NB04_GRID_DIR, NB04_RADIUS),
                            NB04_GRID_DIR, NB04_RADIUS, NB04_EIGENSTRAIN))
    return configs
