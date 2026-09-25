"""
Benchmark the C++ kernels against the original Python.

Two measurements, reported separately so neither gets credit for the other:

1. Kernel: one call of compute_strain_and_energy (and build_greens_function),
   C++ vs Python, on the notebook-03 inclusion configuration at several grid
   sizes. This is the number that measures the C++ port.
2. Notebook 03 end-to-end: the full greedy minimisation (50 sweeps x 64
   inclusions x 9 trial moves = 28,800 energy evaluations), run three ways:
     original  -- original full-grid mask + Python kernel (the committed code)
     python    -- bounding-box mask + Python kernel      (algorithmic fix only)
     cpp       -- bounding-box mask + C++ kernel          (fix + port)

Timing method: time.perf_counter; the two backends are interleaved within each
repeat so slow drift (thermal throttling, other load, P- vs E-core scheduling
on Apple silicon) affects both equally; we report the median over repeats.
Both paths are single-threaded: numpy.fft, scipy.ndimage and the C++ kernel
each use one core.

Usage:  python benchmarks/bench.py [--quick]
Writes benchmarks/results.json and benchmarks/results.md.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "tests"))

import numpy as np
import scipy

import elasticity_utils as eu
import notebook_configs as nb
from test_cpp_vs_python import generate_inclusion_mask_original

if eu._cpp is None:
    sys.exit("C++ extension not built: run `pip install -e .` first.")


def cpu_name():
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment():
    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpp_build": eu._cpp.build_info(),
    }


def per_call_seconds(fn, calls):
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    return (time.perf_counter() - t0) / calls


def bench_pair(fn_py, fn_cpp, calls, repeats):
    fn_py(); fn_cpp()  # warm-up: FFT plan cache, page faults
    t_py, t_cpp = [], []
    for _ in range(repeats):
        t_py.append(per_call_seconds(fn_py, calls))
        t_cpp.append(per_call_seconds(fn_cpp, calls))
    return statistics.median(t_py), statistics.median(t_cpp)


def kernel_benchmarks(sizes, repeats):
    rows = []
    for N in sizes:
        positions = nb.initialize_positions(nb.NB03_NUM, N, nb.NB03_RADIUS)
        fields = eu.generate_inclusion_mask(positions, N, nb.NB03_RADIUS, nb.NB03_EIGENSTRAIN)
        G = eu._build_greens_function_py(N, nb.MU, nb.NU)
        calls = max(2, int(400 * (128 / N) ** 2))  # ~0.5 s of Python work per repeat

        k_py, k_cpp = bench_pair(
            lambda: eu._compute_strain_and_energy_py(*fields, *G, nb.MU, nb.NU),
            lambda: eu.compute_strain_and_energy(*fields, *G, nb.MU, nb.NU),
            calls, repeats)
        g_py, g_cpp = bench_pair(
            lambda: eu._build_greens_function_py(N, nb.MU, nb.NU),
            lambda: eu._cpp.build_greens_function(N, nb.MU, nb.NU),
            calls, repeats)
        rows.append({"N": N, "calls_per_repeat": calls, "repeats": repeats,
                     "compute_py_ms": k_py * 1e3, "compute_cpp_ms": k_cpp * 1e3,
                     "compute_speedup": k_py / k_cpp,
                     "greens_py_ms": g_py * 1e3, "greens_cpp_ms": g_cpp * 1e3,
                     "greens_speedup": g_py / g_cpp})
        print(f"  N={N:5d}  compute: py {k_py*1e3:8.3f} ms  cpp {k_cpp*1e3:8.3f} ms  "
              f"x{k_py/k_cpp:5.2f}   |  greens: x{g_py/g_cpp:5.2f}", flush=True)
    return rows


def nb03_end_to_end(iters):
    import unittest.mock as mock
    G = eu._build_greens_function_py(nb.NB03_GRID, nb.MU, nb.NU)
    start = nb.initialize_positions(nb.NB03_NUM, nb.NB03_GRID, nb.NB03_RADIUS)
    args = (start, nb.NB03_GRID, nb.NB03_RADIUS, nb.NB03_EIGENSTRAIN, *G, nb.MU, nb.NU, iters)

    variants = {
        "original": (generate_inclusion_mask_original, eu._compute_strain_and_energy_py),
        "python": (eu.generate_inclusion_mask, eu._compute_strain_and_energy_py),
        "cpp": (eu.generate_inclusion_mask, eu.compute_strain_and_energy),
    }
    out, final = {}, {}
    for name, (mask_fn, compute_fn) in variants.items():
        with mock.patch.object(nb, "generate_inclusion_mask", mask_fn):
            t0 = time.perf_counter()
            positions, _ = nb.minimize_strain_energy(*args, compute=compute_fn)
            out[name] = time.perf_counter() - t0
        final[name] = positions
        print(f"  {name:9s} {out[name]:8.2f} s", flush=True)
    same = final["original"] == final["python"] == final["cpp"]
    return {"sweeps": iters, "energy_evaluations": iters * nb.NB03_NUM * 9,
            "seconds": out, "same_final_positions": same,
            "speedup_mask_fix_only": out["original"] / out["python"],
            "speedup_total": out["original"] / out["cpp"],
            "speedup_from_cpp_given_mask_fix": out["python"] / out["cpp"]}


def to_markdown(env, kernel, e2e):
    b = env["cpp_build"]
    lines = [
        f"CPU: {env['cpu']} ({env['logical_cores']} logical cores; all runs single-threaded)  ",
        f"OS: {env['os']}; Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}  ",
        f"C++: {b['compiler']}, {b['build_type']}, `{b['cxx_flags']}`; FFT: {b['fft']}",
        "",
        "| grid | Python (ms/call) | C++ (ms/call) | speedup |",
        "|---:|---:|---:|---:|",
    ]
    for r in kernel:
        lines.append(f"| {r['N']}² | {r['compute_py_ms']:.3f} | {r['compute_cpp_ms']:.3f} | "
                     f"{r['compute_speedup']:.1f}× |")
    if e2e:
        s = e2e["seconds"]
        lines += [
            "",
            f"Notebook 03 end-to-end ({e2e['energy_evaluations']:,} energy evaluations, 128²):",
            "",
            "| variant | time (s) | speedup vs committed code |",
            "|---|---:|---:|",
            f"| committed code (full-grid mask + Python kernel) | {s['original']:.1f} | 1.0× |",
            f"| bounding-box mask + Python kernel | {s['python']:.1f} | {e2e['speedup_mask_fix_only']:.1f}× |",
            f"| bounding-box mask + C++ kernel | {s['cpp']:.1f} | {e2e['speedup_total']:.1f}× |",
            "",
            f"Same final inclusion positions in all three: {e2e['same_final_positions']}",
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer repeats, 2 sweeps of notebook 03")
    ap.add_argument("--skip-nb03", action="store_true")
    a = ap.parse_args()

    env = environment()
    print(json.dumps(env, indent=2))
    print("Kernel (median of repeats):")
    kernel = kernel_benchmarks([128, 256, 512, 1024], repeats=3 if a.quick else 9)
    e2e = None
    if not a.skip_nb03:
        iters = 2 if a.quick else 50
        print(f"Notebook 03, {iters} sweeps:")
        e2e = nb03_end_to_end(iters)

    (HERE / "results.json").write_text(json.dumps(
        {"environment": env, "kernel": kernel, "nb03_end_to_end": e2e}, indent=2) + "\n")
    md = to_markdown(env, kernel, e2e)
    (HERE / "results.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
