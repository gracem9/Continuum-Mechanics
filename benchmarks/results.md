CPU: Apple M1 Pro (10 logical cores; all runs single-threaded)  
OS: Darwin 23.5.0 (arm64); Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1  
C++: AppleClang 16.0.0.16000026, Release, `-O3 -DNDEBUG -std=c++17 -ffp-contract=off`; FFT: pocketfft (vendored, commit c90e55b), single-threaded, real-to-complex

| grid | Python (ms/call) | C++ (ms/call) | speedup |
|---:|---:|---:|---:|
| 128² | 1.348 | 0.519 | 2.6× |
| 256² | 7.359 | 2.482 | 3.0× |
| 512² | 39.336 | 12.926 | 3.0× |
| 1024² | 192.472 | 56.201 | 3.4× |

Notebook 03 end-to-end (28,800 energy evaluations, 128²):

| variant | time (s) | speedup vs committed code |
|---|---:|---:|
| committed code (full-grid mask + Python kernel) | 151.5 | 1.0× |
| bounding-box mask + Python kernel | 73.5 | 2.1× |
| bounding-box mask + C++ kernel | 47.6 | 3.2× |

Same final inclusion positions in all three: True
