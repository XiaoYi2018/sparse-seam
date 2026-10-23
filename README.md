# Sparse-Seam

Reference CUDA implementation and benchmark harnesses for *Sparse-Seam: Exact
Dual Regional-Extrema Masks with Perimeter-Bounded Connectivity*.

Sparse-Seam computes both the 8-connected regional-maximum mask and the
regional-minimum mask of an 8-bit grayscale image, exactly, in one pass over the
image. Each `T x T` tile resolves its equal-valued flat zones in shared memory
and writes the tile-contained ones immediately; only flat zones that cross a
tile seam enter a global union--find, and their decisions return to the interior
through a one-byte per-pixel record. No full-image label array and no
component-tree hierarchy is built.

Every comparator in this repository returns the same two exact masks, so
throughput and memory are compared under one output contract rather than across
different ones.

## Contents

```
harness/primary                    Global-UF, FlatZone-OLE and Sparse-Seam in one
                                   process, one dataset traversal, identical
                                   warm-up and repetition settings
harness/quantization_and_scaling    the same three routes over quantized,
                                   synthetic-scaling, BSDS500 and BBBC038 inputs
harness/configuration_study         tile-size and thread-count sweep
harness/seam_sparsity_profiler      seam-crossing pixel fraction and distinct
                                   fragment density
baselines/npp_exact_repair          NVIDIA NPP label markers plus timed exact repair
baselines/iwpp                      independent adaptation of irregular wavefront
                                   propagation
baselines/blin_dual_tree            independent adaptation of the GPU dual-tree route
datasets/synthetic_scaling          deterministic generator for the resolution sweep
datasets/synthetic_alignment_control generator for the tile-alignment control:
                                   the same block suite with the block grid
                                   shifted by half a tile
oracles                            MATLAB oracle builders and the 206 reference
                                   cases used by the correctness gate
```

The three comparators under `baselines/` are independent task-complete
adaptations written for this study, not the original authors' code. Each is
documented as such in its own header. `baselines/npp_exact_repair` and
`baselines/blin_dual_tree` both include the same NPP label-marker adapter,
`npp_labelmarkersuf_extrema_v0_1.cu`; the two copies are byte-identical so that
each directory builds on its own. FlatZone-OLE follows the pinned reference
implementation of the label-equivalence route, including its convergence
protocol, in which the iteration flag is written and read back through the host
once per pass.

## Correspondence to the paper

| Paper | Produced by |
| --- | --- |
| Table 1 (routes compared) | `harness/primary`, `baselines/` |
| Table 2 (configuration sweep) | `harness/configuration_study` |
| Table 3 (DIV2K) | `harness/primary` |
| Table 4, Figure 3 (gray levels) | `harness/quantization_and_scaling` |
| Table 5 (BSDS500) | `harness/quantization_and_scaling` |
| Figure 4 (resolution scaling) | `datasets/synthetic_scaling` |
| Tile-alignment control | `datasets/synthetic_alignment_control` |
| Figure 5 (BBBC038 markers) | `harness/quantization_and_scaling` |
| Seam-crossing fractions and fragment densities (Results text) | `harness/seam_sparsity_profiler` |

## Requirements

* CUDA 12.6 (nvcc V12.6.20 was used) and a compute-capability 8.9 device
* MSVC 19.44 or later, C++17
* OpenCV 4.11 for image loading in the harnesses
* NVIDIA NPP, shipped with the CUDA toolkit, for the NPP route

## Building

Every kernel is a single translation unit. The primary harness, as built by
`harness/primary/build_run_gpu_common_v0_5.ps1`:

```
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc /O2" ^
     -DSS_TILE=32 -DSS_CLASSIFY_VARIANT=1 ^
     -I%OPENCV_ROOT%\include -L%OPENCV_ROOT%\x64\vc16\lib -lopencv_world4110 ^
     harness/primary/gpu_common_v0_5.cu -o harness/primary/gpu_common_v0_5.exe
```

Every build script writes its executable next to the source, which is where the
corresponding run scripts look for it.

The primary and quantization harnesses accept `SS_TILE` of 32 or 64 (a static
assertion rejects anything else); the source default is 64 and the reported
results use `SS_TILE=32` with classification variant 1.

The configuration study builds one binary per configuration from
`harness/configuration_study/sparse_config_benchmark_v0_7.cu`, which includes
the generated kernel `generated_sparse_seam_kernel_v0_6.cu`. From the
repository root, in PowerShell:

```
foreach ($c in "16x256", "20x256", "24x256", "28x256", "64x256",
               "32x128", "32x256", "32x512", "32x1024") {
    $t, $n = $c -split "x"
    nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc /O2" `
        "-DSS_TILE=$t" "-DSS_CLASSIFY_VARIANT=1" "-DSS_CLASSIFY_THREADS=$n" `
        "-I$env:OPENCV_ROOT\include" "-L$env:OPENCV_ROOT\x64\vc16\lib" -lopencv_world4110 `
        harness/configuration_study/sparse_config_benchmark_v0_7.cu `
        -o "harness/configuration_study/sparse_config_tile${t}_threads$n.exe"
}
```

Static assertions admit only the nine preregistered configurations:
`SS_CLASSIFY_THREADS=256` with `SS_TILE` in `{16, 20, 24, 28, 32, 64}`, and
`SS_TILE=32` with `SS_CLASSIFY_THREADS` in `{128, 512, 1024}`. Each binary must
be called with `--config=tile<T>_threads<N>` matching its build and with
`CUDA_MODULE_LOADING=EAGER` set in the environment; oracle admission uses
`--round=0`, timing runs use `--skip-expected --round=1..9` with five warm-ups
and ten repetitions on the 100 DIV2K validation images (the binary refuses any
other image count; pass empty strings for the two expected-mask directories). The generated kernel is reproduced from
the primary snapshot by `generate_sparse_config_kernel_v0_6.ps1`, which checks
the snapshot's SHA-256 before generating.

The PowerShell scripts reproduce the measurement protocol of the paper. Timed
runs use five per-image warm-ups, thirty measured repetitions and per-image
medians; the three in-process routes (`harness/primary`,
`harness/quantization_and_scaling`, `datasets/synthetic_scaling`) are run in
both execution orders, the separate baseline processes in one. The
`*_correctness.ps1` scripts run each image once against the oracle, and the
configuration study uses the fixed five warm-ups and ten repetitions per round
noted above. The scripts read three environment variables:

```
SPARSE_SEAM_ROOT   workspace for data\ and results\ (defaults to the repository root)
OPENCV_ROOT        OpenCV build directory
CUDA_PATH          CUDA toolkit directory (usually already set by the installer)
```

Sources, executables and oracles are taken from the repository itself; the
workspace only receives downloaded data under `data\` and measurements under
`results\`. Both directories are ignored by git.

## Datasets

All formal experiments use public data, downloaded and verified by the
`prepare_*` scripts:

* DIV2K validation, 100 high-resolution images
* BSDS500, 500 images
* BBBC038v1 stage-1 training samples, from the Broad Bioimage Benchmark
  Collection

The synthetic scaling and tile-alignment suites are generated
deterministically from the sources in `datasets/` and are not distributed. The
only images shipped are the 206 small oracle cases under `oracles/`, each an
input image with its two expected masks.

## Correctness

Admission to the timed comparison requires zero pixel mismatches against a
morphological oracle on all 206 hand-designed, randomized and seam-adversarial
images. The reference masks were produced with MATLAB `imregionalmax` and
`imregionalmin` at 8-connectivity and are shipped in `oracles/` (34 designed
cases, 160 fuzz cases, 12 seam-adversarial cases); the MATLAB builders next to
them regenerate the same cases and masks with seed 20260728. The `*_correctness.ps1`
scripts run that gate. Every route reported in the paper passes it.

## License

BSD 3-Clause. See `LICENSE`.
