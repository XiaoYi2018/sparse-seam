# Sparse-Seam

Reference CUDA implementation and benchmark harnesses for *Sparse-Seam: Exact
Dual Regional-Extrema Masks with Perimeter-Bounded Connectivity*.

Sparse-Seam computes both the 8-connected regional-maximum mask and the
regional-minimum mask of an 8-bit grayscale image exactly. Each `T x T` tile labels its equal-valued flat zones in shared memory by
Komura equivalence and writes the tile-contained ones immediately; only flat zones that cross a
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
harness/ablation_four_edge_local    the earlier tile-local stage (four-edge union
                                   from singleton parents) for the ablation
baselines/npp_exact_repair          NVIDIA NPP label markers plus timed exact repair
baselines/iwpp                      independent adaptation of irregular wavefront
                                   propagation
baselines/blin_dual_tree            independent adaptation of the GPU dual-tree route
baselines/yacclab_gray              grayscale UF and KE labeling (from YACCLAB) plus
                                   classification and masks
baselines/cucim                     NVIDIA cuCIM 26.10 local_maxima/local_minima
datasets/synthetic_scaling          deterministic generator for the resolution sweep
datasets/synthetic_alignment_control generator for the tile-alignment control:
                                   the same block suite with the block grid
                                   shifted by half a tile
datasets/bbbc038_distance            derivation of the BBBC038 distance maps and
                                   marker statistics from the official archive
oracles                            MATLAB oracle builders and the 206 reference
                                   cases used by the correctness gate
tools                              environment record, the script that runs
                                   every GPU measurement of the paper, and the
                                   scripts that aggregate a campaign and
                                   compare two campaigns
```

The IWPP, dual-tree and NPP-repair comparators under `baselines/` are
independent task-complete adaptations written for this study, not the original
authors' code; each is documented as such in its own header. UF-Gray and
KE-Gray start from the YACCLAB implementations of the two pixel-based algorithms
and change only the connectivity test to value equality, then share the
classification and output stages of the dense union-find reference. The cuCIM
route calls the unmodified library from Python and ORs a device-side flatness
flag into both masks, because cuCIM reports no extremum for a constant image. `baselines/npp_exact_repair` and
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
| Table 2 (configuration sweep) | `harness/configuration_study/run_config_study_v0_8.ps1` |
| Table 3, Figure 2 (DIV2K) | `harness/primary`, `baselines/` |
| Table 4 (the six routes measured on all three corpora) | `harness/primary`, `harness/quantization_and_scaling`, `baselines/cucim`, `baselines/yacclab_gray` |
| Table 5, Figure 3 (gray levels) | `harness/quantization_and_scaling`, `baselines/yacclab_gray` |
| Table 6 (BSDS500) | `harness/quantization_and_scaling`, `baselines/cucim`, `baselines/yacclab_gray` |
| Figure 4 (resolution scaling) | `datasets/synthetic_scaling`, `harness/quantization_and_scaling`, `baselines/yacclab_gray` |
| Tile-alignment control | `datasets/synthetic_alignment_control`, `harness/quantization_and_scaling`, `baselines/yacclab_gray` |
| Figure 5 (BBBC038 markers) | `datasets/bbbc038_distance/make_bbbc038_marker_figure_v0_3.py` |
| BBBC038 throughput and marker statistics | `datasets/bbbc038_distance`, `harness/quantization_and_scaling`, `baselines/cucim`, `baselines/yacclab_gray` |
| Seam-crossing fractions and fragment densities (Results text) | `harness/seam_sparsity_profiler` |
| Table 7 (tile-local labeling ablation) | `harness/ablation_four_edge_local` against `harness/quantization_and_scaling` |

## Requirements

* CUDA 12.6 (nvcc V12.6.20 was used) and a compute-capability 8.9 device
* MSVC 19.44 or later, C++17
* OpenCV 4.11 for image loading in the harnesses
* NVIDIA NPP, shipped with the CUDA toolkit, for the NPP route
* Python 3.13 for the cuCIM route; `baselines/cucim/setup_cucim_env.ps1` builds
  its environment from `requirements.txt` and the cuCIM v26.10.00 source archive

## Building

Every kernel is a single translation unit. The primary harness, as built by
`harness/primary/build_run_gpu_common_v0_6.ps1`:

```
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc /O2" ^
     -DSS_TILE=32 -DSS_CLASSIFY_VARIANT=1 ^
     -I%OPENCV_ROOT%\include -L%OPENCV_ROOT%\x64\vc16\lib -lopencv_world4110 ^
     harness/primary/gpu_common_v0_6.cu -o harness/primary/gpu_common_v0_6.exe
```

Every build script writes its executable next to the source, which is where the
corresponding run scripts look for it. `tools\run_all_campaigns.ps1` needs all of
them; from an x64 Native Tools prompt, run once:

```
harness\primary\build_run_gpu_common_v0_6.ps1
harness\quantization_and_scaling\build_gpu_common_quant_v0_7.ps1
harness\seam_sparsity_profiler\build_gpu_seam_profiler_v0_1.ps1
harness\ablation_four_edge_local\build_gpu_common_quant_v0_6.ps1
baselines\npp_exact_repair\build_npp_labelmarkersuf_exact_repair_v0_3.ps1
baselines\iwpp\build_iwpp_reconstruction_cuda_v0_1.ps1
baselines\blin_dual_tree\build_blin_optim1d_relaxed_flatten_extrema_v0_6_0.ps1
baselines\yacclab_gray\build_yacclab_gray_ccl_v0_2.ps1
datasets\synthetic_scaling\build_generate_scaling_suite_v0_1.ps1
datasets\synthetic_alignment_control\build_generate_scaling_suite_v0_2.ps1
```

each with `pwsh -File`. The configuration study builds its own nine binaries
(below). The cuCIM route needs its Python environment, built from the cuCIM
v26.10.00 release source
(https://github.com/rapidsai/cucim/archive/refs/tags/v26.10.00.tar.gz, SHA-256
`53F95163F450B6E355D11E2870038DB51D6CCBD23214B3C1CDAB9A30F8546063`, checked by
the script):

```
pwsh -File baselines\cucim\setup_cucim_env.ps1 -SourceTarball <path to v26.10.00.tar.gz>
```

The primary and quantization harnesses accept `SS_TILE` of 32 or 64 (a static
assertion rejects anything else); the source default is 64 and the reported
results use `SS_TILE=32` with classification variant 1.

The configuration study builds one binary per configuration from
`harness/configuration_study/sparse_config_benchmark_v0_8.cu`, which includes
the generated kernel `generated_sparse_seam_kernel_v0_7.cu`. From the
repository root, in PowerShell:

```
foreach ($c in "16x256", "20x256", "24x256", "28x256", "64x256",
               "32x128", "32x256", "32x512", "32x1024") {
    $t, $n = $c -split "x"
    nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc /O2" `
        "-DSS_TILE=$t" "-DSS_CLASSIFY_VARIANT=1" "-DSS_CLASSIFY_THREADS=$n" `
        "-I$env:OPENCV_ROOT\include" "-L$env:OPENCV_ROOT\x64\vc16\lib" -lopencv_world4110 `
        harness/configuration_study/sparse_config_benchmark_v0_8.cu `
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
other image count; the two expected-mask arguments are then ignored, and the
scripts pass the image directory in their place). `run_config_study_v0_8.ps1`
performs the whole study: it builds the nine binaries, admits each on the
oracle, runs nine rounds in which the execution order is cyclically shifted by
one position per round, and reports per configuration the total pixels divided
by the sum of per-image medians over the nine round medians. The generated
kernel is reproduced from the primary snapshot by `generate_sparse_config_kernel_v0_7.ps1`, which checks
the snapshot's SHA-256 before generating.

The PowerShell scripts reproduce the measurement protocol of the paper. Timed
runs use five per-image warm-ups, thirty measured repetitions and per-image
medians; the harnesses that run Global-UF, FlatZone-OLE and Sparse-Seam in one
process (`harness/primary`, `harness/quantization_and_scaling`, and the scaling
and alignment scripts under `datasets/`) are run in both execution orders, the
separate baseline processes in one. The oracle gate runs each image once (after
one warm-up in `harness/primary/build_run_gpu_common_v0_6.ps1`), and the
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

## Reproducing the measurements

After the builds and the cuCIM environment, place the public datasets as
described under Datasets and generate the two synthetic suites:

```
pwsh -File datasets\synthetic_scaling\run_generate_scaling_suite_v0_1.ps1
pwsh -File datasets\synthetic_scaling\run_generate_scaling_suite_v0_2.ps1
pwsh -File datasets\synthetic_alignment_control\run_generate_alignment_suite_v0_2.ps1
```

(the second command records the checksums of the resolution suite). Then run,
from an x64 Native Tools prompt (nvcc needs `cl.exe`):

```
pwsh -File tools\run_all_campaigns.ps1
```

It hashes every input image, then runs in order the oracle gate of all routes,
DIV2K for every route, the configuration study, the six quantization levels,
the resolution and tile-alignment suites, BSDS500, BBBC038, the seam profile
and the tile-local labeling ablation, and records driver, clocks, power draw
and the processes holding the GPU before and after each stage
(`results\campaign_logs`). Laptop GPU throughput depends on driver and power state, so compare routes within one
campaign rather than across sessions.

Every aggregate throughput, speedup, per-image win count and auxiliary-state
figure of the results section is printed by

```
python tools/aggregate_campaign.py results
```

The BBBC038 marker statistics are the column sums of the derivation manifest
(see Datasets), and Figure 5 is drawn from the derivation's audit examples by
`datasets/bbbc038_distance/make_bbbc038_marker_figure_v0_3.py` (needs
Matplotlib). Figures 2-4 plot the same aggregates: Figure 2 places the three
in-process routes, which run in both execution orders, at the midpoint of their
two orders and the six separate-process routes at their single measured value,
Figure 3 uses the pooled quantization aggregates and Figure 4 the medians of
the six trials.
The session-to-session
stability check pairs the outputs of two complete campaigns:

```
python tools/compare_campaigns.py earlier_aggregate.txt final_aggregate.txt
```

## Datasets

All formal experiments use public data. The scripts expect it under the
workspace as follows:

```
data\DIV2K_valid_HR_official\DIV2K_valid_HR      DIV2K validation, 100 PNG images,
                                                  extracted from the official
                                                  DIV2K_valid_HR.zip
data\BSDS500_official_v0.1\images_v0.2           BSDS500, 500 images, written by
                                                  harness/quantization_and_scaling/
                                                  prepare_bsds500_official_v0_2.ps1
                                                  from BSR_bsds500.tgz, which must
                                                  first be downloaded to
                                                  data\BSDS500_official_v0.1\download\
data\BBBC038_stage1_train_derived_v0.3\...        BBBC038v1 stage-1 distance maps,
                                                  written by
                                                  datasets/bbbc038_distance/
                                                  derive_bbbc038_distance_markers_v0_2.py
data\synthetic_scaling_v0.1                       resolution suite (datasets/synthetic_scaling)
data\synthetic_scaling_alignment_v0.2             tile-alignment suite
                                                  (datasets/synthetic_alignment_control)
```

`tools\run_all_campaigns.ps1` records the SHA-256 of every input image it reads.
The BSDS500 archive is published at
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz.

The BBBC038 inputs are derived on the CPU from the official `stage1_train.zip`
(SHA-256 `DCB6EDC2690F137406638B2309581A71522C4DFF19157D118453B448DCDDCB68`):

```
python datasets/bbbc038_distance/derive_bbbc038_distance_markers_v0_2.py ^
       stage1_train.zip %SPARSE_SEAM_ROOT%\data\BBBC038_stage1_train_derived_v0.3
```

For each of the 670 samples it unions the ground-truth nucleus masks, computes
the exact Euclidean distance transform, normalizes it monotonically to 8 bits,
and writes the map together with `application_manifest_v0.2.csv`, which lists
per sample the instances, the 8-connected regional-maximum markers inside the
foreground, and how many instances they hit once or more than once; the paper's
marker statistics are the column sums. The script needs NumPy and Pillow.

The synthetic scaling and tile-alignment suites are generated
deterministically from the sources in `datasets/` and are not distributed. The
only images shipped are the 206 small oracle cases under `oracles/`, each a PGM
input (with a PNG copy for the designed and fuzz cases) and its two expected
masks.

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
