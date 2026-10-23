# Third-party notices

## FlatZone-OLE comparator

`harness/*/snapshot_gpu_benchmark_globaluf_ole_v0_1.cu` contains a comparator
that implements the iterative label-equivalence algorithm of

> O. Kalentev, A. Rai, S. Kemnitz, R. Schneider. Connected component labeling on
> a 2D grid using CUDA. *Journal of Parallel and Distributed Computing*,
> 71(4):615-620, 2011.

and follows the reference implementation distributed with YACCLAB, which is
released under the BSD 3-Clause License. The adaptation extends the binary
formulation to grayscale equal-valued flat zones and to the exact dual-mask
output contract used throughout this repository, and preserves the reference
convergence protocol.

## IWPP and dual-tree comparators

`baselines/iwpp` and `baselines/blin_dual_tree` are independent task-complete
adaptations written for this study. They follow the published descriptions of
the corresponding methods and do not contain code from those authors.

## NPP comparator

`baselines/npp_exact_repair` and `baselines/blin_dual_tree` call the NVIDIA
Performance Primitives library shipped with the CUDA Toolkit, subject to the
NVIDIA CUDA Toolkit End User License Agreement. Only the surrounding adapter,
repair and classification stages are provided here.

## Datasets

No dataset is redistributed. The `prepare_*` scripts download DIV2K, BSDS500 and
BBBC038v1 from their official sources; each remains under its own license.
