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

## UF-Gray and KE-Gray comparators

`baselines/yacclab_gray/yacclab_gray_ccl_v0_2.cu` adapts two GPU labeling
algorithms as implemented in YACCLAB (https://github.com/prittt/YACCLAB, commit
54e43543a02f): the union-find algorithm of Oliveira and Lotufo
(`labeling_oliveira_2010.cu`) and the 8-connected Komura-equivalence algorithm
of Allegretti et al. (`labeling_allegretti_2018.cu`). The kernels are modified to
label grayscale equal-valued flat zones and are completed with the
classification and mask stages used throughout this repository.

## YACCLAB license

The FlatZone-OLE, UF-Gray and KE-Gray comparators derive from YACCLAB code,
which is distributed under the following license:

```
BSD 3-Clause License

Copyright (c) 2020, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## cuCIM comparator

`baselines/cucim` calls `cucim.skimage.morphology.local_maxima` and
`local_minima` from NVIDIA cuCIM v26.10.00 (https://github.com/rapidsai/cucim),
released under the Apache License 2.0. cuCIM is not redistributed here;
`setup_cucim_env.ps1` extracts it from the official release archive.

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
