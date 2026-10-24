"""Task-complete regional-extrema benchmark for cuCIM local_maxima/local_minima, v0.1

The route returns both full 8-connected regional-extrema masks of an 8-bit image
with NVIDIA cuCIM (cucim.skimage.morphology.local_maxima and local_minima,
connectivity=2, allow_borders=True) running on CuPy.

cuCIM requires a plateau to have at least one strictly lower (higher) neighbor,
so a constant image has no extrema under its definition, whereas the task
definition (MATLAB imregionalmax/imregionalmin) makes a constant image one
maximum and one minimum. The route therefore ORs a device-side flatness flag
into both masks; the flag is computed on the GPU inside the timed region and
causes no extra host synchronization.

Timing: the input is resident on the device before timing and both masks stay on
the device; CUDA events bracket the two library calls and the flatness fix, so
the library's own host synchronizations and Python call overhead are included.
Device memory: a CuPy memory hook records the peak of live pool allocations
during one untimed call; the auxiliary figure excludes the two output masks
(the input is allocated before the hook and is not counted).
"""

import argparse
import csv
import sys
from pathlib import Path

import cupy as cp
import cv2
import numpy as np

from cucim.skimage.morphology import local_maxima, local_minima

EXTENSIONS = {".png", ".pgm", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def route(image):
    maxima = local_maxima(image, connectivity=2, allow_borders=True)
    minima = local_minima(image, connectivity=2, allow_borders=True)
    flat = image.max() == image.min()
    maxima |= flat
    minima |= flat
    return maxima, minima


class PeakHook(cp.cuda.MemoryHook):
    name = "PeakHook"

    def __init__(self):
        self.live = 0
        self.peak = 0

    def malloc_postprocess(self, **kwargs):
        self.live += kwargs["mem_size"]
        self.peak = max(self.peak, self.live)

    def free_postprocess(self, **kwargs):
        self.live -= kwargs["mem_size"]


def percentile(values, probability):
    return float(np.percentile(np.asarray(values, dtype=np.float64), probability * 100.0))


def list_images(directory):
    files = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images found in {directory}")
    return files


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-max")
    parser.add_argument("--expected-min")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    cp.cuda.Device(args.device).use()
    check = args.expected_max is not None and args.expected_min is not None

    files = list_images(args.image_dir)
    total_mismatches = 0
    total_pixels = 0
    total_median_s = 0.0
    total_aux = 0
    with open(output / "cucim_extrema_v0_1.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "algorithm", "image", "width", "height", "megapixels", "warmup",
            "repetitions", "peak_live_bytes", "auxiliary_bytes", "auxiliary_Bpp",
            "median_compute_ms", "p95_compute_ms", "compute_mpix_s",
            "max_mismatches", "min_mismatches"])
        for path in files:
            host = read_gray(path)
            height, width = host.shape
            pixels = width * height
            image = cp.asarray(host)
            cp.cuda.Stream.null.synchronize()

            hook = PeakHook()
            with hook:
                maxima, minima = route(image)
                cp.cuda.Stream.null.synchronize()
            auxiliary = hook.peak - maxima.nbytes - minima.nbytes
            del maxima, minima

            for _ in range(args.warmup):
                route(image)
            cp.cuda.Stream.null.synchronize()
            start, finish = cp.cuda.Event(), cp.cuda.Event()
            elapsed = []
            for _ in range(args.reps):
                start.record()
                maxima, minima = route(image)
                finish.record()
                finish.synchronize()
                elapsed.append(cp.cuda.get_elapsed_time(start, finish))

            max_mismatch = min_mismatch = 0
            if check:
                name = path.stem + ".pgm"
                expected_max = read_gray(Path(args.expected_max) / name) != 0
                expected_min = read_gray(Path(args.expected_min) / name) != 0
                max_mismatch = int(np.count_nonzero(maxima.get() != expected_max))
                min_mismatch = int(np.count_nonzero(minima.get() != expected_min))
            total_mismatches += max_mismatch + min_mismatch

            median_ms = percentile(elapsed, 0.5)
            megapixels = pixels / 1e6
            total_pixels += pixels
            total_median_s += median_ms / 1000.0
            total_aux += auxiliary
            writer.writerow([
                "cuCIM-26.10-local-extrema", path.name, width, height,
                f"{megapixels:.6f}", args.warmup, args.reps, hook.peak, auxiliary,
                f"{auxiliary / pixels:.6f}", f"{median_ms:.6f}",
                f"{percentile(elapsed, 0.95):.6f}",
                f"{megapixels / (median_ms / 1000.0):.6f}",
                max_mismatch, min_mismatch])
            handle.flush()
            print(f"[CUCIM-EXTREMA-v0.1] {path.name} compute_ms={median_ms:.6f} "
                  f"aux_Bpp={auxiliary / pixels:.3f} mismatch={max_mismatch + min_mismatch}")
            del image, maxima, minima

    throughput = total_pixels / 1e6 / total_median_s
    with open(output / "summary.txt", "w") as handle:
        handle.write(f"images={len(files)}\npixels={total_pixels}\n"
                     f"aggregate_mpix_s={throughput:.6f}\n"
                     f"auxiliary_Bpp={total_aux / total_pixels:.6f}\n"
                     f"oracle_checked={int(check)}\nmismatches={total_mismatches}\n"
                     f"cupy={cp.__version__}\n"
                     f"cuda_runtime={cp.cuda.runtime.runtimeGetVersion()}\n"
                     f"opencv={cv2.__version__}\n")
    print(f"[CUCIM-EXTREMA-v0.1-OK] images={len(files)} mpix_s={throughput:.3f} "
          f"aux_Bpp={total_aux / total_pixels:.3f} mismatches={total_mismatches} "
          f"output={output}")
    return 0 if total_mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
