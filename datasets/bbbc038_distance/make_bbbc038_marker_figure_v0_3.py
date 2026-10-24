#!/usr/bin/env python3
"""Build the BBBC038 distance-transform marker figure for the Sparse-Seam paper.

Inputs are the derived artefacts of the official CC0 BBBC038v1 stage1_train
archive (SHA-256 DCB6EDC2690F137406638B2309581A71522C4DFF19157D118453B448DCDDCB68),
i.e. exactly the distance maps that the timed experiments consume.

The regional maxima drawn here are recomputed from the distance map by the same
8-connected plateau rule the derivation used, and the script asserts that the
recomputation reproduces the stored marker mask pixel for pixel before it draws
anything.  Nothing in the figure is hand-placed.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

# Same red the method figure uses for regional maxima.
MAX_RED = "#C4453E"
INK = "#1F2A37"

# Sample choice.  Only twelve samples have an archived source image, so the
# choice is made among those.  This one is the densest of them and its
# INSTANCE-level rates (86 ground-truth instances, 95.3% carrying a marker,
# 19.8% carrying more than one) are the closest of the twelve to the corpus
# rates of 97.770% and 27.202%.
#
# Note for anyone extending this: those corpus rates are per ground-truth
# INSTANCE, and cannot be reproduced as a share of 8-connected foreground
# COMPONENTS -- touching nuclei merge into one component, so the two
# denominators differ.  For this field the component-level share of components
# carrying more than one marker is 35.9%, and inside the detail window it is
# 12.5%.  Do not describe either as matching the corpus rate.
SAMPLE_ID = "0121d6759c5adb290c8e828fc882f37dfaf3663ec885c663859948c154a443ed"

# Zoom window (x0, y0, w, h) in pixels.  It holds four substantial components
# (683, 481, 439 and 388 px inside the window) plus four fragments clipped by the
# window edge.  Three of the four are singly seeded; the 683 px one is an
# elongated, nearly flat ridge and carries six markers.  The window is sized so
# that individual marker pixels survive printing.
ZOOM = (204, 140, 96, 76)


def regional_maxima(image: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """8-connected regional maxima of `image`, restricted to `foreground`.

    Identical rule to the derivation script: flood each equal-valued plateau
    over foreground pixels only, and keep it when no foreground 8-neighbour is
    strictly greater.
    """
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    markers = np.zeros((height, width), dtype=np.uint8)
    for start_y, start_x in np.argwhere(foreground):
        if visited[start_y, start_x]:
            continue
        value = int(image[start_y, start_x])
        visited[start_y, start_x] = True
        pending = deque([(int(start_y), int(start_x))])
        plateau = []
        has_greater = False
        while pending:
            y, x = pending.popleft()
            plateau.append((y, x))
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= height:
                    continue
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if (dx == 0 and dy == 0) or nx < 0 or nx >= width:
                        continue
                    if not foreground[ny, nx]:
                        continue
                    neighbour = int(image[ny, nx])
                    if neighbour > value:
                        has_greater = True
                    elif neighbour == value and not visited[ny, nx]:
                        visited[ny, nx] = True
                        pending.append((ny, nx))
        if not has_greater:
            for y, x in plateau:
                markers[y, x] = 255
    return markers


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """4-neighbour dilation, used only so single-pixel markers survive printing."""
    out = mask.copy()
    for _ in range(radius):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out


def overlay(gray: np.ndarray, mask: np.ndarray, colour: str) -> np.ndarray:
    rgb = np.repeat((gray.astype(np.float64) / 255.0)[:, :, None], 3, axis=2)
    tint = np.array(
        [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)],
        dtype=np.float64,
    )
    rgb[mask] = tint
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def load(kind: str) -> np.ndarray:
        path = args.audit_dir / f"{SAMPLE_ID}_{kind}.png"
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

    source = load("source")
    foreground = load("foreground") > 0
    distance = load("distance")
    stored_markers = load("markers")

    computed = regional_maxima(distance, foreground)
    mismatches = int((computed != stored_markers).sum())
    if mismatches:
        raise SystemExit(
            f"recomputed markers disagree with the stored mask in "
            f"{mismatches} pixels; refusing to draw"
        )
    print(f"marker recomputation: {mismatches} mismatching pixels (exact)")

    marker_bool = computed > 0
    print(
        f"markers: {int(marker_bool.sum())} pixels "
        f"({100.0 * marker_bool.mean():.2f}% of the field)"
    )

    fat = dilate(marker_bool, 1)
    overlay_rgb = overlay(distance, fat, MAX_RED)

    x0, y0, zw, zh = ZOOM
    zoom_rgb = overlay_rgb[y0 : y0 + zh, x0 : x0 + zw]

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.35), dpi=400)
    panels = [
        (source, "gray", "(a)  Annotated field"),
        (distance, "gray", "(b)  8-bit normalized EDT"),
        (overlay_rgb, None, "(c)  Regional-maxima markers"),
        (zoom_rgb, None, "(d)  Detail of (c)"),
    ]
    for axis, (data, cmap, title) in zip(axes.ravel(), panels):
        if cmap:
            axis.imshow(data, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        else:
            axis.imshow(data, interpolation="nearest")
        axis.set_title(title, fontsize=8.5, color=INK, pad=4, loc="left")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#BEC9D6")
            spine.set_linewidth(0.6)

    axes.ravel()[2].add_patch(
        Rectangle(
            (x0, y0),
            zw,
            zh,
            fill=False,
            edgecolor=INK,
            linewidth=0.9,
        )
    )

    fig.tight_layout(pad=0.5, h_pad=0.9, w_pad=0.7)
    pdf_path = args.output_dir / "bbbc038_markers.pdf"
    png_path = args.output_dir / "bbbc038_markers.png"
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(png_path, format="png", facecolor="white")
    plt.close(fig)
    print(f"PDF={pdf_path}")
    print(f"PNG={png_path}")


if __name__ == "__main__":
    main()
