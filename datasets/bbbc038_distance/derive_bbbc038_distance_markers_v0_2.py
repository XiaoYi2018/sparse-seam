#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def edt_1d(values: np.ndarray) -> np.ndarray:
    size = values.shape[0]
    sites = np.empty(size, dtype=np.int32)
    boundaries = np.empty(size + 1, dtype=np.float64)
    output = np.empty(size, dtype=np.float64)
    envelope_size = 0
    sites[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf
    for q in range(1, size):
        q_value = values[q] + q * q
        while True:
            site = int(sites[envelope_size])
            intersection = (
                q_value - (values[site] + site * site)
            ) / (2.0 * (q - site))
            if intersection > boundaries[envelope_size]:
                break
            envelope_size -= 1
        envelope_size += 1
        sites[envelope_size] = q
        boundaries[envelope_size] = intersection
        boundaries[envelope_size + 1] = np.inf
    envelope_index = 0
    for q in range(size):
        while boundaries[envelope_index + 1] < q:
            envelope_index += 1
        delta = q - int(sites[envelope_index])
        output[q] = (
            delta * delta + values[int(sites[envelope_index])]
        )
    return output


def euclidean_distance_transform(foreground: np.ndarray) -> np.ndarray:
    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    large = 1.0e15
    first = np.where(padded, large, 0.0).astype(np.float64)
    for y in range(first.shape[0]):
        first[y, :] = edt_1d(first[y, :])
    second = np.empty_like(first)
    for x in range(first.shape[1]):
        second[:, x] = edt_1d(first[:, x])
    return np.sqrt(second[1:-1, 1:-1])


def regional_maxima_labels(
    image: np.ndarray,
    foreground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    visited = np.zeros((height, width), dtype=np.bool_)
    marker_mask = np.zeros((height, width), dtype=np.uint8)
    marker_labels = np.zeros((height, width), dtype=np.int32)
    marker_id = 0
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
                    if (
                        (dx == 0 and dy == 0)
                        or nx < 0
                        or nx >= width
                        or not foreground[ny, nx]
                    ):
                        continue
                    neighbor = int(image[ny, nx])
                    if neighbor > value:
                        has_greater = True
                    if neighbor == value and not visited[ny, nx]:
                        visited[ny, nx] = True
                        pending.append((ny, nx))
        if not has_greater:
            marker_id += 1
            for y, x in plateau:
                marker_mask[y, x] = 255
                marker_labels[y, x] = marker_id
    return marker_mask, marker_labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("work_root", type=Path)
    args = parser.parse_args()

    extracted = args.work_root / "extracted"
    derived = args.work_root / "derived_marker_watershed_v0.2"
    distance_root = derived / "distance_maps"
    audit_root = derived / "audit_examples"
    if extracted.exists() or derived.exists():
        raise RuntimeError(
            "Refusing to overwrite an existing extracted/derived directory")
    extracted.mkdir(parents=True)
    distance_root.mkdir(parents=True)
    audit_root.mkdir(parents=True)

    with zipfile.ZipFile(args.archive) as archive:
        archive.extractall(extracted)

    sample_roots = sorted(
        path
        for path in extracted.iterdir()
        if path.is_dir()
        and (path / "images").is_dir()
        and (path / "masks").is_dir()
    )
    if len(sample_roots) != 670:
        raise RuntimeError(
            f"Expected 670 samples, found {len(sample_roots)}")

    rows = []
    total_instances = 0
    total_hit = 0
    total_multiple = 0
    total_markers = 0
    for sample_index, sample_root in enumerate(sample_roots):
        image_files = sorted((sample_root / "images").glob("*.png"))
        mask_files = sorted((sample_root / "masks").glob("*.png"))
        if len(image_files) != 1 or not mask_files:
            raise RuntimeError(f"Unexpected structure: {sample_root}")
        source = np.asarray(
            Image.open(image_files[0]).convert("L"),
            dtype=np.uint8,
        )
        foreground = np.zeros(source.shape, dtype=np.bool_)
        instance_masks = []
        for mask_path in mask_files:
            mask = np.asarray(
                Image.open(mask_path).convert("L"),
                dtype=np.uint8,
            ) > 0
            if mask.shape != source.shape:
                raise RuntimeError(f"Mask size mismatch: {mask_path}")
            foreground |= mask
            instance_masks.append(mask)

        distance = euclidean_distance_transform(foreground)
        max_distance = float(distance.max(initial=0.0))
        if max_distance > 0:
            distance_8u = np.rint(
                distance * (255.0 / max_distance)
            ).clip(0, 255).astype(np.uint8)
        else:
            distance_8u = np.zeros(source.shape, dtype=np.uint8)
        distance_8u[~foreground] = 0
        marker_mask, marker_labels = regional_maxima_labels(
            distance_8u, foreground)
        marker_count = int(marker_labels.max(initial=0))

        instances_hit = 0
        instances_multiple = 0
        for instance_mask in instance_masks:
            labels = np.unique(marker_labels[instance_mask])
            labels = labels[labels > 0]
            instances_hit += int(labels.size > 0)
            instances_multiple += int(labels.size > 1)

        sample_id = sample_root.name
        distance_path = distance_root / f"{sample_id}.png"
        Image.fromarray(distance_8u, mode="L").save(
            distance_path, optimize=True)
        if sample_index < 12:
            Image.fromarray(source, mode="L").save(
                audit_root / f"{sample_id}_source.png", optimize=True)
            Image.fromarray(
                foreground.astype(np.uint8) * 255, mode="L"
            ).save(
                audit_root / f"{sample_id}_foreground.png",
                optimize=True,
            )
            Image.fromarray(distance_8u, mode="L").save(
                audit_root / f"{sample_id}_distance.png", optimize=True)
            Image.fromarray(marker_mask, mode="L").save(
                audit_root / f"{sample_id}_markers.png", optimize=True)

        rows.append(
            {
                "sample_id": sample_id,
                "width": source.shape[1],
                "height": source.shape[0],
                "gt_instances": len(mask_files),
                "max_distance": f"{max_distance:.9f}",
                "foreground_pixels": int(foreground.sum()),
                "marker_pixels": int((marker_mask > 0).sum()),
                "marker_components": marker_count,
                "instances_hit": instances_hit,
                "instances_with_multiple_markers": instances_multiple,
                "distance_map_sha256": sha256(distance_path),
            }
        )
        total_instances += len(mask_files)
        total_hit += instances_hit
        total_multiple += instances_multiple
        total_markers += marker_count
        print(
            f"[BBBC038-DERIVE] {sample_index + 1}/670 "
            f"{sample_id} gt={len(mask_files)} "
            f"markers={marker_count} hit={instances_hit} "
            f"multi={instances_multiple}",
            flush=True,
        )

    manifest = derived / "application_manifest_v0.2.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    record = derived / "source_record_v0.2.txt"
    record.write_text(
        "\n".join(
            [
                "dataset=BBBC038v1 stage1_train",
                "source_url=https://data.broadinstitute.org/bbbc/"
                "BBBC038/stage1_train.zip",
                f"archive_sha256={sha256(args.archive)}",
                "license=CC0",
                "derivation=ground-truth union mask -> exact Euclidean "
                "distance transform -> per-image monotone 8-bit "
                "normalization -> 8-connected regional-max markers",
                "watershed_role=regional-max components are the seeds for "
                "marker-controlled watershed on the negative distance map",
                f"samples={len(rows)}",
                f"gt_instances={total_instances}",
                f"instances_hit={total_hit}",
                f"instance_hit_rate={total_hit / total_instances:.9f}",
                f"instances_with_multiple_markers={total_multiple}",
                f"marker_components={total_markers}",
                f"manifest_sha256={sha256(manifest)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[BBBC038-DERIVE-v0.2-OK] "
        f"samples={len(rows)} gt_instances={total_instances} "
        f"instances_hit={total_hit} "
        f"hit_rate={total_hit / total_instances:.9f} "
        f"instances_multiple={total_multiple} "
        f"marker_components={total_markers} "
        f"manifest_sha256={sha256(manifest)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
