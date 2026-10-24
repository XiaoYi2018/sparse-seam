"""Aggregate one campaign of tools/run_all_campaigns.ps1 into the numbers of the paper.

Usage: python3 aggregate_campaign.py <results_dir>
Throughput is total pixels divided by the sum of per-image median compute times;
percentiles are linear-interpolated.  Every figure is printed with its mismatch
count against the oracle and the cross-route check.
"""
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

R = sys.argv[1]


class Missing(Exception):
    pass


def rows(pattern):
    paths = sorted(glob.glob(os.path.join(R, pattern)))
    if not paths:
        raise Missing(pattern)
    out = []
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            out.extend(csv.DictReader(handle))
    return out


def pct(values, p):
    v = sorted(values)
    pos = p * (len(v) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def pixels(r):
    return int(r["width"]) * int(r["height"])


def agg(rs):
    pix = sum(pixels(r) for r in rs)
    sec = sum(float(r["median_compute_ms"]) / 1000 for r in rs)
    aux = sum(float(r["auxiliary_bytes"]) for r in rs)
    tot = sum(float(r.get("total_allocated_bytes") or 0) for r in rs)
    mism = sum(int(r.get(k) or 0) for r in rs for k in ("oracle_max_mismatches", "oracle_min_mismatches", "max_mismatches", "min_mismatches", "cross_max_mismatches", "cross_min_mismatches", "cross_route_mismatches"))
    return {"n": len(rs), "mpix_s": pix / 1e6 / sec, "aux_Bpp": aux / pix, "total_Bpp": tot / pix if tot else aux / pix + 3, "mismatches": mism}


def by(rs, key):
    d = defaultdict(list)
    for r in rs:
        d[r[key]].append(r)
    return d


def show(title, table):
    print(f"\n## {title}")
    for name, a in table:
        print(f"  {name:<34} {a['mpix_s']:10.3f} MP/s  total {a['total_Bpp']:7.3f}  aux {a['aux_Bpp']:7.3f} B/px  n={a['n']}  mism={a['mismatches']}")


# ---------------- DIV2K primary and baselines ----------------
div = {}
for order in ("gos", "sog"):
    for alg, rs in by(rows(f"gpu_common_v0.6_DIV2K_valid100_30rep_{order}/gpu_common_v0_6.csv"), "algorithm").items():
        div[(alg, order)] = rs
singles = {
    "NPP+ExactRepair": rows("npp_labelmarkersuf_exact_repair_v0.3_div2k_validation_w5_r30/*.csv"),
    "IWPP": rows("iwpp_reconstruction_cuda_v0.1_DIV2K_valid100_30rep_v0.2/iwpp_reconstruction_v0_1.csv"),
    "Blin dual-tree": rows("blin_optim1d_relaxed_flatten_extrema_v0.6.0_div2k_validation_w5_r30/*.csv"),
}
try:
    singles["cuCIM"] = rows("cucim_extrema_v0.1_DIV2K_valid100_w5_r30/cucim_extrema_v0_1.csv")
    for alg, rs in by(rows("yacclab_gray_ccl_v0.2_DIV2K_valid100_w5_r30/*.csv"), "algorithm").items():
        singles[alg] = rs
except Missing as m:
    print("skipped new DIV2K baselines:", m)
show("DIV2K, in-process routes per order", [(f"{a} [{o}]", agg(rs)) for (a, o), rs in sorted(div.items())])
show("DIV2K, separate-process routes", [(a, agg(rs)) for a, rs in singles.items()])
sparse = {o: agg(div[("Sparse-Seam-v1.0", o)])["mpix_s"] for o in ("gos", "sog")}
print("\n  speedups of Sparse-Seam (per order gos / sog):")
for (a, o), rs in sorted(div.items()):
    if a.startswith("Sparse") or o != "gos":
        continue
    print(f"    vs {a:<30} {sparse['gos'] / agg(div[(a, 'gos')])['mpix_s']:.3f} / {sparse['sog'] / agg(div[(a, 'sog')])['mpix_s']:.3f}")
for a, rs in singles.items():
    x = agg(rs)["mpix_s"]
    print(f"    vs {a:<30} {sparse['gos'] / x:.3f} / {sparse['sog'] / x:.3f}")


def paired(sparse_rows, other_rows):
    s = {r["image"]: float(r["median_compute_ms"]) for r in sparse_rows}
    o = {r["image"]: float(r["median_compute_ms"]) for r in other_rows}
    sp = [o[k] / s[k] for k in s]
    return statistics.median(sp), pct(sp, 0.25), pct(sp, 0.75), sum(v > 1 for v in sp), len(sp)


print("\n  per-image speedup of Sparse-Seam: median (Q1-Q3), wins/n")
for o in ("gos", "sog"):
    for a in ("Global-UF-CUDA", "FlatZone-OLE-CUDA"):
        m, q1, q3, w, n = paired(div[("Sparse-Seam-v1.0", o)], div[(a, o)])
        print(f"    [{o}] vs {a:<24} {m:.3f} ({q1:.3f}-{q3:.3f})  {w}/{n}")
    for a, rs in singles.items():
        m, q1, q3, w, n = paired(div[("Sparse-Seam-v1.0", o)], rs)
        print(f"    [{o}] vs {a:<24} {m:.3f} ({q1:.3f}-{q3:.3f})  {w}/{n}")

# h2h on DIV2K
for o in ("gos", "sog"):
    rs = div[("Sparse-Seam-v1.0", o)]
    print(f"  h2h Sparse [{o}] {sum(pixels(r) for r in rs) / 1e6 / sum(float(r['median_h2h_ms']) / 1000 for r in rs):.3f} MP/s")

# ---------------- quantization ----------------
print("\n## Quantization (pooled orders)")
q = defaultdict(list)
for path in glob.glob(os.path.join(R, "gpu_common_quant_v0.7_DIV2K_valid100_q*_w5_r30_*/gpu_common_quant_v0_7.csv")):
    for r in rows(os.path.relpath(path, R)):
        q[(int(r["quantization_levels"]), r["algorithm"])].append(r)
for lv in sorted({k[0] for k in q}, reverse=True):
    v = {a: agg(q[(lv, a)]) for (l, a) in q if l == lv}
    s = v["Sparse-Seam-v1.0"]["mpix_s"]; g = v["Global-UF-CUDA"]["mpix_s"]; e = v["FlatZone-OLE-CUDA"]["mpix_s"]
    m = sum(x["mismatches"] for x in v.values())
    print(f"  q{lv:<4} Dense {g:9.3f}  OLE {e:9.3f}  Sparse {s:9.3f}  vsDense {s / g:.3f}  vsOLE {s / e:.3f}  mism={m}")

# ---------------- scaling and alignment ----------------
for tag in ("scaling_repeat", "scaling_alignment"):
    print(f"\n## {tag} (median of six per-trial throughputs)")
    per = defaultdict(list)
    mism = 0
    for path in sorted(glob.glob(os.path.join(R, f"gpu_common_quant_v0.7_{tag}_v0.2_trial*/gpu_common_quant_v0_7.csv"))):
        for r in rows(os.path.relpath(path, R)):
            per[(r["image"], r["algorithm"])].append(pixels(r) / 1e6 / (float(r["median_compute_ms"]) / 1000))
            mism += sum(int(r[k]) for k in ("oracle_max_mismatches", "oracle_min_mismatches", "cross_max_mismatches", "cross_min_mismatches"))
    images = sorted({k[0] for k in per}, key=lambda s: (s.split("_")[0], int(s.rsplit("_", 1)[1].split("x")[0]), s))
    for im in images:
        g = statistics.median(per[(im, "Global-UF-CUDA")]); e = statistics.median(per[(im, "FlatZone-OLE-CUDA")]); s = statistics.median(per[(im, "Sparse-Seam-v1.0")])
        print(f"  {im:<34} Dense {g:9.1f}  OLE {e:9.1f}  Sparse {s:9.1f}  vsDense {s / g:.3f}  vsOLE {s / e:.3f}  trials={len(per[(im, 'Global-UF-CUDA')])}")
    print(f"  mismatches={mism}")

# ---------------- BSDS500 ----------------
print("\n## BSDS500")
for o in ("gos", "sog"):
    rs = rows(f"gpu_common_quant_v0.7_BSDS500_v0.2_q256_w5_r30_{o}/gpu_common_quant_v0_7.csv")
    a = by(rs, "algorithm")
    for split in ("all", "train", "val", "test"):
        sel = lambda rr: [r for r in rr if split == "all" or r["image"].startswith(split + "_")]
        g = agg(sel(a["Global-UF-CUDA"]))["mpix_s"]; e = agg(sel(a["FlatZone-OLE-CUDA"]))["mpix_s"]; s = agg(sel(a["Sparse-Seam-v1.0"]))["mpix_s"]
        print(f"  [{o}] {split:<5} n={len(sel(a['Global-UF-CUDA'])):3d} Dense {g:9.3f}  OLE {e:9.3f}  Sparse {s:9.3f}  vsDense {s / g:.3f}  vsOLE {s / e:.3f}")
    sm = {r["image"]: float(r["median_compute_ms"]) for r in a["Sparse-Seam-v1.0"]}
    for other in ("Global-UF-CUDA", "FlatZone-OLE-CUDA"):
        om = {r["image"]: float(r["median_compute_ms"]) for r in a[other]}
        w = sum(sm[k] < om[k] for k in sm); t = sum(sm[k] == om[k] for k in sm)
        print(f"  [{o}] vs {other}: wins {w}, ties {t}, losses {len(sm) - w - t}")
    print(f"  [{o}] mismatches={sum(agg(v)['mismatches'] for v in a.values())}")
try:
    for name, pattern in (("cuCIM", "cucim_extrema_v0.1_BSDS500_v0.2_w5_r30/*.csv"), ("YACCLAB", "yacclab_gray_ccl_v0.2_BSDS500_v0.2_w5_r30/*.csv")):
        for alg, rs in by(rows(pattern), "algorithm").items():
            x = agg(rs); print(f"  {alg:<30} {x['mpix_s']:9.3f} MP/s aux {x['aux_Bpp']:.3f} mism={x['mismatches']}")
except Missing as m:
    print("  skipped:", m)

# ---------------- BBBC038 ----------------
print("\n## BBBC038 distance maps")
for o in ("gos", "sog"):
    a = by(rows(f"gpu_common_quant_v0.7_BBBC038_distance_v0.3_q256_w5_r30_{o}/gpu_common_quant_v0_7.csv"), "algorithm")
    v = {k: agg(rs) for k, rs in a.items()}
    s = v["Sparse-Seam-v1.0"]["mpix_s"]
    h2h = {k: sum(pixels(r) for r in rs) / 1e6 / sum(float(r["median_h2h_ms"]) / 1000 for r in rs) for k, rs in a.items()}
    sm = {r["image"]: float(r["median_compute_ms"]) for r in a["Sparse-Seam-v1.0"]}
    wins = {k: sum(sm[i] < float(r["median_compute_ms"]) for r in rs for i in [r["image"]]) for k, rs in a.items() if not k.startswith("Sparse")}
    print(f"  [{o}] " + "  ".join(f"{k} {x['mpix_s']:.3f}" for k, x in v.items()) + f"  vsDense {s / v['Global-UF-CUDA']['mpix_s']:.3f} vsOLE {s / v['FlatZone-OLE-CUDA']['mpix_s']:.3f}  wins {wins}")
    print(f"  [{o}] h2h " + "  ".join(f"{k} {x:.3f}" for k, x in h2h.items()) + f"  mism={sum(x['mismatches'] for x in v.values())}")
try:
    for name, pattern in (("cuCIM", "cucim_extrema_v0.1_BBBC038_distance_v0.3_w5_r30/*.csv"), ("YACCLAB", "yacclab_gray_ccl_v0.2_BBBC038_distance_v0.3_w5_r30/*.csv")):
        for alg, rs in by(rows(pattern), "algorithm").items():
            x = agg(rs); print(f"  {alg:<30} {x['mpix_s']:9.3f} MP/s aux {x['aux_Bpp']:.3f} mism={x['mismatches']}")
except Missing as m:
    print("  skipped:", m)

# ---------------- configuration study ----------------
try:
    print("\n## Configuration study")
    cfg = rows("config_study_v0_8/config_study_summary.csv")
    base = [float(r["aggregate_mpix_s"]) for r in cfg if r["config"] == "tile32_threads256"][0]
    for r in cfg:
        print(f"  {r['config']:<20} {float(r['aggregate_mpix_s']):10.3f}  rel {100 * (float(r['aggregate_mpix_s']) / base - 1):8.3f}%  aux {float(r['auxiliary_Bpp']):.3f}")
except Missing as m:
    print("skipped configuration study:", m)

# ---------------- seam statistics ----------------
print("\n## Seam statistics (corpus-weighted)")
seam = rows("gpu_seam_sparsity_DIV2K_valid100_q2_q256_v0.2/seam_sparsity.csv")
for lv, rs in sorted(by(seam, "levels").items(), key=lambda kv: -int(kv[0])):
    for tile, tr in sorted(by(rs, "tile").items()):
        mp = sum(float(r["megapixels"]) for r in tr)
        print(f"  q{lv:<4} tile {tile:<4} seam_pixel_fraction {100 * sum(float(r['seam_pixels']) for r in tr) / (mp * 1e6):.3f}%  fragments/MP {sum(float(r['seam_fragments']) for r in tr) / mp:.1f}")


# ---------------- final-campaign additions ----------------
print("\n## Quantization: labeling routes (YACCLAB v0.2)")
for lv in (256, 32, 16, 8, 4, 2):
    try:
        y = by(rows(f"yacclab_gray_ccl_v0.2_DIV2K_valid100_q{lv}_w5_r30/*.csv"), "algorithm")
    except Missing as m:
        print("  skipped", m); continue
    q_s = agg(q[(lv, "Sparse-Seam-v1.0")])["mpix_s"]
    ke = agg(y["KE-Gray-YACCLAB"]); uf = agg(y["UF-Gray-YACCLAB"])
    print(f"  q{lv:<4} KE {ke['mpix_s']:9.3f}  UF {uf['mpix_s']:9.3f}  Sparse {q_s:9.3f}  vsKE {q_s / ke['mpix_s']:.3f}  vsUF {q_s / uf['mpix_s']:.3f}  mism={ke['mismatches'] + uf['mismatches']}")

for tag in ("scaling_repeat", "scaling_alignment"):
    print(f"\n## {tag}: labeling routes (median of six trials)")
    per = defaultdict(list); sp = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(R, f"yacclab_gray_ccl_v0.2_{tag}_v0.2_trial*/*.csv"))):
        for r in rows(os.path.relpath(path, R)):
            per[(r["image"], r["algorithm"])].append(pixels(r) / 1e6 / (float(r["median_compute_ms"]) / 1000))
    for path in sorted(glob.glob(os.path.join(R, f"gpu_common_quant_v0.7_{tag}_v0.2_trial*/gpu_common_quant_v0_7.csv"))):
        for r in rows(os.path.relpath(path, R)):
            if r["algorithm"] == "Sparse-Seam-v1.0":
                sp[r["image"]].append(pixels(r) / 1e6 / (float(r["median_compute_ms"]) / 1000))
    for im in sorted(sp, key=lambda s: (s.split("_")[0], int(s.rsplit("_", 1)[1].split("x")[0]), s)):
        s = statistics.median(sp[im]); ke = statistics.median(per[(im, "KE-Gray-YACCLAB")]); uf = statistics.median(per[(im, "UF-Gray-YACCLAB")])
        print(f"  {im:<34} KE {ke:9.1f}  UF {uf:9.1f}  Sparse {s:9.1f}  vsKE {s / ke:.3f}  vsUF {s / uf:.3f}  trials={len(per[(im, 'KE-Gray-YACCLAB')])}")

print("\n## Ablation (same campaign, quantization harness, pooled orders)")
for name, v1pat, lv in (("DIV2K_q256", "gpu_common_quant_v0.7_DIV2K_valid100_q256_w5_r30_*", None),
                        ("DIV2K_q32", "gpu_common_quant_v0.7_DIV2K_valid100_q32_w5_r30_*", None),
                        ("DIV2K_q8", "gpu_common_quant_v0.7_DIV2K_valid100_q8_w5_r30_*", None),
                        ("DIV2K_q2", "gpu_common_quant_v0.7_DIV2K_valid100_q2_w5_r30_*", None),
                        ("BSDS500_v0.2", "gpu_common_quant_v0.7_BSDS500_v0.2_q256_w5_r30_*", None),
                        ("BBBC038_distance_v0.3", "gpu_common_quant_v0.7_BBBC038_distance_v0.3_q256_w5_r30_*", None)):
    try:
        four = [r for r in rows(f"ablation_four_edge_v0.6_{name}_w5_r30_*/*.csv") if r["algorithm"] == "Sparse-Seam-v0.9-variant1"]
    except Missing as m:
        print("  skipped", m); continue
    ke = [r for r in rows(v1pat + "/*.csv") if r["algorithm"] == "Sparse-Seam-v1.0"]
    a4 = agg(four); ak = agg(ke)
    print(f"  {name:<24} four-edge {a4['mpix_s']:9.1f}  Komura {ak['mpix_s']:9.1f}  ratio {ak['mpix_s'] / a4['mpix_s']:.3f}  mism={a4['mismatches']}")

print("\n## Per-image comparison with the separate-process routes (BSDS500, BBBC038)")
for name, harness, single in (
        ("BSDS500", "gpu_common_quant_v0.7_BSDS500_v0.2_q256_w5_r30_{}/gpu_common_quant_v0_7.csv",
         ("cucim_extrema_v0.1_BSDS500_v0.2_w5_r30/*.csv", "yacclab_gray_ccl_v0.2_BSDS500_v0.2_w5_r30/*.csv")),
        ("BBBC038", "gpu_common_quant_v0.7_BBBC038_distance_v0.3_q256_w5_r30_{}/gpu_common_quant_v0_7.csv",
         ("cucim_extrema_v0.1_BBBC038_distance_v0.3_w5_r30/*.csv", "yacclab_gray_ccl_v0.2_BBBC038_distance_v0.3_w5_r30/*.csv"))):
    try:
        others = {}
        for pattern in single:
            others.update(by(rows(pattern), "algorithm"))
    except Missing as m:
        print("  skipped", m); continue
    for o in ("gos", "sog"):
        sm = {r["image"]: float(r["median_compute_ms"]) for r in rows(harness.format(o)) if r["algorithm"] == "Sparse-Seam-v1.0"}
        for alg, rs in sorted(others.items()):
            om = {r["image"]: float(r["median_compute_ms"]) for r in rs}
            assert set(om) == set(sm), (name, alg)
            ratio = [om[k] / sm[k] for k in sm]
            w = sum(sm[k] < om[k] for k in sm); t = sum(sm[k] == om[k] for k in sm)
            print(f"  {name:<8} [{o}] vs {alg:<26} median ratio {statistics.median(ratio):.3f} ({pct(ratio, 0.25):.3f}-{pct(ratio, 0.75):.3f})  wins {w}, ties {t}, losses {len(sm) - w - t}")
