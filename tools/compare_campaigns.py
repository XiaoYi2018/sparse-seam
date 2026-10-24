"""Session-to-session stability between two campaigns of the same code and protocol.

Usage: python3 compare_campaigns.py <aggregate_earlier.txt> <aggregate_final.txt>
Inputs are outputs of aggregate_campaign.py for two complete campaigns.  Every device-compute throughput figure
of the DIV2K, quantization, BSDS500 and BBBC038 sections is paired by section,
order/split/level and route; host-to-host figures, which include the PCIe
transfers, and the nine configuration aggregates are reported separately.
"""
import re
import sys

SECTIONS = ("DIV2K, in-process routes per order", "DIV2K, separate-process routes",
            "Quantization (pooled orders)", "BSDS500", "BBBC038 distance maps")
NUM = r"([0-9]+\.[0-9]+)"


def figures(path):
    compute, h2h, config, section = {}, {}, {}, None
    for line in open(path, encoding="utf-8"):
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        s = line.strip()
        if section == "Configuration study" and s.startswith("tile"):
            config[(section, s.split()[0])] = float(s.split()[1])
            continue
        if section not in SECTIONS or not s:
            continue
        if s.startswith("h2h Sparse"):                         # DIV2K host-to-host
            m = re.match(r"h2h Sparse (\[\w+\]) " + NUM, s)
            h2h[(section, m.group(1), "Sparse")] = float(m.group(2))
        elif re.match(r"\[\w+\] h2h ", s):                      # BBBC038 host-to-host
            tag = s.split()[0]
            for name, v in re.findall(r"([A-Za-z][\w.+-]*) " + NUM, s.split("h2h", 1)[1]):
                h2h[(section, tag, name)] = float(v)
        elif re.match(r"^\S.*?\s+" + NUM + " MP/s", s):          # one route per line
            m = re.match(r"^(\S.*?)\s+(\[\w+\]\s+)?" + NUM + " MP/s", s)
            name = m.group(1) + (" " + m.group(2).strip() if m.group(2) else "")
            compute[(section, name)] = float(m.group(3))
        elif section.startswith("Quantization") and s.startswith("q"):
            lvl = s.split()[0]
            for name in ("Dense", "OLE", "Sparse"):
                compute[(section, lvl, name)] = float(re.search(name + r"\s+" + NUM, s).group(1))
        elif section == "BSDS500" and re.match(r"\[\w+\] (all|train|val|test)", s):
            tag, split = s.split()[:2]
            for name in ("Dense", "OLE", "Sparse"):
                compute[(section, tag, split, name)] = float(re.search(name + r"\s+" + NUM, s).group(1))
        elif section.startswith("BBBC038") and re.match(r"\[\w+\] Global", s):
            tag = s.split()[0]
            for name in ("Global-UF-CUDA", "FlatZone-OLE-CUDA", "Sparse-Seam-v1.0"):
                compute[(section, tag, name)] = float(re.search(re.escape(name) + r"\s+" + NUM, s).group(1))
    return compute, h2h, config


def report(title, a, b):
    common = sorted(set(a) & set(b))
    dev = sorted(((abs(b[k] / a[k] - 1) * 100, k) for k in common), reverse=True)
    print(f"{title}: paired {len(common)} (earlier only {len(set(a) - set(b))}, final only {len(set(b) - set(a))})")
    print(f"  within 1.0%: {sum(d <= 1.0 for d, _ in dev)}   within 0.5%: {sum(d <= 0.5 for d, _ in dev)}   max {dev[0][0]:.3f}%")
    for d, k in dev:
        print(f"  {d:6.3f}%  {' | '.join(k)}  {a[k]:.3f} -> {b[k]:.3f}")


(ca, ha, ga), (cb, hb, gb) = figures(sys.argv[1]), figures(sys.argv[2])
report("device-compute throughput", ca, cb)
report("host-to-host throughput", ha, hb)
report("configuration study", ga, gb)
