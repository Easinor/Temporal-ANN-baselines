"""
Summarize all bigann1M baseline results into a single comparison table.
Outputs: QPS at recall >= 0.90, 0.95, 0.99 for each (method, range).

Usage:
    python3 summarize_baselines.py [--results-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

BASELINES_ROOT = Path(__file__).resolve().parents[3]
RECALL_THRESHOLDS = [0.90, 0.95, 0.99]


class OperatingPoint(NamedTuple):
    recall: float
    qps: float


class MethodSummary(NamedTuple):
    method: str
    range_width: int
    build_seconds: float
    space_gb: float
    points: list[OperatingPoint]  # sorted by recall ascending


def qps_at_recall(points: list[OperatingPoint], threshold: float) -> float | None:
    """Max QPS among points where recall >= threshold."""
    candidates = [p.qps for p in points if p.recall >= threshold]
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Loaders for each TSV format
# ---------------------------------------------------------------------------

def load_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split("\t")
            rows.append(dict(zip(header, values)))
    return rows


def summaries_from_tsv(rows: list[dict], method_col: str = "baseline") -> list[MethodSummary]:
    """
    Generic loader for TSVs that have: baseline/range/recall/qps/build_seconds/space_usage_bytes.
    Groups rows by (method, range) and collects operating points.
    """
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    meta: dict[tuple, tuple] = {}

    for row in rows:
        method = row[method_col]
        rng = int(row["range"])
        key = (method, rng)
        recall = float(row["recall"])
        qps = float(row["qps"])
        groups[key].append(OperatingPoint(recall, qps))

        if key not in meta:
            build = float(row.get("build_seconds", 0))
            space_bytes = float(row.get("space_usage_bytes", 0))
            meta[key] = (build, space_bytes / 1e9)

    result = []
    for key, pts in groups.items():
        method, rng = key
        build, space_gb = meta[key]
        pts_sorted = sorted(pts, key=lambda p: p.recall)
        result.append(MethodSummary(method, rng, build, space_gb, pts_sorted))
    return result


def load_annlib_logs(log_dir: Path) -> list[MethodSummary]:
    """
    Parse test_timerange_parallel_merge log files.
    Each log file covers one range width; contains one ef operating point.
    """
    summaries = []
    pattern_range = re.compile(r"range(\d+)")
    pattern_kqps = re.compile(r"Find neighbors.*?:\s+[\d.]+\s+s,\s+([\d.e+]+)\s+kqps")
    pattern_recall = re.compile(r"query recall@\d+:\s+([\d.]+)")
    pattern_build = re.compile(r"Parallel merge (?:tree build|cache load) time:\s+([\d.]+)\s+s")

    for log_file in sorted(log_dir.glob("*.log")):
        m = pattern_range.search(log_file.name)
        if not m:
            continue
        rng = int(m.group(1))

        text = log_file.read_text()
        kqps_match = pattern_kqps.search(text)
        if not kqps_match:
            continue
        qps = float(kqps_match.group(1)) * 1000

        recalls = [float(r) for r in pattern_recall.findall(text)]
        recall = float(np.mean(recalls)) if recalls else 0.0

        build_times = [float(t) for t in pattern_build.findall(text)]
        build_sec = sum(build_times)

        summaries.append(MethodSummary(
            method="annlib_parallel_merge",
            range_width=rng,
            build_seconds=build_sec,
            space_gb=0.0,
            points=[OperatingPoint(recall, qps)],
        ))
    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_all_summaries(results_dir: Path) -> list[MethodSummary]:
    summaries: list[MethodSummary] = []

    def load(fname):
        p = results_dir / fname
        if p.exists():
            return load_tsv(p)
        print(f"  [skip] {fname} not found", file=sys.stderr)
        return []

    # RangeFilteredANN — prefer v2 if present
    rfa_file = "bigann1M_rangefilteredann_v2.tsv"
    if not (results_dir / rfa_file).exists():
        rfa_file = "bigann1M_rangefilteredann.tsv"
    summaries += summaries_from_tsv(load(rfa_file))

    summaries += summaries_from_tsv(load("bigann1M_dsg.tsv"))
    summaries += summaries_from_tsv(load("bigann1M_serf.tsv"))
    summaries += summaries_from_tsv(load("bigann1M_irangegraph.tsv"))
    summaries += summaries_from_tsv(load("bigann1M_unify.tsv"))

    annlib_dir = results_dir / "ANNlib" / "bigann1M" / "simple_step1000_proportion0p3"
    if annlib_dir.exists():
        summaries += load_annlib_logs(annlib_dir)
    else:
        print(f"  [skip] ANNlib logs not found at {annlib_dir}", file=sys.stderr)

    return summaries


def format_qps(v: float | None) -> str:
    if v is None:
        return "-"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.0f}"


def write_output(summaries: list[MethodSummary], output: Path | None) -> None:
    ranges = sorted({s.range_width for s in summaries})
    methods = sorted({s.method for s in summaries})

    # Build lookup: (method, range) -> MethodSummary
    lookup: dict[tuple, MethodSummary] = {}
    for s in summaries:
        lookup[(s.method, s.range_width)] = s

    # --- TSV output ---
    thresh_headers = "\t".join(f"qps@r{int(t*100)}" for t in RECALL_THRESHOLDS)
    header = f"method\trange\tbuild_s\tspace_gb\t{thresh_headers}"

    lines = [header]
    for method in methods:
        for rng in ranges:
            key = (method, rng)
            if key not in lookup:
                continue
            s = lookup[key]
            qps_cols = "\t".join(
                format_qps(qps_at_recall(s.points, t)) for t in RECALL_THRESHOLDS
            )
            lines.append(
                f"{method}\t{rng}\t{s.build_seconds:.1f}\t{s.space_gb:.2f}\t{qps_cols}"
            )

    tsv_text = "\n".join(lines) + "\n"

    if output:
        output.write_text(tsv_text)
        print(f"Wrote {len(lines)-1} rows to {output}")
    else:
        # Pretty-print to stdout
        col_widths = [max(len(row.split("\t")[i]) for row in lines)
                      for i in range(len(lines[0].split("\t")))]
        for i, line in enumerate(lines):
            cols = line.split("\t")
            print("  ".join(c.ljust(col_widths[j]) for j, c in enumerate(cols)))
            if i == 0:
                print("  ".join("-" * w for w in col_widths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASELINES_ROOT / "results" / "temporal_baselines",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write TSV to this file instead of printing to stdout",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}", file=sys.stderr)
    summaries = collect_all_summaries(args.results_dir)
    print(f"Loaded {len(summaries)} (method, range) entries", file=sys.stderr)
    write_output(summaries, args.output)


if __name__ == "__main__":
    main()
