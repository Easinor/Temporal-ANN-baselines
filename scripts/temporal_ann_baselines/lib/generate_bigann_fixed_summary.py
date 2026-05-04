from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/zwan/Temporal-ANN-baselines/results/temporal_baselines")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_row(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    raise KeyError(f"row not found: {conds}")


def gb(bytes_str: str) -> str:
    return f"{int(bytes_str) / 1e9:.6f}"


def main() -> None:
    out = ROOT / "bigann1M_fixed_summary.md"

    rf = read_rows(ROOT / "bigann1M_rangefilteredann.tsv")
    unify = read_rows(ROOT / "bigann1M_unify_hsig.tsv")
    serf = read_rows(ROOT / "bigann1M_serf.tsv")
    dsg = read_rows(ROOT / "bigann1M_dsg.tsv")
    irg = read_rows(ROOT / "bigann1M_irangegraph.tsv")

    rows: list[tuple[str, str, dict[str, str]]] = []

    for rng in ["100", "1000", "10000", "100000", "1000000"]:
        rows.append((
            "RangeFilteredANN (beam=160, fbm=1)",
            rng,
            pick_row(rf, baseline="rangefilteredann_vamana_tree", range=rng, beam_size="160", final_beam_multiply="1"),
        ))

    for rng in ["100", "1000", "10000", "100000", "1000000"]:
        rows.append((
            "UNIFY (hsig_hybrid, ef=160, al=16)",
            rng,
            pick_row(unify, baseline="hsig_hybrid", range=rng, ef="160", al="16"),
        ))

    for rng in ["100", "1000", "10000", "100000", "1000000"]:
        rows.append((
            "SeRF (ef_con=80, ef_search=160)",
            rng,
            pick_row(serf, baseline="serf_2d", range=rng, index_k="16", ef_construction="80", ef_max="300", ef_search="160"),
        ))

    for rng in ["100", "1000", "10000", "100000", "1000000"]:
        rows.append((
            "DSG (ef_con=80, search_ef=160)",
            rng,
            pick_row(dsg, baseline="dsg_static", range=rng, search_ef="160"),
        ))

    for rng in ["100", "1000", "10000", "100000", "1000000"]:
        rows.append((
            "iRangeGraph (ef_con=80, ef=160)",
            rng,
            pick_row(irg, baseline="irangegraph", range=rng, ef="160"),
        ))

    with out.open("w") as f:
        f.write("# BigANN1M Fixed-Parameter Summary\n\n")
        f.write("| baseline | range | build_parallel | query_parallel | build_seconds | space_gb | recall | avg_ms | qps | kpqs | params |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for baseline, rng, row in rows:
            if baseline.startswith("RangeFilteredANN"):
                build_parallel = "yes"
                query_parallel = "yes"
            elif baseline.startswith("UNIFY"):
                build_parallel = "yes"
                query_parallel = "partial"
            elif baseline.startswith("SeRF"):
                build_parallel = "no"
                query_parallel = "no"
            elif baseline.startswith("DSG"):
                build_parallel = "no"
                query_parallel = "no"
            elif baseline.startswith("iRangeGraph"):
                build_parallel = "yes"
                query_parallel = "no"
            else:
                build_parallel = "unknown"
                query_parallel = "unknown"
            params = []
            for key in ["beam_size", "final_beam_multiply", "ef", "al", "index_k", "ef_construction", "ef_max", "ef_search", "search_ef"]:
                if key in row and row[key] != "":
                    params.append(f"{key}={row[key]}")
            qps = row.get("qps", row.get("kpqs", ""))
            f.write(
                f"| {baseline} | {rng} | {build_parallel} | {query_parallel} | {row['build_seconds']} | {gb(row['space_usage_bytes'])} | "
                f"{row['recall']} | {row['avg_ms']} | {qps} | {row.get('kpqs', '')} | {', '.join(params)} |\n"
            )

    print(out)


if __name__ == "__main__":
    main()
