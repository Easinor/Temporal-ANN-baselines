from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
import numpy as np

from common import (
    build_queries,
    compute_recall,
    compress_scalars_and_ranges,
    ensure_float32,
    file_size_bytes,
    kpqs,
    load_timerange_artifacts,
    load_vectors,
    milliseconds_per_query,
    normalize_angular,
    process_rss_bytes,
    qps,
    write_tsv,
)

BASELINES_ROOT = Path(__file__).resolve().parents[3]


def default_thread_count() -> int:
    for key in ("THREADS", "OMP_NUM_THREADS"):
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def load_unify_modules(unify_repo: Path):
    code_dir = unify_repo / "benchmark" / "code"
    sys.path.insert(0, str(code_dir))
    from alg_hannlib import HannLib  # type: ignore
    from alg_hnswlib import HnswLib  # type: ignore

    return HannLib, HnswLib


def run_hybrid_queries(
    index,
    queries: np.ndarray,
    ranges: np.ndarray,
    empty_mask: np.ndarray,
    k: int,
    num_workers: int,
) -> np.ndarray:
    ids = np.full((queries.shape[0], k), -1, dtype=np.int64)

    valid_queries = np.flatnonzero(~empty_mask)
    if valid_queries.size == 0:
        return ids

    batch_queries = np.asarray(queries[valid_queries], dtype=np.float32)
    batch_ranges = np.asarray(ranges[valid_queries], dtype=np.int64)
    result = index.batch_hybrid_query(batch_queries, batch_ranges, k, num_workers)

    if isinstance(result, tuple) and len(result) == 2:
        batch_ids = np.asarray(result[0], dtype=np.int64)
    else:
        batch_ids = np.asarray(result, dtype=np.int64)

    if batch_ids.ndim == 1:
        batch_ids = batch_ids.reshape(valid_queries.size, -1)

    ids[valid_queries, : min(k, batch_ids.shape[1])] = batch_ids[:, :k]
    return ids


def build_index(builder, index_path: Path):
    gc.collect()
    rss_before = process_rss_bytes()
    start = time.perf_counter()
    index = builder()
    build_seconds = time.perf_counter() - start
    gc.collect()
    rss_after = process_rss_bytes()
    rss_delta_bytes = max(0, rss_after - rss_before)
    index_bytes = file_size_bytes(index_path)
    if index_bytes > 0:
        space_usage_bytes = index_bytes
        space_usage_source = "index_file_bytes"
    else:
        space_usage_bytes = rss_delta_bytes
        space_usage_source = "rss_delta_bytes"
    return index, {
        "build_seconds": f"{build_seconds:.6f}",
        "space_usage_bytes": int(space_usage_bytes),
        "space_usage_source": space_usage_source,
        "rss_delta_bytes": int(rss_delta_bytes),
        "index_file_bytes": int(index_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UNIFY/HNSW baselines on ANNlib timerange artifacts.")
    parser.add_argument("--unify-repo", type=Path, default=BASELINES_ROOT / "UNIFY")
    parser.add_argument("--dataset", required=True, help="ANNlib dataset spec, e.g. ./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--dist", choices=["L2", "angular"], required=True)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--ranges", type=int, nargs="*")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hsig_hybrid", "hsig_pre", "hsig_post"],
        choices=["hsig_hybrid", "hsig_pre", "hsig_post", "hsig_cbo", "hnsw_postfilter"],
    )
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efConstruction", type=int, default=500)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--ef-list", type=int, nargs="+", default=[160])
    parser.add_argument("--al-list", type=int, nargs="+", default=[16])
    parser.add_argument("--low-range", type=float, default=0.1)
    parser.add_argument("--high-range", type=float, default=0.5)
    parser.add_argument("--index-cache-dir", type=Path, default=BASELINES_ROOT / "cache" / "unify")
    parser.add_argument("--output", type=Path, default=BASELINES_ROOT / "results" / "unify_timerange.tsv")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel query workers; 0 means auto-detect")
    args = parser.parse_args()

    HannLib, HnswLib = load_unify_modules(args.unify_repo)
    args.index_cache_dir.mkdir(parents=True, exist_ok=True)

    base = load_vectors(args.dataset, max_points=args.max_points if args.max_points > 0 else None)
    artifacts = load_timerange_artifacts(args.artifacts_dir, args.ranges)
    if artifacts.point_timestamps.shape[0] < base.shape[0]:
        raise ValueError("timestamp count is smaller than loaded base size")

    query_ids = artifacts.query_ids
    queries = build_queries(base, query_ids)

    if args.dist == "angular":
        base = normalize_angular(base)
        queries = normalize_angular(queries)

    base = ensure_float32(base)
    queries = ensure_float32(queries)

    compressed = compress_scalars_and_ranges(
        artifacts.point_timestamps[: base.shape[0]],
        artifacts.query_ranges,
    )
    scalars = compressed.scalars.astype(np.int64)

    rows: list[dict[str, object]] = []

    hsig_index = None
    hsig_metrics: dict[str, object] | None = None
    if any(m.startswith("hsig_") for m in args.methods):
        cache_path = args.index_cache_dir / "hsig.index"
        def build_hsig():
            index = HannLib(metric="angular" if args.dist == "angular" else "euclidean", method_param={"num_slots": args.B, "M": args.M, "efConstruction": args.efConstruction})
            if cache_path.exists():
                index.loadIndex(base, str(cache_path))
                index.scalars = scalars
            else:
                index.fit(base, scalars)
                index.saveIndex(str(cache_path))
            return index

        hsig_index, hsig_metrics = build_index(build_hsig, cache_path)

    hnsw_index = None
    hnsw_metrics: dict[str, object] | None = None
    if "hnsw_postfilter" in args.methods:
        cache_path = args.index_cache_dir / "hnsw.index"
        def build_hnsw():
            index = HnswLib(metric="angular" if args.dist == "angular" else "euclidean", method_param={"M": args.M, "efConstruction": args.efConstruction})
            if cache_path.exists():
                index.loadIndex(base, str(cache_path))
                index.scalars = scalars
            else:
                index.fit(base, scalars)
                index.saveIndex(str(cache_path))
            return index

        hnsw_index, hnsw_metrics = build_index(build_hnsw, cache_path)

    plan_codes = {
        "hsig_hybrid": 0,
        "hsig_pre": 1,
        "hsig_post": 2,
        "hsig_cbo": 3,
    }

    num_workers = args.num_workers if args.num_workers > 0 else default_thread_count()
    num_workers = max(1, min(num_workers, default_thread_count()))
    print(f"Using {num_workers} query workers for UNIFY benchmark")

    for range_width in sorted(artifacts.query_ranges):
        gt = artifacts.groundtruth[range_width]
        mapped_ranges = compressed.query_ranges[range_width]
        empty_mask = compressed.empty_masks[range_width]
        valid_queries = int(np.count_nonzero(~empty_mask))

        if hsig_index is not None:
            for method_name in [m for m in args.methods if m.startswith("hsig_")]:
                for ef in args.ef_list:
                    for al in args.al_list:
                        hsig_index.set_query_arguments(
                            {
                                "ef": ef,
                                "al": al,
                                "search_strategy": plan_codes[method_name],
                                "target_recall": 0.9,
                                "low_range": args.low_range,
                                "high_range": args.high_range,
                            }
                        )
                        start = time.perf_counter()
                        ids = run_hybrid_queries(
                            hsig_index,
                            queries,
                            mapped_ranges,
                            empty_mask,
                            args.k,
                            num_workers,
                        )
                        elapsed = time.perf_counter() - start
                        query_qps = qps(elapsed, valid_queries)
                        rows.append(
                            {
                                "baseline": method_name,
                                "range": range_width,
                                "ef": ef,
                                "al": al,
                                **(hsig_metrics or {}),
                                "recall": f"{compute_recall(ids, gt, args.k):.6f}",
                                "avg_ms": f"{milliseconds_per_query(elapsed, valid_queries):.6f}",
                                "qps": f"{query_qps:.6f}",
                                "kpqs": f"{kpqs(elapsed, valid_queries):.6f}",
                                "valid_queries": valid_queries,
                            }
                        )

        if hnsw_index is not None:
            for ef in args.ef_list:
                hnsw_index.set_query_arguments({"ef": ef})
                start = time.perf_counter()
                ids = run_hybrid_queries(
                    hnsw_index,
                    queries,
                    mapped_ranges,
                    empty_mask,
                    args.k,
                    num_workers,
                )
                elapsed = time.perf_counter() - start
                query_qps = qps(elapsed, valid_queries)
                rows.append(
                    {
                        "baseline": "hnsw_postfilter",
                        "range": range_width,
                        "ef": ef,
                        "al": 0,
                        **(hnsw_metrics or {}),
                        "recall": f"{compute_recall(ids, gt, args.k):.6f}",
                        "avg_ms": f"{milliseconds_per_query(elapsed, valid_queries):.6f}",
                        "qps": f"{query_qps:.6f}",
                        "kpqs": f"{kpqs(elapsed, valid_queries):.6f}",
                        "valid_queries": valid_queries,
                    }
                )

    write_tsv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
