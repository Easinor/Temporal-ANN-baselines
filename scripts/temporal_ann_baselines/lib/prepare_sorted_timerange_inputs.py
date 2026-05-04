from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    build_sorted_timerange_view,
    load_timerange_artifacts,
    load_vectors,
    write_dense_bin,
    write_i32_groundtruth,
    write_i32_ranges,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ANNlib timerange artifacts into a sorted float32 view for external baselines."
    )
    parser.add_argument("--dataset", required=True, help="ANNlib dataset spec, e.g. ./ANNdataset/BIGANN/base.1B.u8bin:u8bin")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--ranges", type=int, nargs="*")
    args = parser.parse_args()

    base = load_vectors(args.dataset, max_points=args.max_points if args.max_points > 0 else None)
    artifacts = load_timerange_artifacts(args.artifacts_dir, args.ranges)
    view = build_sorted_timerange_view(
        base=base,
        point_timestamps=artifacts.point_timestamps,
        query_ids=artifacts.query_ids,
        query_ranges=artifacts.query_ranges,
        groundtruth=artifacts.groundtruth,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dense_bin(args.output_dir / "base_sorted.bin", view.sorted_base)
    write_dense_bin(args.output_dir / "queries.bin", view.queries)
    np.asarray(view.sorted_timestamps, dtype=np.uint32).tofile(args.output_dir / "sorted_timestamps.u32")
    np.asarray(view.permutation, dtype=np.int32).tofile(args.output_dir / "permutation.i32")
    np.asarray(view.inverse_permutation, dtype=np.int32).tofile(args.output_dir / "inverse_permutation.i32")

    for width, ranges in sorted(view.query_ranges.items()):
        write_i32_ranges(args.output_dir / f"range{width}.ranges.bin", ranges)
        write_i32_groundtruth(args.output_dir / f"range{width}.gt.bin", view.groundtruth[width])
        np.asarray(view.empty_masks[width], dtype=np.uint8).tofile(args.output_dir / f"range{width}.empty.u8")

    print(f"wrote sorted baseline inputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
