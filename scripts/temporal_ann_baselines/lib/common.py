from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ANNLIB_TEST_DIR = Path("/home/zwan/ANNlib/test")
UINT32_MAX = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    format: str


@dataclass(frozen=True)
class TimerangeArtifacts:
    point_timestamps: np.ndarray
    query_ids: np.ndarray
    query_ranges: dict[int, np.ndarray]
    groundtruth: dict[int, np.ndarray]


@dataclass(frozen=True)
class CompressedTimerangeView:
    scalars: np.ndarray
    query_ranges: dict[int, np.ndarray]
    empty_masks: dict[int, np.ndarray]


@dataclass(frozen=True)
class SortedTimerangeView:
    sorted_base: np.ndarray
    sorted_timestamps: np.ndarray
    queries: np.ndarray
    query_ranges: dict[int, np.ndarray]
    groundtruth: dict[int, np.ndarray]
    empty_masks: dict[int, np.ndarray]
    permutation: np.ndarray
    inverse_permutation: np.ndarray


def parse_dataset_spec(spec: str, annlib_test_dir: Path = ANNLIB_TEST_DIR) -> DatasetSpec:
    if ":" not in spec:
        raise ValueError(f"dataset spec must look like <path>:<format>, got: {spec}")
    path_str, fmt = spec.rsplit(":", 1)
    path = Path(path_str)
    if not path.is_absolute():
        candidates = [Path.cwd() / path, annlib_test_dir / path]
        for candidate in candidates:
            if candidate.exists():
                path = candidate.resolve()
                break
        else:
            path = (annlib_test_dir / path).resolve()
    return DatasetSpec(path=path, format=fmt)


def _read_annlib_bin(path: Path, dtype: np.dtype, max_points: int | None = None) -> np.ndarray:
    with path.open("rb") as fh:
        header = np.fromfile(fh, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"invalid ANNlib binary header: {path}")
        total_points, dim = map(int, header)
        if max_points and max_points > 0:
            total_points = min(total_points, int(max_points))
        data = np.fromfile(fh, dtype=dtype, count=total_points * dim)
    if data.size != total_points * dim:
        raise ValueError(f"unexpected vector payload length in {path}")
    return data.reshape(total_points, dim)


def load_vectors(spec: str, max_points: int | None = None) -> np.ndarray:
    dataset = parse_dataset_spec(spec)
    fmt = dataset.format.lower()
    if fmt == "fbin":
        return _read_annlib_bin(dataset.path, np.float32, max_points)
    if fmt == "u8bin":
        return _read_annlib_bin(dataset.path, np.uint8, max_points)
    if fmt == "i8bin":
        return _read_annlib_bin(dataset.path, np.int8, max_points)
    raise ValueError(f"unsupported vector format: {dataset.format}")


def write_dense_bin(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.ascontiguousarray(array)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D array, got ndim={matrix.ndim}")
    header = np.asarray([matrix.shape[0], matrix.shape[1]], dtype=np.uint32)
    with path.open("wb") as fh:
        header.tofile(fh)
        matrix.tofile(fh)


def write_i32_ranges(path: str | Path, ranges: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.ascontiguousarray(ranges.astype(np.int32, copy=False))
    if packed.ndim != 2 or packed.shape[1] != 2:
        raise ValueError(f"expected a (n, 2) range array, got shape={packed.shape}")
    header = np.asarray([packed.shape[0]], dtype=np.uint32)
    with path.open("wb") as fh:
        header.tofile(fh)
        packed.tofile(fh)


def write_i32_groundtruth(path: str | Path, groundtruth: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.ascontiguousarray(groundtruth.astype(np.int32, copy=False))
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D groundtruth array, got ndim={matrix.ndim}")
    header = np.asarray([matrix.shape[0], matrix.shape[1]], dtype=np.uint32)
    with path.open("wb") as fh:
        header.tofile(fh)
        matrix.tofile(fh)


def read_u32_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as fh:
        count = np.fromfile(fh, dtype=np.uint32, count=1)
        if count.size != 1:
            raise ValueError(f"invalid u32 array header: {path}")
        values = np.fromfile(fh, dtype=np.uint32, count=int(count[0]))
    if values.size != int(count[0]):
        raise ValueError(f"unexpected payload length in {path}")
    return values


def read_query_ranges(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as fh:
        count = np.fromfile(fh, dtype=np.uint32, count=1)
        if count.size != 1:
            raise ValueError(f"invalid query range header: {path}")
        values = np.fromfile(fh, dtype=np.uint32, count=int(count[0]) * 2)
    if values.size != int(count[0]) * 2:
        raise ValueError(f"unexpected query range payload length in {path}")
    return values.reshape(int(count[0]), 2)


def read_timerange_groundtruth(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as fh:
        header = np.fromfile(fh, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"invalid timerange gt header: {path}")
        num_queries, k = map(int, header)
        values = np.fromfile(fh, dtype=np.uint32, count=num_queries * k)
    if values.size != num_queries * k:
        raise ValueError(f"unexpected timerange gt payload length in {path}")
    return values.reshape(num_queries, k)


def infer_ranges(artifacts_dir: str | Path) -> list[int]:
    artifacts_dir = Path(artifacts_dir)
    pattern = re.compile(r"range(\d+)\.query_ranges\.bin$")
    ranges: list[int] = []
    for path in artifacts_dir.glob("range*.query_ranges.bin"):
        match = pattern.match(path.name)
        if match:
            ranges.append(int(match.group(1)))
    return sorted(set(ranges))


def load_timerange_artifacts(
    artifacts_dir: str | Path,
    ranges: Iterable[int] | None = None,
) -> TimerangeArtifacts:
    artifacts_dir = Path(artifacts_dir)
    resolved_ranges = list(ranges) if ranges is not None else infer_ranges(artifacts_dir)
    if not resolved_ranges:
        raise ValueError(f"no range*.query_ranges.bin files found in {artifacts_dir}")

    point_timestamps = read_u32_array(artifacts_dir / "point_timestamps.bin")
    query_ids = read_u32_array(artifacts_dir / "query_ids.bin")

    query_ranges = {
        width: read_query_ranges(artifacts_dir / f"range{width}.query_ranges.bin")
        for width in resolved_ranges
    }
    groundtruth = {
        width: read_timerange_groundtruth(artifacts_dir / f"range{width}.gt.ibin")
        for width in resolved_ranges
    }
    return TimerangeArtifacts(
        point_timestamps=point_timestamps,
        query_ids=query_ids,
        query_ranges=query_ranges,
        groundtruth=groundtruth,
    )


def normalize_angular(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def compress_scalars_and_ranges(
    point_timestamps: np.ndarray,
    query_ranges: dict[int, np.ndarray],
) -> CompressedTimerangeView:
    unique_values = np.unique(point_timestamps.astype(np.uint32, copy=False))
    if unique_values.size == 0:
        raise ValueError("cannot compress empty timestamp array")

    scalars = np.searchsorted(unique_values, point_timestamps, side="left").astype(np.int64)

    mapped_ranges: dict[int, np.ndarray] = {}
    empty_masks: dict[int, np.ndarray] = {}
    max_rank = unique_values.size - 1

    for width, ranges in query_ranges.items():
        begins = np.searchsorted(unique_values, ranges[:, 0], side="left")
        ends = np.searchsorted(unique_values, ranges[:, 1], side="right") - 1
        empty = begins > ends
        begins = np.clip(begins, 0, max_rank)
        ends = np.clip(ends, 0, max_rank)
        mapped = np.stack([begins, ends], axis=1).astype(np.int64)
        mapped_ranges[width] = mapped
        empty_masks[width] = empty

    return CompressedTimerangeView(
        scalars=scalars,
        query_ranges=mapped_ranges,
        empty_masks=empty_masks,
    )


def build_sorted_timerange_view(
    base: np.ndarray,
    point_timestamps: np.ndarray,
    query_ids: np.ndarray,
    query_ranges: dict[int, np.ndarray],
    groundtruth: dict[int, np.ndarray],
) -> SortedTimerangeView:
    if point_timestamps.shape[0] < base.shape[0]:
        raise ValueError("timestamp count is smaller than base size")

    timestamps = np.asarray(point_timestamps[: base.shape[0]], dtype=np.uint32)
    permutation = np.argsort(timestamps, kind="stable")
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(permutation.shape[0], dtype=permutation.dtype)

    sorted_base = np.ascontiguousarray(base[permutation].astype(np.float32, copy=False))
    sorted_timestamps = timestamps[permutation]
    queries = np.ascontiguousarray(base[query_ids.astype(np.int64)].astype(np.float32, copy=False))

    mapped_ranges: dict[int, np.ndarray] = {}
    mapped_groundtruth: dict[int, np.ndarray] = {}
    empty_masks: dict[int, np.ndarray] = {}

    for width, ranges in query_ranges.items():
        begins = np.searchsorted(sorted_timestamps, ranges[:, 0], side="left")
        ends = np.searchsorted(sorted_timestamps, ranges[:, 1], side="right") - 1
        empty = begins > ends
        mapped_ranges[width] = np.stack([begins, ends], axis=1).astype(np.int32)
        empty_masks[width] = empty

        gt = np.asarray(groundtruth[width], dtype=np.uint32)
        mapped_gt = np.full(gt.shape, -1, dtype=np.int32)
        valid = gt != UINT32_MAX
        mapped_gt[valid] = inverse_permutation[gt[valid]].astype(np.int32, copy=False)
        mapped_groundtruth[width] = mapped_gt

    return SortedTimerangeView(
        sorted_base=sorted_base,
        sorted_timestamps=sorted_timestamps,
        queries=queries,
        query_ranges=mapped_ranges,
        groundtruth=mapped_groundtruth,
        empty_masks=empty_masks,
        permutation=permutation.astype(np.int32, copy=False),
        inverse_permutation=inverse_permutation.astype(np.int32, copy=False),
    )


def build_queries(base: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    if query_ids.size == 0:
        return np.empty((0, base.shape[1]), dtype=base.dtype)
    max_id = int(query_ids.max())
    if max_id >= base.shape[0]:
        raise ValueError(f"query id {max_id} exceeds base size {base.shape[0]}")
    return np.ascontiguousarray(base[query_ids.astype(np.int64)])


def dtype_token(array: np.ndarray) -> str:
    if array.dtype == np.uint8:
        return "uint8"
    if array.dtype == np.int8:
        return "int8"
    if np.issubdtype(array.dtype, np.floating):
        return "float"
    raise ValueError(f"unsupported dtype: {array.dtype}")


def compute_recall(results: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    total_hits = 0
    total_valid = 0
    limit = min(k, results.shape[1], groundtruth.shape[1])
    for row_pred, row_gt in zip(results, groundtruth, strict=True):
        gt = {int(x) for x in row_gt[:limit] if int(x) != UINT32_MAX}
        if not gt:
            continue
        pred = {int(x) for x in row_pred[:limit] if int(x) >= 0}
        total_hits += len(gt & pred)
        total_valid += len(gt)
    return 0.0 if total_valid == 0 else total_hits / total_valid


def write_tsv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def ensure_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def ensure_c_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array)


def milliseconds_per_query(total_seconds: float, num_queries: int) -> float:
    return 0.0 if num_queries == 0 else 1000.0 * total_seconds / num_queries


def qps(total_seconds: float, num_queries: int) -> float:
    return 0.0 if total_seconds <= 0 else num_queries / total_seconds


def kpqs(total_seconds: float, num_queries: int) -> float:
    return qps(total_seconds, num_queries) / 1000.0


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return int(root.stat().st_size)

    total = 0
    for child in root.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def file_size_bytes(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return 0
    return int(file_path.stat().st_size)


def process_rss_bytes() -> int:
    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return 0
    contents = statm_path.read_text().split()
    if len(contents) < 2:
        return 0
    return int(contents[1]) * int(os.sysconf("SC_PAGE_SIZE"))
