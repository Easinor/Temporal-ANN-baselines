# Temporal ANN Baseline Wrappers

These wrappers live under `Temporal-ANN-baselines` and consume ANNlib timerange artifacts directly.

## What They Cover

- `generate_annlib_timerange_artifacts.sh`
  Reuses `ANNlib/test/prepare_timerange_dataset.sh` to generate:
  `point_timestamps.bin`, `query_ids.bin`, `range*.query_ranges.bin`, and `range*.gt.ibin`.
- `run_rangefilteredann.py`
  Runs local `RangeFilteredANN` baselines on ANNlib timerange artifacts.
- `run_unify.py`
  Runs local `UNIFY` / `HSIG` and plain HNSW postfilter baselines on ANNlib timerange artifacts.
- `run_all_local_baselines.sh`
  Convenience wrapper that generates artifacts first and then launches both baseline families.

These wrappers target the `point timestamp + query time range` setting used by
`ANNlib/test/test_timerange_parallel_merge.cpp`.

They do not directly cover the `lifespan + point-in-time snapshot` setting used by
`ANNlib/test/test_snapshots_with_deletion.cpp`.

## Build The Baseline Bindings

```bash
./scripts/temporal_ann_baselines/build_python_baselines.sh
```

## Generate ANNlib Artifacts

```bash
DATASET=./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin \
DATASET_TAG=deep1M_yandex \
TYPE=float \
DIST=L2 \
MAX_POINTS=1000000 \
NUM_QUERIES=1000 \
RANGES_STR="100 1000 10000 100000" \
./scripts/temporal_ann_baselines/generate_annlib_timerange_artifacts.sh
```

## Run RangeFilteredANN

```bash
python3 ./scripts/temporal_ann_baselines/run_rangefilteredann.py \
  --dataset ./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin \
  --artifacts-dir ./artifacts/point_timerange_ann/deep1M_yandex \
  --dist L2 \
  --max-points 1000000
```

## Run UNIFY / HSIG

```bash
python3 ./scripts/temporal_ann_baselines/run_unify.py \
  --dataset ./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin \
  --artifacts-dir ./artifacts/point_timerange_ann/deep1M_yandex \
  --dist L2 \
  --max-points 1000000
```

## Run Both Families

```bash
DATASET=./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin \
DATASET_TAG=deep1M_yandex \
TYPE=float \
DIST=L2 \
MAX_POINTS=1000000 \
NUM_QUERIES=1000 \
RANGES_STR="100 1000 10000 100000" \
./scripts/temporal_ann_baselines/run_all_local_baselines.sh
```

## Notes

- The wrappers rank-compress timestamps before feeding them into the baselines.
  This avoids scalar precision problems and preserves exact range membership.
- For `angular` datasets, the wrappers L2-normalize vectors before running the baselines.
  This keeps the baseline metric aligned with ANNlib's cosine-style angular distance.
