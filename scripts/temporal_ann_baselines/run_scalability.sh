#!/usr/bin/env bash
# Scalability tests: vary thread count for SeRF, iRangeGraph, and RangeFilteredANN on bigann1M.
# Results written to:
#   ${result_root}/scalability/serf/bigann1M_threads{N}.tsv
#   ${result_root}/scalability/irangegraph/bigann1M_threads{N}.tsv
#   ${result_root}/scalability/rangefilteredann/bigann1M_threads{N}.tsv
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${script_dir}/datasets.sh"


thread_list="${THREAD_LIST:-1 2 4 8 16 28 56 112 224}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue
  artifacts_dir="${artifact_root}/${tag}"
  sorted_dir="${sorted_root}/${tag}"
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue

  echo "[${tag}] generating artifacts"
  DATASET="${dataset}" \
  DATASET_TAG="${tag}" \
  TYPE="${type}" \
  DIST="${dist}" \
  MAX_POINTS="${max_points}" \
  NUM_QUERIES="${num_queries}" \
  K="${k}" \
  RANGES_STR="${ranges_str}" \
  ARTIFACT_ROOT="${artifact_root}" \
  ANNLIB_TEST_DIR="${annlib_test_dir}" \
  "${script_dir}/lib/generate_annlib_timerange_artifacts.sh"

  echo "[${tag}] preparing sorted inputs"
  "${python_bin}" "${script_dir}/lib/prepare_sorted_timerange_inputs.py" \
    --dataset "${dataset}" \
    --artifacts-dir "${artifact_root}/${tag}" \
    --output-dir "${sorted_root}/${tag}" \
    --max-points "${max_points}"

  # ---------------------------------------------------------------------------
  # SeRF scalability
  # ---------------------------------------------------------------------------
  serf_result_root="${result_root}/scalability/serf"
  mkdir -p "${serf_result_root}"

  echo "=== SeRF scalability ==="
  for threads in ${thread_list}; do
    echo "  threads=${threads}"
    OMP_NUM_THREADS="${threads}" \
    "${repo_root}/SeRF/build/benchmark/serf_annlib" \
      --data-path "${sorted_dir}/base_sorted.bin" \
      --query-path "${sorted_dir}/queries.bin" \
      --artifacts-dir "${sorted_dir}" \
      --output "${serf_result_root}/${tag}_threads${threads}.tsv" \
      --data-size "${max_points}" \
      --k "${k}" \
      --index-k-list 16 \
      --ef-con-list 80 \
      --ef-max-list 300 \
      --ef-search-list 160 \
      --ranges "${ranges_csv}"
  done
  echo "SeRF scalability done."

  # ---------------------------------------------------------------------------
  # iRangeGraph scalability
  # ---------------------------------------------------------------------------
  irg_result_root="${result_root}/scalability/irangegraph"
  mkdir -p "${irg_result_root}"

  echo "=== iRangeGraph scalability ==="
  for threads in ${thread_list}; do
    irg_cache_dir="${cache_root}/irangegraph/scalability/threads${threads}"
    index_path="${irg_cache_dir}/irangegraph.bin"
    time_file="${irg_cache_dir}/build_seconds.txt"
    rm -rf "${irg_cache_dir}"
    mkdir -p "${irg_cache_dir}"

    echo "  threads=${threads}"
    /usr/bin/time -f "%e" -o "${time_file}" \
      "${repo_root}/iRangeGraph/build/tests/buildindex" \
        --data_path "${sorted_dir}/base_sorted.bin" \
        --index_file "${index_path}" \
        --M 16 \
        --ef_construction 80 \
        --threads "${threads}"

    build_seconds="$(<"${time_file}")"
    index_bytes="$(stat -c%s "${index_path}")"

    OMP_NUM_THREADS="${threads}" \
    "${repo_root}/iRangeGraph/build/tests/search_annlib" \
      --data-path "${sorted_dir}/base_sorted.bin" \
      --query-path "${sorted_dir}/queries.bin" \
      --artifacts-dir "${sorted_dir}" \
      --index-file "${index_path}" \
      --output "${irg_result_root}/${tag}_threads${threads}.tsv" \
      --M 16 \
      --k "${k}" \
      --build-seconds "${build_seconds}" \
      --index-bytes "${index_bytes}" \
      --space-usage-source "index_file_bytes" \
      --ef-list 160 \
      --ranges "${ranges_csv}"
  done
  echo "iRangeGraph scalability done."

  # ---------------------------------------------------------------------------
  # RangeFilteredANN scalability
  # ---------------------------------------------------------------------------
  rf_result_root="${result_root}/scalability/rangefilteredann"
  mkdir -p "${rf_result_root}"

  echo "=== RangeFilteredANN scalability ==="
  for threads in ${thread_list}; do
    cache_dir="${cache_root}/rangefilteredann/scalability/threads${threads}"
    rm -rf "${cache_dir}"

    echo "  threads=${threads}"
    PARLAY_NUM_THREADS="${threads}" \
    OMP_NUM_THREADS="${threads}" \
    "${python_bin}" "${script_dir}/lib/run_rangefilteredann.py" \
      --dataset "${dataset}" \
      --artifacts-dir "${artifacts_dir}" \
      --dist "${dist}" \
      --max-points "${max_points}" \
      --ranges ${ranges_str} \
      --methods vamana_tree \
      --beam-sizes 160 \
      --final-beam-multiplies 1 \
      --alpha 1 --R 32 --L 80 \
      --threads "${threads}" \
      --index-cache-dir "${cache_dir}" \
      --output "${rf_result_root}/${tag}_threads${threads}.tsv"
  done
  echo "RangeFilteredANN scalability done."

done