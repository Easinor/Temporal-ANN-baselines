# datasets.sh — Dataset and baseline-generated graph path definitions.
# Source this file from other scripts. Caller must set repo_root before sourcing.

anndataset_root="${ANNDATASET_ROOT:-${repo_root}/ANNdataset}"

# ---------------------------------------------------------------------------
# ANN base graphs (raw dataset files)
# Format per entry: "tag|path:format|type|dist|max_points"
#   tag        — short identifier used for output dirs/files
#   path:fmt   — ANNlib dataset spec (absolute path to .u8bin/.fbin + format token)
#   type       — uint8 | int8 | float
#   dist       — L2 | angular
#   max_points — number of base vectors to load
# ---------------------------------------------------------------------------
DATASETS=(
  "bigann1M|${anndataset_root}/sift/bigann.base.1B.u8bin:u8bin|uint8|L2|1000000"
  "deep1M_yandex|${anndataset_root}/Yandex-DEEP.sampled.350M.fbin:fbin|float|L2|1000000"
)

# ---------------------------------------------------------------------------
# Common output roots
# ---------------------------------------------------------------------------
artifact_root="${ARTIFACT_ROOT:-${repo_root}/artifacts/point_timerange_ann}"
sorted_root="${SORTED_ROOT:-${repo_root}/artifacts/sorted_timerange}"
result_root="${RESULT_ROOT:-${repo_root}/results/temporal_baselines}"
log_root="${LOG_ROOT:-${result_root}/logs}"

# ---------------------------------------------------------------------------
# Paths to baseline-generated graphs / index files (parameterised by tag)
#
# Usage: var="$(rf_cache_dir bigann1M)"
# ---------------------------------------------------------------------------

# RangeFilteredANN — Vamana tree cache directory
rf_cache_dir()    { echo "${repo_root}/cache/rangefilteredann/$1/tree"; }

# UNIFY / HSIG — HSIG index file and HNSW index file
unify_hsig_index()  { echo "${repo_root}/cache/unify/$1/hsig.index"; }
unify_hnsw_index()  { echo "${repo_root}/cache/unify/$1/hnsw.index"; }

# DSG (Dynamic Segment Graph) — static index binary
dsg_index_path()    { echo "${repo_root}/Dynamic-Range-Filtering-ANNS/index/static/$1/dsg.index"; }

# iRangeGraph — index binary
irg_index_path()    { echo "${repo_root}/iRangeGraph/index/$1/irangegraph.bin"; }

# SeRF — builds in memory; no persistent index file

# ---------------------------------------------------------------------------
# Common query parameters (override via environment variables)
# ---------------------------------------------------------------------------
num_queries="${NUM_QUERIES:-1000}"
k="${K:-10}"
ranges_str="${RANGES_STR:-100 1000 10000 100000 1000000}"
ranges_csv="${RANGES_CSV:-100,1000,10000,100000,1000000}"
