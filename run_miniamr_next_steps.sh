#!/usr/bin/env bash
set -euo pipefail

# miniAMR immediate next-step runner template.
#
# Default behavior is a dry run. Configure paths/runtime below, inspect the
# generated command files, then run with DRY_RUN=0 when your site setup is ready.

# ----------------------------- user config ---------------------------------

DRY_RUN="${DRY_RUN:-1}"

MINIAMR_ROOT="${MINIAMR_ROOT:-/Users/avilla/Downloads/softwares/miniAMR}"
MINIAMR_EXE="${MINIAMR_EXE:-${MINIAMR_ROOT}/ref/miniAMR.x}"
ANALYSIS_PY="${ANALYSIS_PY:-${MINIAMR_ROOT}/miniAMR_analysis/mpip_analysisv2.py}"

OUT_ROOT="${OUT_ROOT:-${PWD}/miniAMR_next_steps_$(date +%Y%m%d_%H%M%S)}"

# Runtime selection: srun, mpirun, or local.
# local is only useful for one-rank smoke tests.
MPI_MODE="${MPI_MODE:-srun}"
TASKS_PER_NODE="${TASKS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
CPU_BIND="${CPU_BIND:-cores}"

USE_MPIP="${USE_MPIP:-1}"
MPIP_LIB="${MPIP_LIB:-/path/to/libmpiP.so}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

RANKS_LIST="${RANKS_LIST:-2 4 8 16}"
REPS="${REPS:-3}"

# Matrix A: clean apples-to-apples scaling sweep.
CLEAN_TIMESTEPS="${CLEAN_TIMESTEPS:-50}"
CLEAN_LB_OPT="${CLEAN_LB_OPT:-0}"
CLEAN_CHECKSUM_FREQ_LIST="${CLEAN_CHECKSUM_FREQ_LIST:-1}"

# Matrix B: load-balance-enabled sweep, separated from Matrix A.
LB_TIMESTEPS="${LB_TIMESTEPS:-100}"
LB_LB_OPT="${LB_LB_OPT:-1}"
LB_CHECKSUM_FREQ_LIST="${LB_CHECKSUM_FREQ_LIST:-1}"

# miniAMR workload shape matching the previous communication-oriented runs.
MAX_BLOCKS="${MAX_BLOCKS:-24000}"
NUM_REFINE="${NUM_REFINE:-5}"
BLOCK_NX="${BLOCK_NX:-8}"
BLOCK_NY="${BLOCK_NY:-8}"
BLOCK_NZ="${BLOCK_NZ:-8}"
NUM_VARS="${NUM_VARS:-80}"
COMM_VARS="${COMM_VARS:-4}"
STAGES_PER_TS="${STAGES_PER_TS:-40}"
REFINE_FREQ="${REFINE_FREQ:-2}"
INBALANCE="${INBALANCE:-0}"
REPORT_PERF="${REPORT_PERF:-7}"

# ----------------------------------------------------------------------------

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

rank_grid() {
  case "$1" in
    1)  printf '1 1 1\n' ;;
    2)  printf '2 1 1\n' ;;
    4)  printf '2 2 1\n' ;;
    8)  printf '2 2 2\n' ;;
    16) printf '4 2 2\n' ;;
    27) printf '3 3 3\n' ;;
    32) printf '4 4 2\n' ;;
    64) printf '4 4 4\n' ;;
    *)
      printf 'No rank-grid mapping for %s. Add it in rank_grid().\n' "$1" >&2
      return 1
      ;;
  esac
}

require_runtime_config() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  if [[ ! -x "${MINIAMR_EXE}" ]]; then
    printf 'MINIAMR_EXE is not executable: %s\n' "${MINIAMR_EXE}" >&2
    exit 1
  fi

  if [[ "${USE_MPIP}" -eq 1 && ! -f "${MPIP_LIB}" ]]; then
    printf 'USE_MPIP=1 but MPIP_LIB does not exist: %s\n' "${MPIP_LIB}" >&2
    exit 1
  fi
}

make_miniamr_args() {
  local ranks="$1"
  local timesteps="$2"
  local lb_opt="$3"
  local checksum_freq="$4"
  local npx npy npz

  read -r npx npy npz < <(rank_grid "${ranks}")

  MINIAMR_ARGS=(
    --num_refine "${NUM_REFINE}"
    --max_blocks "${MAX_BLOCKS}"
    --init_x 1 --init_y 1 --init_z 1
    --npx "${npx}" --npy "${npy}" --npz "${npz}"
    --nx "${BLOCK_NX}" --ny "${BLOCK_NY}" --nz "${BLOCK_NZ}"
    --num_vars "${NUM_VARS}"
    --comm_vars "${COMM_VARS}"
    --num_objects 2
    --object 0 1 0.25 0.50 0.50  0.010 0.0 0.0  0.150 0.150 0.150  0.0 0.0 0.0
    --object 0 1 0.75 0.50 0.50 -0.010 0.0 0.0  0.150 0.150 0.150  0.0 0.0 0.0
    --num_tsteps "${timesteps}"
    --stages_per_ts "${STAGES_PER_TS}"
    --checksum_freq "${checksum_freq}"
    --refine_freq "${REFINE_FREQ}"
    --lb_opt "${lb_opt}"
    --inbalance "${INBALANCE}"
    --report_perf "${REPORT_PERF}"
    --rcb
  )
}

build_mpi_command() {
  local case_name="$1"
  local ranks="$2"
  local nodes="$3"
  shift 3
  local app_args=("$@")

  case "${MPI_MODE}" in
    srun)
      MPI_CMD=(srun
        --nodes="${nodes}"
        --ntasks="${ranks}"
        --ntasks-per-node="${TASKS_PER_NODE}"
        --cpus-per-task="${CPUS_PER_TASK}"
      )
      if [[ -n "${CPU_BIND}" ]]; then
        MPI_CMD+=(--cpu-bind="${CPU_BIND}")
      fi
      if [[ "${USE_MPIP}" -eq 1 ]]; then
        MPI_CMD+=(--export="ALL,LD_PRELOAD=${MPIP_LIB},MPIP=-k ${case_name}")
      fi
      MPI_CMD+=("${MINIAMR_EXE}" "${app_args[@]}")
      ;;
    mpirun)
      MPI_CMD=(mpirun -np "${ranks}" "${MINIAMR_EXE}" "${app_args[@]}")
      if [[ "${USE_MPIP}" -eq 1 ]]; then
        MPI_CMD=(env LD_PRELOAD="${MPIP_LIB}" MPIP="-k ${case_name}" "${MPI_CMD[@]}")
      fi
      ;;
    local)
      if [[ "${ranks}" -ne 1 ]]; then
        printf 'MPI_MODE=local only supports ranks=1; got %s\n' "${ranks}" >&2
        return 1
      fi
      MPI_CMD=("${MINIAMR_EXE}" "${app_args[@]}")
      ;;
    *)
      printf 'Unknown MPI_MODE=%s. Use srun, mpirun, or local.\n' "${MPI_MODE}" >&2
      return 1
      ;;
  esac
}

run_case() {
  local matrix_name="$1"
  local rep_id="$2"
  local ranks="$3"
  local timesteps="$4"
  local lb_opt="$5"
  local checksum_freq="$6"
  local nodes="${ranks}"
  local case_name="miniAMR_${matrix_name}_${rep_id}_nodes_${nodes}_ranks_${ranks}_ck${checksum_freq}"
  local case_dir="${OUT_ROOT}/${matrix_name}/${rep_id}/miniAMR_nodes_${nodes}_ranks_${ranks}_ck${checksum_freq}"
  local mpip_dir="${case_dir}/mpiP"
  local stdout="${case_dir}/${case_name}.out"
  local stderr="${case_dir}/${case_name}.err"
  local command_file="${case_dir}/command.sh"
  local cwd="${case_dir}"

  if [[ "${USE_MPIP}" -eq 1 ]]; then
    cwd="${mpip_dir}"
  fi

  mkdir -p "${case_dir}" "${mpip_dir}"

  make_miniamr_args "${ranks}" "${timesteps}" "${lb_opt}" "${checksum_freq}"
  build_mpi_command "${case_name}" "${ranks}" "${nodes}" "${MINIAMR_ARGS[@]}"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "${cwd}"
    printf 'exec '
    printf '%q ' "${MPI_CMD[@]}"
    printf '> %q 2> %q\n' "${stdout}" "${stderr}"
  } > "${command_file}"
  chmod +x "${command_file}"

  log "Prepared ${case_name}"
  log "  command: ${command_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  (
    cd "${cwd}"
    "${MPI_CMD[@]}" > "${stdout}" 2> "${stderr}"
  )
}

run_matrix() {
  local matrix_name="$1"
  local timesteps="$2"
  local lb_opt="$3"
  local checksum_freq_list="$4"
  local rep ranks checksum_freq

  log "Matrix ${matrix_name}: timesteps=${timesteps}, lb_opt=${lb_opt}, checksum_freq_list=${checksum_freq_list}"

  for ((rep=1; rep<=REPS; rep++)); do
    local rep_id
    rep_id="$(printf 'rep_%02d' "${rep}")"
    for checksum_freq in ${checksum_freq_list}; do
      for ranks in ${RANKS_LIST}; do
        run_case "${matrix_name}" "${rep_id}" "${ranks}" "${timesteps}" "${lb_opt}" "${checksum_freq}"
      done
    done
  done
}

analyze_profiles() {
  if [[ "${RUN_ANALYSIS}" -ne 1 || "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  if [[ ! -f "${ANALYSIS_PY}" ]]; then
    log "Skipping analysis; ANALYSIS_PY not found: ${ANALYSIS_PY}"
    return 0
  fi

  local rep_dir
  while IFS= read -r rep_dir; do
    if find "${rep_dir}" -name '*.mpiP' -print -quit | grep -q .; then
      log "Running mpiP analysis for ${rep_dir}"
      python3 "${ANALYSIS_PY}" "${rep_dir}" --outdir "${rep_dir}/analysis"
    fi
  done < <(find "${OUT_ROOT}" -type d -name 'rep_*' | sort)
}

main() {
  require_runtime_config
  mkdir -p "${OUT_ROOT}"

  log "Output root: ${OUT_ROOT}"
  log "DRY_RUN=${DRY_RUN}; set DRY_RUN=0 to launch jobs."

  run_matrix "clean_50_no_lb" "${CLEAN_TIMESTEPS}" "${CLEAN_LB_OPT}" "${CLEAN_CHECKSUM_FREQ_LIST}"
  run_matrix "lb_100_enabled" "${LB_TIMESTEPS}" "${LB_LB_OPT}" "${LB_CHECKSUM_FREQ_LIST}"

  analyze_profiles

  log "Done. Inspect ${OUT_ROOT}"
}

main "$@"
