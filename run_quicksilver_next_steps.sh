#!/usr/bin/env bash
set -euo pipefail

# Quicksilver immediate next-step runner template.
#
# Default behavior is a dry run. Configure paths/runtime below, inspect the
# generated inputs and command files, then run with DRY_RUN=0 when ready.

# ----------------------------- user config ---------------------------------

DRY_RUN="${DRY_RUN:-1}"

QS_ROOT="${QS_ROOT:-/Users/avilla/Downloads/softwares/Quicksilver}"
QS_EXE_BLOCKING="${QS_EXE_BLOCKING:-${QS_ROOT}/src/qs}"
QS_EXE_ASYNC="${QS_EXE_ASYNC:-${QS_ROOT}/src/qs_async}"
ANALYSIS_PY="${ANALYSIS_PY:-${QS_ROOT}/mpip_scale.py}"

OUT_ROOT="${OUT_ROOT:-${PWD}/quicksilver_next_steps_$(date +%Y%m%d_%H%M%S)}"

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

# Experiments from the immediate plan.
RUN_BLOCKING_CT0="${RUN_BLOCKING_CT0:-1}"
RUN_ASYNC_CT0="${RUN_ASYNC_CT0:-1}"

# Optional: reproduce the high-stress timer-enabled case alongside controls.
RUN_BLOCKING_CT1="${RUN_BLOCKING_CT1:-0}"
RUN_ASYNC_CT1="${RUN_ASYNC_CT1:-0}"

# Build async binary only if requested. Otherwise point QS_EXE_ASYNC at your
# prebuilt -DHAVE_ASYNC_MPI executable.
BUILD_ASYNC="${BUILD_ASYNC:-0}"
ASYNC_CPPFLAGS="${ASYNC_CPPFLAGS:--DHAVE_MPI -DHAVE_ASYNC_MPI}"
QS_CXXFLAGS="${QS_CXXFLAGS:--std=c++11 -O2}"
QS_LDFLAGS="${QS_LDFLAGS:-}"

# Quicksilver communication-stress workload knobs.
PARTICLES_PER_RANK="${PARTICLES_PER_RANK:-1000000}"
NUM_STEPS="${NUM_STEPS:-100}"
LOAD_BALANCE="${LOAD_BALANCE:-0}"

NX="${NX:-10}"
NY="${NY:-10}"
NZ="${NZ:-10}"
LX="${LX:-100}"
LY="${LY:-100}"
LZ="${LZ:-100}"

DT="${DT:-1e-08}"
FMAX="${FMAX:-0.1}"
TOTAL_XS="${TOTAL_XS:-0.01}"
SCATTER_RATIO="${SCATTER_RATIO:-1.0}"
ABSORB_RATIO="${ABSORB_RATIO:-0.1}"
FISSION_RATIO="${FISSION_RATIO:-0.1}"
NUBAR="${NUBAR:-2.4}"
N_ISO="${N_ISO:-10}"
N_REACT="${N_REACT:-9}"
SOURCE_RATE="${SOURCE_RATE:-1e+10}"
EMIN="${EMIN:-1e-09}"
EMAX="${EMAX:-20}"
N_GROUPS="${N_GROUPS:-230}"
SEED="${SEED:-1029384756}"

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
    32) printf '4 4 2\n' ;;
    64) printf '4 4 4\n' ;;
    *)
      printf 'No domain-grid mapping for %s. Add it in rank_grid().\n' "$1" >&2
      return 1
      ;;
  esac
}

require_runtime_config() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  if [[ "${USE_MPIP}" -eq 1 && ! -f "${MPIP_LIB}" ]]; then
    printf 'USE_MPIP=1 but MPIP_LIB does not exist: %s\n' "${MPIP_LIB}" >&2
    exit 1
  fi
}

build_async_if_requested() {
  if [[ "${BUILD_ASYNC}" -ne 1 ]]; then
    return 0
  fi

  local src_dir="${QS_ROOT}/src"
  local build_cmd=(make -C "${src_dir}" CPPFLAGS="${ASYNC_CPPFLAGS}" CXXFLAGS="${QS_CXXFLAGS}" LDFLAGS="${QS_LDFLAGS}")
  local copy_cmd=(cp "${src_dir}/qs" "${QS_EXE_ASYNC}")

  log "Async build requested."
  log "  build: $(quote_cmd "${build_cmd[@]}")"
  log "  copy:  $(quote_cmd "${copy_cmd[@]}")"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  "${build_cmd[@]}"
  "${copy_cmd[@]}"
}

generate_input() {
  local inp="$1"
  local ranks="$2"
  local cycle_timers="$3"
  local xdom ydom zdom nparticles

  read -r xdom ydom zdom < <(rank_grid "${ranks}")
  nparticles=$((ranks * PARTICLES_PER_RANK))

  cat > "${inp}" <<EOF
Simulation:
   dt: ${DT}
   fMax: ${FMAX}
   boundaryCondition: reflect
   loadBalance: ${LOAD_BALANCE}
   cycleTimers: ${cycle_timers}
   nParticles: ${nparticles}
   nSteps: ${NUM_STEPS}
   nx: ${NX}
   ny: ${NY}
   nz: ${NZ}
   lx: ${LX}
   ly: ${LY}
   lz: ${LZ}
   xDom: ${xdom}
   yDom: ${ydom}
   zDom: ${zdom}
   eMin: ${EMIN}
   eMax: ${EMAX}
   nGroups: ${N_GROUPS}
   seed: ${SEED}

Geometry:
   material: commStressMat
   shape: brick
   xMax: ${LX}
   xMin: 0
   yMax: ${LY}
   yMin: 0
   zMax: ${LZ}
   zMin: 0

Material:
   name: commStressMat
   nIsotopes: ${N_ISO}
   nReactions: ${N_REACT}
   sourceRate: ${SOURCE_RATE}
   totalCrossSection: ${TOTAL_XS}
   absorptionCrossSection: flat
   fissionCrossSection: flat
   scatteringCrossSection: flat
   absorptionCrossSectionRatio: ${ABSORB_RATIO}
   fissionCrossSectionRatio: ${FISSION_RATIO}
   scatteringCrossSectionRatio: ${SCATTER_RATIO}

CrossSection:
   name: flat
   A: 0
   B: 0
   C: 0
   D: 0
   E: 1
   nuBar: ${NUBAR}
EOF
}

build_mpi_command() {
  local case_name="$1"
  local exe="$2"
  local ranks="$3"
  local nodes="$4"
  local inp="$5"

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
      MPI_CMD+=("${exe}" -i "${inp}")
      ;;
    mpirun)
      MPI_CMD=(mpirun -np "${ranks}" "${exe}" -i "${inp}")
      if [[ "${USE_MPIP}" -eq 1 ]]; then
        MPI_CMD=(env LD_PRELOAD="${MPIP_LIB}" MPIP="-k ${case_name}" "${MPI_CMD[@]}")
      fi
      ;;
    local)
      if [[ "${ranks}" -ne 1 ]]; then
        printf 'MPI_MODE=local only supports ranks=1; got %s\n' "${ranks}" >&2
        return 1
      fi
      MPI_CMD=("${exe}" -i "${inp}")
      ;;
    *)
      printf 'Unknown MPI_MODE=%s. Use srun, mpirun, or local.\n' "${MPI_MODE}" >&2
      return 1
      ;;
  esac
}

run_case() {
  local exp_name="$1"
  local exe="$2"
  local cycle_timers="$3"
  local rep_id="$4"
  local ranks="$5"
  local nodes="${ranks}"
  local case_name="qs_${exp_name}_${rep_id}_nodes_${nodes}_ranks_${ranks}"
  local case_dir="${OUT_ROOT}/${exp_name}/${rep_id}/qs_nodes_${nodes}_ranks_${ranks}"
  local mpip_dir="${case_dir}/mpiP"
  local inp="${case_dir}/${case_name}.inp"
  local stdout="${case_dir}/${case_name}.out"
  local stderr="${case_dir}/${case_name}.err"
  local command_file="${case_dir}/command.sh"
  local cwd="${case_dir}"

  if [[ "${USE_MPIP}" -eq 1 ]]; then
    cwd="${mpip_dir}"
  fi

  mkdir -p "${case_dir}" "${mpip_dir}"
  generate_input "${inp}" "${ranks}" "${cycle_timers}"
  build_mpi_command "${case_name}" "${exe}" "${ranks}" "${nodes}" "${inp}"

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
  log "  input:   ${inp}"
  log "  command: ${command_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  if [[ ! -x "${exe}" ]]; then
    printf 'Quicksilver executable is not executable: %s\n' "${exe}" >&2
    exit 1
  fi

  (
    cd "${cwd}"
    "${MPI_CMD[@]}" > "${stdout}" 2> "${stderr}"
  )
}

run_experiment() {
  local exp_name="$1"
  local exe="$2"
  local cycle_timers="$3"
  local rep ranks

  log "Experiment ${exp_name}: cycleTimers=${cycle_timers}, exe=${exe}"

  for ((rep=1; rep<=REPS; rep++)); do
    local rep_id
    rep_id="$(printf 'rep_%02d' "${rep}")"
    for ranks in ${RANKS_LIST}; do
      run_case "${exp_name}" "${exe}" "${cycle_timers}" "${rep_id}" "${ranks}"
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

  build_async_if_requested

  if [[ "${RUN_BLOCKING_CT0}" -eq 1 ]]; then
    run_experiment "blocking_cycleTimers0" "${QS_EXE_BLOCKING}" 0
  fi
  if [[ "${RUN_BLOCKING_CT1}" -eq 1 ]]; then
    run_experiment "blocking_cycleTimers1" "${QS_EXE_BLOCKING}" 1
  fi
  if [[ "${RUN_ASYNC_CT0}" -eq 1 ]]; then
    run_experiment "async_cycleTimers0" "${QS_EXE_ASYNC}" 0
  fi
  if [[ "${RUN_ASYNC_CT1}" -eq 1 ]]; then
    run_experiment "async_cycleTimers1" "${QS_EXE_ASYNC}" 1
  fi

  analyze_profiles

  log "Done. Inspect ${OUT_ROOT}"
}

main "$@"
