#!/usr/bin/env bash
#SBATCH --job-name=qs_2node_96
#SBATCH --partition=pbatch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=0-01:00:00
#SBATCH --output=qs_2node_96_%j.stdout
#SBATCH --error=qs_2node_96_%j.stderr
#SBATCH --export=ALL
set -euo pipefail

# Quicksilver 2-node x 96-rank/node runner (blocking vs async termination).
#
# Runs the jobs and collects mpiP profiles only; analysis is run separately
# after the job finishes. Always launches via srun; no dry-run.
#
# Mesh scales with the domain grid (nx = BLOCK*xDom, etc.) so 192 ranks
# decompose cleanly into BLOCK^3 cells/rank. The physical box is fixed at
# 100^3 cm, so each rank's subdomain is small relative to the 100 cm MFP,
# driving heavy cross-rank migration to stress the termination Allreduce.

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Allocated nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null || true

module load openmpi

export OMP_NUM_THREADS=1   # no-op: OpenMP not compiled in (-DHAVE_OPENMP removed)

# ----------------------------- user config ---------------------------------

QS_ROOT="${QS_ROOT:-/usr/workspace/villa17/software/Quicksilver}"
QS_EXE_BLOCKING="${QS_EXE_BLOCKING:-${QS_ROOT}/src/qs}"
# Prebuilt -DHAVE_ASYNC_MPI executable.
QS_EXE_ASYNC="${QS_EXE_ASYNC:-${QS_ROOT}/src/qs_async}"

OUT_ROOT="${OUT_ROOT:-${PWD}/quicksilver_next_steps_$(date +%Y%m%d_%H%M%S)}"

TASKS_PER_NODE="${TASKS_PER_NODE:-96}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
CPU_BIND="${CPU_BIND:-cores}"

USE_MPIP="${USE_MPIP:-1}"
MPIP_LIB="${MPIP_LIB:-/usr/workspace/villa17/.local/lib/libmpiP.so}"

# ranks = TASKS_PER_NODE * nodes. 192 = 2 nodes x 96.
RANKS_LIST="${RANKS_LIST:-192}"
REPS="${REPS:-2}"

# Experiments from the immediate plan.
RUN_BLOCKING_CT0="${RUN_BLOCKING_CT0:-1}"
RUN_ASYNC_CT0="${RUN_ASYNC_CT0:-1}"

# Optional: reproduce the high-stress timer-enabled case alongside controls.
RUN_BLOCKING_CT1="${RUN_BLOCKING_CT1:-0}"
RUN_ASYNC_CT1="${RUN_ASYNC_CT1:-0}"

# Quicksilver communication-stress workload knobs.
PARTICLES_PER_RANK="${PARTICLES_PER_RANK:-100000}"
NUM_STEPS="${NUM_STEPS:-50}"
LOAD_BALANCE="${LOAD_BALANCE:-0}"   # 0 adds a global PopulationControl Allreduce/step

# Per-rank mesh block: each rank owns BLOCK^3 cells (guarantees nx>=xDom).
BLOCK="${BLOCK:-10}"

# Fixed physical box (cm); subdomains shrink as ranks grow.
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

# Balanced 3D domain grid. Product MUST equal ranks (= 96 * nodes).
rank_grid() {
  case "$1" in
    96)   printf '4 4 6\n'  ;;   # 1 node
    192)  printf '8 6 4\n'  ;;   # 2 nodes
    384)  printf '8 8 6\n'  ;;   # 4 nodes
    768)  printf '8 8 12\n' ;;   # 8 nodes
    1536) printf '8 12 16\n';;   # 16 nodes
    *)
      printf 'No domain-grid mapping for %s ranks. Add it in rank_grid().\n' "$1" >&2
      return 1
      ;;
  esac
}

require_runtime_config() {
  if [[ "${USE_MPIP}" -eq 1 && ! -f "${MPIP_LIB}" ]]; then
    printf 'USE_MPIP=1 but MPIP_LIB does not exist: %s\n' "${MPIP_LIB}" >&2
    exit 1
  fi
}

generate_input() {
  local inp="$1"
  local ranks="$2"
  local cycle_timers="$3"
  local xdom ydom zdom nx ny nz nparticles

  read -r xdom ydom zdom < <(rank_grid "${ranks}")
  nx=$((BLOCK * xdom)); ny=$((BLOCK * ydom)); nz=$((BLOCK * zdom))
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
   nx: ${nx}
   ny: ${ny}
   nz: ${nz}
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
}

run_case() {
  local exp_name="$1"
  local exe="$2"
  local cycle_timers="$3"
  local rep_id="$4"
  local ranks="$5"
  local nodes=$((ranks / TASKS_PER_NODE))

  if (( nodes < 1 )); then
    printf 'ranks=%s < TASKS_PER_NODE=%s; need at least one full node\n' "${ranks}" "${TASKS_PER_NODE}" >&2
    exit 1
  fi

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

  log "Running ${case_name} (${nodes} nodes x ${TASKS_PER_NODE} ranks/node)"
  log "  input:   ${inp}"

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

main() {
  require_runtime_config
  mkdir -p "${OUT_ROOT}"

  log "Output root: ${OUT_ROOT}"

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

  log "Done. Profiles under ${OUT_ROOT}; run mpiP analysis separately."
}

main "$@"
