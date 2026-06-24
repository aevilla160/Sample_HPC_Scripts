#!/bin/bash
#SBATCH --job-name=qs_2node_96
#SBATCH --partition=pbatch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=0-00:30:00
#SBATCH --output=qs_2node_96_%j.stdout
#SBATCH --error=qs_2node_96_%j.stderr
#SBATCH --export=ALL
set -euo pipefail

# Single Quicksilver run: 2 nodes x 96 ranks/node = 192 ranks.
# Always srun, no dry-run. Collects an mpiP profile; analysis is separate.

echo "Job ID: ${SLURM_JOB_ID:-manual}"
scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null || true

module load openmpi
export OMP_NUM_THREADS=1   # no-op: OpenMP not compiled in

# ----------------------------- config (env-overridable) --------------------
QS_ROOT="${QS_ROOT:-/usr/workspace/villa17/software/Quicksilver}"
QS_EXE="${QS_EXE:-${QS_ROOT}/src/qs}"
OUT_ROOT="${OUT_ROOT:-${PWD}/qs_2node_96_${SLURM_JOB_ID:-manual}}"

USE_MPIP="${USE_MPIP:-1}"
MPIP_LIB="${MPIP_LIB:-/usr/workspace/villa17/.local/lib/libmpiP.so}"

NODES="${NODES:-2}"
RANKS_PER_NODE="${RANKS_PER_NODE:-96}"
RANKS=$((NODES * RANKS_PER_NODE))            # 192

# Balanced 3D domain grid for 192 ranks (product must equal RANKS).
XDOM="${XDOM:-8}"; YDOM="${YDOM:-6}"; ZDOM="${ZDOM:-4}"

# Per-rank mesh block: each rank owns BLOCK^3 cells (guarantees nx>=xDom and a
# clean decomposition). Box fixed at 100^3 cm -> small subdomains vs 100 cm MFP.
BLOCK="${BLOCK:-10}"
NX=$((BLOCK * XDOM)); NY=$((BLOCK * YDOM)); NZ=$((BLOCK * ZDOM))

PARTICLES_PER_RANK="${PARTICLES_PER_RANK:-100000}"
NPARTICLES=$((RANKS * PARTICLES_PER_RANK))
NUM_STEPS="${NUM_STEPS:-50}"
LOAD_BALANCE="${LOAD_BALANCE:-0}"
CYCLE_TIMERS="${CYCLE_TIMERS:-0}"

LX="${LX:-100}"; LY="${LY:-100}"; LZ="${LZ:-100}"
DT="${DT:-1e-08}"; FMAX="${FMAX:-0.1}"
TOTAL_XS="${TOTAL_XS:-0.01}"
SCATTER_RATIO="${SCATTER_RATIO:-1.0}"; ABSORB_RATIO="${ABSORB_RATIO:-0.1}"; FISSION_RATIO="${FISSION_RATIO:-0.1}"
NUBAR="${NUBAR:-2.4}"; N_ISO="${N_ISO:-10}"; N_REACT="${N_REACT:-9}"
SOURCE_RATE="${SOURCE_RATE:-1e+10}"
EMIN="${EMIN:-1e-09}"; EMAX="${EMAX:-20}"; N_GROUPS="${N_GROUPS:-230}"
SEED="${SEED:-1029384756}"
# ---------------------------------------------------------------------------

if (( XDOM * YDOM * ZDOM != RANKS )); then
  echo "ERROR: grid ${XDOM}x${YDOM}x${ZDOM} != ${RANKS} ranks" >&2; exit 1
fi
if [[ "${USE_MPIP}" -eq 1 && ! -f "${MPIP_LIB}" ]]; then
  echo "ERROR: MPIP_LIB not found: ${MPIP_LIB}" >&2; exit 1
fi
if [[ ! -x "${QS_EXE}" ]]; then
  echo "ERROR: QS binary not executable: ${QS_EXE}" >&2; exit 1
fi

CASE="qs_nodes_${NODES}_ranks_${RANKS}"
CASE_DIR="${OUT_ROOT}/${CASE}"
MPIP_DIR="${CASE_DIR}/mpiP"
INP="${CASE_DIR}/${CASE}.inp"
mkdir -p "${MPIP_DIR}"

cat > "${INP}" <<EOF
Simulation:
   dt: ${DT}
   fMax: ${FMAX}
   boundaryCondition: reflect
   loadBalance: ${LOAD_BALANCE}
   cycleTimers: ${CYCLE_TIMERS}
   nParticles: ${NPARTICLES}
   nSteps: ${NUM_STEPS}
   nx: ${NX}
   ny: ${NY}
   nz: ${NZ}
   lx: ${LX}
   ly: ${LY}
   lz: ${LZ}
   xDom: ${XDOM}
   yDom: ${YDOM}
   zDom: ${ZDOM}
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

srun_cmd=(srun --nodes="${NODES}" --ntasks="${RANKS}" --ntasks-per-node="${RANKS_PER_NODE}"
          --cpus-per-task=1 --cpu-bind=cores)
if [[ "${USE_MPIP}" -eq 1 ]]; then
  srun_cmd+=(--export="ALL,LD_PRELOAD=${MPIP_LIB},MPIP=-k ${CASE}")
fi
srun_cmd+=("${QS_EXE}" -i "${INP}")

echo "Running ${CASE}: grid ${XDOM}x${YDOM}x${ZDOM}, mesh ${NX}x${NY}x${NZ}, ${NPARTICLES} particles"
echo "Output: ${CASE_DIR}"

# mpiP writes its report into the working directory.
( cd "${MPIP_DIR}" && "${srun_cmd[@]}" > "${CASE_DIR}/${CASE}.out" 2> "${CASE_DIR}/${CASE}.err" )

echo "Done. Profile + output under ${CASE_DIR}; run mpiP analysis separately."
