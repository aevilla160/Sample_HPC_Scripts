#!/bin/bash
#SBATCH --job-name=qs_smoke
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=qs_smoke_%j.stdout
#SBATCH --error=qs_smoke_%j.stderr
#SBATCH --export=ALL

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

module purge
module load openmpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# -----------------------------------------------------------------------
# TODO: set this to your libmpiP.so path before submitting
# -----------------------------------------------------------------------
MPIP_LIB=/path/to/libmpiP.so

QS_EXE=./src/qs

# Smoke test settings: 5 steps, 10k particles/rank.
# Same communication-heavy knobs as the full scale test so the
# collective pattern is identical — just far fewer iterations.
PARTICLES_PER_RANK=10000
NUM_STEPS=5

NX=10; NY=10; NZ=10
LX=100; LY=100; LZ=100
DT="1e-08"
FMAX=0.1

LOAD_BALANCE=0     # PopulationControl Allreduce fires every step
CYCLE_TIMERS=1     # barrier + 3 Reduce + 7 Gather per step

TOTAL_XS=0.01      # MFP=100 cm → heavy cross-rank streaming
SCATTER_RATIO=1.0
ABSORB_RATIO=0.1
FISSION_RATIO=0.1
NUBAR=2.4
N_ISO=10; N_REACT=9
SOURCE_RATE="1e+10"
EMIN="1e-09"; EMAX=20; N_GROUPS=230
SEED=1029384756

RESULT_DIR=qs_smoke_results_${SLURM_JOB_ID}
mkdir -p ${RESULT_DIR}

# -----------------------------------------------------------------------
# Validate mpiP library exists before launching any runs
# -----------------------------------------------------------------------
if [ ! -f "${MPIP_LIB}" ]; then
    echo "ERROR: mpiP library not found at: ${MPIP_LIB}"
    echo "       Set MPIP_LIB in this script and resubmit."
    exit 1
fi

if [ ! -x "${QS_EXE}" ]; then
    echo "ERROR: Quicksilver binary not found or not executable: ${QS_EXE}"
    exit 1
fi

echo "mpiP library:  ${MPIP_LIB}"
echo "QS binary:     ${QS_EXE}"
echo "Results dir:   ${RESULT_DIR}"

# -----------------------------------------------------------------------
# Run function
# -----------------------------------------------------------------------

run_smoke_case () {
    local NNODES=$1
    local NRANKS=${NNODES}

    if   [ "${NNODES}" -eq 2 ]; then XDOM=2; YDOM=1; ZDOM=1
    elif [ "${NNODES}" -eq 4 ]; then XDOM=2; YDOM=2; ZDOM=1
    else echo "Unsupported node count: ${NNODES}"; exit 1
    fi

    local NPARTICLES=$(( NNODES * PARTICLES_PER_RANK ))
    local SLAB_X=$(echo "scale=1; ${LX}/${XDOM}" | bc)
    local MFP=$(echo "scale=0; 1/${TOTAL_XS}" | bc)

    local CASE_NAME=qs_smoke_nodes_${NNODES}
    local CASE_DIR=${RESULT_DIR}/${CASE_NAME}
    local MPIP_DIR=${CASE_DIR}/mpiP
    local INP=${CASE_DIR}/${CASE_NAME}.inp

    mkdir -p ${CASE_DIR} ${MPIP_DIR}

    cat > ${INP} << EOF
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

    echo "------------------------------------------------------------"
    echo "Smoke case: ${CASE_NAME}"
    echo "  Nodes / ranks:  ${NNODES} / ${NRANKS}"
    echo "  Domain grid:    ${XDOM} x ${YDOM} x ${ZDOM}"
    echo "  nParticles:     ${NPARTICLES} (${PARTICLES_PER_RANK}/rank)"
    echo "  nSteps:         ${NUM_STEPS}"
    echo "  MFP / x-slab:   ${MFP} cm / ${SLAB_X} cm"
    echo "  mpiP output:    ${MPIP_DIR}/"
    echo "------------------------------------------------------------"

    export MPIP="-k ${CASE_NAME}"

    (
        cd ${MPIP_DIR}

        srun \
            --nodes=${NNODES} \
            --ntasks=${NRANKS} \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --cpu-bind=cores \
            --export=ALL,LD_PRELOAD=${MPIP_LIB},MPIP="${MPIP}" \
            ${OLDPWD}/${QS_EXE} -i ${OLDPWD}/${INP} \
            > ${OLDPWD}/${CASE_DIR}/${CASE_NAME}.out \
            2> ${OLDPWD}/${CASE_DIR}/${CASE_NAME}.err
    )

    local EXIT_CODE=$?

    # ------------------------------------------------------------------
    # Basic pass/fail checks
    # ------------------------------------------------------------------
    echo ""
    echo "=== Results: ${CASE_NAME} ==="

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "FAIL  QS exited with code ${EXIT_CODE}"
        echo "      stderr: ${CASE_DIR}/${CASE_NAME}.err"
    else
        echo "PASS  QS exited cleanly"
    fi

    # QS prints "Figure Of Merit" at the end of a successful run
    if grep -q "Figure Of Merit" ${CASE_DIR}/${CASE_NAME}.out 2>/dev/null; then
        local FOM
        FOM=$(grep "Figure Of Merit" ${CASE_DIR}/${CASE_NAME}.out | awk '{print $4}')
        echo "PASS  Figure Of Merit found: ${FOM} segments/sec"
    else
        echo "FAIL  'Figure Of Merit' not found in output"
        echo "      stdout: ${CASE_DIR}/${CASE_NAME}.out"
    fi

    # mpiP writes a .mpiP file named after the key (-k flag)
    local MPIP_FILE
    MPIP_FILE=$(ls ${MPIP_DIR}/${CASE_NAME}*.mpiP 2>/dev/null | head -1)
    if [ -n "${MPIP_FILE}" ]; then
        echo "PASS  mpiP report:  ${MPIP_FILE}"
        # Print the collective summary section from mpiP
        echo ""
        echo "--- mpiP collective summary (${CASE_NAME}) ---"
        grep -A 30 "Collective" ${MPIP_FILE} 2>/dev/null | head -35 || true
    else
        echo "FAIL  No mpiP output file found in ${MPIP_DIR}/"
        echo "      Check LD_PRELOAD attached correctly"
    fi

    echo ""
}

# -----------------------------------------------------------------------
# Smoke sweep: 2 nodes, then 4 nodes
# -----------------------------------------------------------------------

run_smoke_case 2
run_smoke_case 4

echo "============================================================"
echo "Smoke test complete. Results in: ${RESULT_DIR}"
echo "============================================================"
