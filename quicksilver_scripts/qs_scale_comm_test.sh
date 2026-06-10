#!/bin/bash
#SBATCH --job-name=qs_scale_comm
#SBATCH --partition=<partition>
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=qs_scale_comm_%j.stdout
#SBATCH --error=qs_scale_comm_%j.stderr
#SBATCH --export=ALL

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST
echo "Max allocated nodes:  $SLURM_JOB_NUM_NODES"
echo "Ranks per node:       $SLURM_NTASKS_PER_NODE"
echo "CPUs per rank:        $SLURM_CPUS_PER_TASK"

module purge
module load openmpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# -----------------------------------------------------------------------
# User paths
# -----------------------------------------------------------------------

QS_EXE=./src/qs          # path to the compiled Quicksilver binary
MPIP_LIB=/path/to/libmpiP.so
USE_MPIP=1

# -----------------------------------------------------------------------
# Communication-heavy Quicksilver design rationale
#
# Three knobs drive collective frequency in QS (from source analysis):
#
# (A) Termination-detection Allreduce  [DOMINANT — inside tracking loop]
#       MPI_Allreduce(2 x int64_t = 16 bytes, MPI_SUM) per vault round.
#       Called V+1 times per time step, where V = number of rounds of
#       cross-rank particle migration before the global population drains.
#
#       Lever: totalCrossSection=0.01 → MFP = 1/Σ_t = 100 cm
#       Geometry: lx=ly=lz=100 cm; domain grid for 16 nodes = 4x2x2
#         → x-slab width = 100/4 = 25 cm per rank
#         → avg crossings before collision ≈ MFP/slab ≈ 4 in x alone
#       boundaryCondition=reflect: no escapes, particles stay in system
#       → V ≈ 6-10 rounds/step → ~700-1100 termination Allreduces total
#         over 100 steps.
#
#       Without this (totalCrossSection=1.0, MFP=1 cm): V ≈ 1-2 rounds.
#       That is a 5-10x reduction in termination Allreduce count.
#
# (B) Per-step admin Allreduces  [cycleInit + cycleFinalize, 4/step]
#       MC_SourceNow:      MPI_Allreduce(1 double)   — source weight norm
#       PopulationControl: MPI_Allreduce(1 uint64_t) — ENABLED by
#                          loadBalance=0 (the QS default, homogeneous
#                          benchmark overrides to 1 and skips this)
#       CycleFinalize:     MPI_Allreduce(13 x uint64_t = 104 bytes)
#                          — balance counters (absorb/census/escape/etc.)
#       ScalarFluxSum:     MPI_Allreduce(1 double)
#       Total: 4 x 100 steps = 400 admin Allreduces
#
# (C) Per-step timer collectives  [cycleTimers=1, 11 collectives/step]
#       After each step: 1 MPI_Barrier
#                      + 3 MPI_Reduce (MAX/MIN/SUM over 7 timer values)
#                      + 7 MPI_Gather (1 uint64_t/rank per timer, to
#                        rank 0 for std-dev calculation)
#       The "cycleTracking_Test_Done" timer captures Allreduce stall
#       time directly — useful for scaling analysis alongside mpiP.
#       Total: 100 steps x 11 = 1100 extra collectives
#
# Grand total estimate — 16 nodes, 100 steps, V=8 (mid-range):
#   MPI_Allreduce: ~900 (termination) + 400 (admin) = ~1300
#   MPI_Reduce:    300  (timers, to rank 0)
#   MPI_Gather:    700  (timers, each gathers N uint64_t to rank 0)
#   MPI_Barrier:   ~104 (100 cycleTimers + 4 init/teardown)
#   ─────────────────────────────────────────────────────────────
#   Total:         ~2400 collective operations
#   vs ~60 in default 10-step loadBalance=1 cycleTimers=0 run  (~40x)
#
# Material cross section design:
#   absorptionCrossSectionRatio=0.1  low absorption → particles survive
#                                    longer, accumulating more migrations
#   fissionCrossSectionRatio=0.1     light fission (nuBar=2.4) produces
#                                    secondaries; PopulationControl clamps
#                                    total count to nParticles each step
#   scatteringCrossSectionRatio=1.0  dominant scatter → isotropic
#                                    redirection per collision → sustained
#                                    cross-rank traffic after each scatter
#
# Weak scaling: nParticles = NNODES x PARTICLES_PER_RANK
#   Keeps per-rank work constant; Allreduce latency scales with log2(N).
#   This isolates the collective overhead from the computation load.
#   Switch to fixed nParticles for strong scaling to isolate latency.
#
# Non-blocking Allreduce note:
#   Compile with -DHAVE_ASYNC_MPI to use MPI_Iallreduce for termination
#   detection (MC_Particle_Buffer.cc:638). Default build uses blocking
#   MPI_Allreduce — higher stall time, harder stress on the network.
# -----------------------------------------------------------------------

PARTICLES_PER_RANK=1000000   # 1M/rank weak scaling; reduce to 100000 for quick tests

NX=10
NY=10
NZ=10
LX=100
LY=100
LZ=100

NUM_STEPS=100
DT="1e-08"
FMAX=0.1

LOAD_BALANCE=0      # enables PopulationControl Allreduce each step
CYCLE_TIMERS=1      # enables 11 timer collectives (barrier+reduce+gather) per step

TOTAL_XS=0.01       # MFP = 100 cm >> per-rank domain width → heavy streaming
SCATTER_RATIO=1.0
ABSORB_RATIO=0.1
FISSION_RATIO=0.1
NUBAR=2.4
N_ISO=10
N_REACT=9
SOURCE_RATE="1e+10"

EMIN="1e-09"
EMAX=20
N_GROUPS=230
SEED=1029384756

RESULT_DIR=qs_scale_results_${SLURM_JOB_ID}
mkdir -p ${RESULT_DIR}

echo "Results directory: ${RESULT_DIR}"

# -----------------------------------------------------------------------
# Function to run one Quicksilver scale case
# -----------------------------------------------------------------------

run_qs_scale_case () {
    local NNODES=$1
    local NRANKS=${NNODES}   # 1 rank per node

    # xDom * yDom * zDom must equal NRANKS.
    # Grid chosen to keep domain aspect ratio close to cubic.
    if   [ "${NNODES}" -eq  2 ]; then XDOM=2; YDOM=1; ZDOM=1
    elif [ "${NNODES}" -eq  4 ]; then XDOM=2; YDOM=2; ZDOM=1
    elif [ "${NNODES}" -eq  8 ]; then XDOM=2; YDOM=2; ZDOM=2
    elif [ "${NNODES}" -eq 16 ]; then XDOM=4; YDOM=2; ZDOM=2
    else
        echo "Unsupported node count: ${NNODES}"; exit 1
    fi

    # Weak scaling: total particles grows with node count.
    local NPARTICLES=$(( NNODES * PARTICLES_PER_RANK ))

    # Per-rank domain dimensions and MFP ratio (informational).
    local SLAB_X=$(echo "scale=1; ${LX}/${XDOM}" | bc)
    local SLAB_Y=$(echo "scale=1; ${LY}/${YDOM}" | bc)
    local SLAB_Z=$(echo "scale=1; ${LZ}/${ZDOM}" | bc)
    local MFP=$(echo "scale=1; 1/${TOTAL_XS}" | bc)

    local CASE_NAME=qs_nodes_${NNODES}_ranks_${NRANKS}
    local CASE_DIR=${RESULT_DIR}/${CASE_NAME}
    local MPIP_DIR=${CASE_DIR}/mpiP

    mkdir -p ${CASE_DIR} ${MPIP_DIR}

    local INP=${CASE_DIR}/qs_comm_stress_${CASE_NAME}.inp

    # ------------------------------------------------------------------
    # Generate per-case input file.
    # Note: QS input file values override command-line args.
    # The output of every QS run is itself a valid input file.
    # ------------------------------------------------------------------
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

    echo "============================================================"
    echo "Running Quicksilver scale case: ${CASE_NAME}"
    echo "  Nodes:             ${NNODES}"
    echo "  MPI ranks:         ${NRANKS} (1 per node)"
    echo "  Domain grid:       ${XDOM} x ${YDOM} x ${ZDOM}"
    echo "  nParticles:        ${NPARTICLES} (${PARTICLES_PER_RANK}/rank)"
    echo "  MFP:               ${MFP} cm"
    echo "  Per-rank slab:     ${SLAB_X} x ${SLAB_Y} x ${SLAB_Z} cm"
    echo "  MFP/x-slab ratio:  $(echo "scale=1; ${MFP}/${SLAB_X}" | bc)  (avg x-crossings before collision)"
    echo "  Input file:        ${INP}"
    echo "  Case directory:    ${CASE_DIR}"
    echo "  mpiP directory:    ${MPIP_DIR}"
    echo "============================================================"

    if [ "${USE_MPIP}" -eq 1 ]; then
        echo "mpiP enabled: ${MPIP_LIB}"
        export MPIP="-k ${CASE_NAME}"
        (
            cd ${MPIP_DIR}
            echo "Running from mpiP output directory: $(pwd)"
            echo "MPIP=${MPIP}"

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
    else
        echo "mpiP disabled"
        srun \
            --nodes=${NNODES} \
            --ntasks=${NRANKS} \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --cpu-bind=cores \
            ${QS_EXE} -i ${INP} \
            > ${CASE_DIR}/${CASE_NAME}.out \
            2> ${CASE_DIR}/${CASE_NAME}.err
    fi

    echo "Finished case: ${CASE_NAME}"
}

# -----------------------------------------------------------------------
# Scale sweep: 2, 4, 8, 16 nodes
# -----------------------------------------------------------------------

run_qs_scale_case  2
run_qs_scale_case  4
run_qs_scale_case  8
run_qs_scale_case 16

echo "============================================================"
echo "Quicksilver scale test complete"
echo "Results stored in: ${RESULT_DIR}"
echo "============================================================"
