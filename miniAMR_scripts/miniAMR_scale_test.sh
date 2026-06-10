#!/bin/bash
#SBATCH --job-name=miniAMR_scale_comm
#SBATCH --partition=<partition>
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=miniAMR_scale_comm_%j.stdout
#SBATCH --error=miniAMR_scale_comm_%j.stderr
#SBATCH --export=ALL

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

echo "Max allocated nodes: $SLURM_JOB_NUM_NODES"
echo "Ranks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per rank: $SLURM_CPUS_PER_TASK"

module purge
module load openmpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# -------------------------------
# User paths
# -------------------------------

MINIAMR_EXE=./miniAMR.x

MPIP_LIB=/path/to/libmpiP.so

USE_MPIP=1

# -------------------------------
# miniAMR communication-heavy setup
#
# Design rationale (default params, 100 timesteps):
#
#   checksum Allreduce:  CHECKSUM_FREQ=1 fires every stage
#                        80 vars x 40 stages x 100 ts = 320,000 Allreduce
#                        (vs 64,000 at CHECKSUM_FREQ=5 -- 5x reduction avoided)
#
#   refine Allreduce:    REFINE_FREQ=2 -> 50 refine() calls
#                        each call: ~12 Allreduce/inner-iter from 2:1 convergence
#                        loops (6 levels x 2 do-while each), x NUM_REFINE_STEP=5
#                        iterations = ~600+ Allreduce from AMR convergence alone
#
#   Alltoall:            active objects trigger redistribute_blocks() on each
#                        refine inner iteration -> ~2 Alltoall/iter x 5 iters
#                        x 50 refine() calls = ~500 Alltoall
#                        (zero without objects -- the biggest gap in the
#                        original script)
#
#   P2P (halo exchange): COMM_VARS=4 -> 20 comm rounds/stage, stressing
#                        latency; each round sends 4 x 8x8 face = 256 doubles
#
#   STENCIL=7:           identical comm pattern to stencil=27 but avoids
#                        check_input() warning and answer divergence under
#                        non-uniform AMR
# -------------------------------

NX=8
NY=8
NZ=8

NUM_REFINE=5
MAX_BLOCKS=8000

NUM_TSTEPS=100
STAGES_PER_TS=40
CHECKSUM_FREQ=1
REFINE_FREQ=2

NUM_VARS=80
COMM_VARS=4

STENCIL=7

# Two bouncing spheres moving in opposite x-directions.
# bounce=1 reflects at boundaries so they move continuously for all 100 ts.
# move=+-0.01/ts traverses the domain in ~50 ts, then bounces back.
# size=0.15 (15% radius) covers O(multiple blocks) per refinement level.
# inc=0.0 (no size change over time).
#
# Object format: type bounce xcen ycen zcen xmov ymov zmov xsize ysize zsize xinc yinc zinc
OBJECTS="--num_objects 2 \
    --object 0 1  0.25 0.5 0.5  0.01 0.0 0.0  0.15 0.15 0.15  0.0 0.0 0.0 \
    --object 0 1  0.75 0.5 0.5 -0.01 0.0 0.0  0.15 0.15 0.15  0.0 0.0 0.0"

RESULT_DIR=miniAMR_scale_results_${SLURM_JOB_ID}
mkdir -p ${RESULT_DIR}

echo "Results directory: ${RESULT_DIR}"

# -------------------------------
# Function to run miniAMR
# -------------------------------

run_miniamr_scale_case () {
    local NNODES=$1
    local NRANKS=${NNODES}

    # npx * npy * npz must equal number of MPI ranks.
    # init_x/y/z matches the processor grid so every rank starts with 1 block;
    # without this, rank 0 would hold all initial blocks and others start idle.
    if [ "${NNODES}" -eq 2 ]; then
        NPX=2; NPY=1; NPZ=1
        INIT_X=2; INIT_Y=1; INIT_Z=1
    elif [ "${NNODES}" -eq 4 ]; then
        NPX=2; NPY=2; NPZ=1
        INIT_X=2; INIT_Y=2; INIT_Z=1
    elif [ "${NNODES}" -eq 8 ]; then
        NPX=2; NPY=2; NPZ=2
        INIT_X=2; INIT_Y=2; INIT_Z=2
    elif [ "${NNODES}" -eq 16 ]; then
        NPX=4; NPY=2; NPZ=2
        INIT_X=4; INIT_Y=2; INIT_Z=2
    else
        echo "Unsupported node count: ${NNODES}"
        exit 1
    fi

    local CASE_NAME=miniAMR_nodes_${NNODES}_ranks_${NRANKS}
    local CASE_DIR=${RESULT_DIR}/${CASE_NAME}
    local MPIP_DIR=${CASE_DIR}/mpiP

    mkdir -p ${CASE_DIR}
    mkdir -p ${MPIP_DIR}

    echo "============================================================"
    echo "Running miniAMR scale case: ${CASE_NAME}"
    echo "Nodes: ${NNODES}"
    echo "MPI ranks: ${NRANKS}"
    echo "Ranks per node: 1"
    echo "Processor grid: ${NPX} x ${NPY} x ${NPZ}"
    echo "Initial blocks: ${INIT_X} x ${INIT_Y} x ${INIT_Z}"
    echo "Case directory: ${CASE_DIR}"
    echo "mpiP directory: ${MPIP_DIR}"
    echo "============================================================"

    CMD="${MINIAMR_EXE} \
        --npx ${NPX} --npy ${NPY} --npz ${NPZ} \
        --init_x ${INIT_X} --init_y ${INIT_Y} --init_z ${INIT_Z} \
        --nx ${NX} --ny ${NY} --nz ${NZ} \
        --num_refine ${NUM_REFINE} \
        --max_blocks ${MAX_BLOCKS} \
        --num_tsteps ${NUM_TSTEPS} \
        --stages_per_ts ${STAGES_PER_TS} \
        --checksum_freq ${CHECKSUM_FREQ} \
        --refine_freq ${REFINE_FREQ} \
        --num_vars ${NUM_VARS} \
        --comm_vars ${COMM_VARS} \
        --stencil ${STENCIL} \
        ${OBJECTS} \
        --report_perf 15"

    echo "miniAMR command:"
    echo "${CMD}"

    if [ "${USE_MPIP}" -eq 1 ]; then
        echo "mpiP enabled"
        echo "mpiP library: ${MPIP_LIB}"

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
                ${OLDPWD}/${CMD} \
                > ${OLDPWD}/${CASE_DIR}/miniAMR_${CASE_NAME}.out \
                2> ${OLDPWD}/${CASE_DIR}/miniAMR_${CASE_NAME}.err
        )

    else
        echo "mpiP disabled"

        srun \
            --nodes=${NNODES} \
            --ntasks=${NRANKS} \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --cpu-bind=cores \
            ${CMD} \
            > ${CASE_DIR}/miniAMR_${CASE_NAME}.out \
            2> ${CASE_DIR}/miniAMR_${CASE_NAME}.err
    fi

    echo "Finished case: ${CASE_NAME}"
}

# -------------------------------
# Scale sweep: 2, 4, 8, 16 nodes
# -------------------------------

run_miniamr_scale_case 2
run_miniamr_scale_case 4
run_miniamr_scale_case 8
run_miniamr_scale_case 16

echo "============================================================"
echo "miniAMR scale test complete"
echo "Results stored in: ${RESULT_DIR}"
echo "============================================================"
