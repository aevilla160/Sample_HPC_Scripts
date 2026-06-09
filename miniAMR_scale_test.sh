#!/bin/bash
#SBATCH --job-name=miniAMR_scale_mpi_comm
#SBATCH --partition=<partition> 
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output=miniAMR_scale_mpi_comm_%j.stdout
#SBATCH --error=miniAMR_scale_mpi_comm_%j.stderr
#SBATCH --export=ALL

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

echo "Max allocated nodes: $SLURM_JOB_NUM_NODES"
echo "Ranks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per rank: $SLURM_CPUS_PER_TASK"

module purge
module load openmpi

# One MPI rank per node, no OpenMP threading
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# -------------------------------
# User paths
# -------------------------------

MINIAMR_EXE=./miniAMR.x

# Fill this in later with your mpiP library path.
# Example:
# MPIP_LIB=/path/to/mpiP/lib/libmpiP.so
MPIP_LIB=/path/to/libmpiP.so

# Enable or disable mpiP
USE_MPIP=1

# -------------------------------
# miniAMR communication-heavy setup
# -------------------------------

NX=8
NY=8
NZ=8

NUM_REFINE=4
MAX_BLOCKS=8000

NUM_TSTEPS=100
STAGES_PER_TS=40

NUM_VARS=80
COMM_VARS=2

STENCIL=27

# Main result directory
RESULT_DIR=miniAMR_scale_results_${SLURM_JOB_ID}
mkdir -p ${RESULT_DIR}

echo "Results directory: ${RESULT_DIR}"

# -------------------------------
# Function to run miniAMR
# -------------------------------

run_miniamr_scale_case () {
    local NNODES=$1
    local NRANKS=${NNODES}

    # miniAMR requires npx * npy * npz = number of MPI ranks.
    if [ "${NNODES}" -eq 2 ]; then
        NPX=2
        NPY=1
        NPZ=1
    elif [ "${NNODES}" -eq 4 ]; then
        NPX=2
        NPY=2
        NPZ=1
    elif [ "${NNODES}" -eq 8 ]; then
        NPX=2
        NPY=2
        NPZ=2
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
    echo "Case directory: ${CASE_DIR}"
    echo "mpiP directory: ${MPIP_DIR}"
    echo "============================================================"

    CMD="${MINIAMR_EXE} \
        --npx ${NPX} --npy ${NPY} --npz ${NPZ} \
        --nx ${NX} --ny ${NY} --nz ${NZ} \
        --num_refine ${NUM_REFINE} \
        --max_blocks ${MAX_BLOCKS} \
        --num_tsteps ${NUM_TSTEPS} \
        --stages_per_ts ${STAGES_PER_TS} \
        --num_vars ${NUM_VARS} \
        --comm_vars ${COMM_VARS} \
        --stencil ${STENCIL} \
        --report_perf 4"

    echo "miniAMR command:"
    echo "${CMD}"

    if [ "${USE_MPIP}" -eq 1 ]; then
        echo "mpiP enabled"
        echo "mpiP library: ${MPIP_LIB}"

        # mpiP options:
        # -k keeps the generated report files
        # The label makes it easier to identify which report belongs to which scale run.
        export MPIP="-k ${CASE_NAME}"

        # Run from inside the mpiP directory so mpiP output lands there.
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
# Scale test
# -------------------------------

run_miniamr_scale_case 2
run_miniamr_scale_case 4
run_miniamr_scale_case 8

echo "============================================================"
echo "miniAMR scale test complete"
echo "Results stored in: ${RESULT_DIR}"
echo "============================================================"
