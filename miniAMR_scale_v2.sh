#!/bin/bash
#SBATCH --job-name=miniAMR_scale_comm
#SBATCH --partition=<partition>
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --output=miniAMR_scale_comm_%j.stdout
#SBATCH --error=miniAMR_scale_comm_%j.stderr
#SBATCH --export=ALL
#
# Optional but recommended for a *communication* benchmark: own the whole node
# so no other job shares the NIC / memory bandwidth and pollutes comm timings.
# Under --exclusive your job gets all node memory; some sites then require
# --mem=0 or no --mem at all. Enable per your cluster's policy:
##SBATCH --exclusive
#
# -------------------------------------------------------------------------
# MEMORY ACCOUNTING (why --mem=48G, not 96G)
#
# Per-rank footprint is driven by the 4-level double**** block storage in
# allocate(). With nx=ny=nz=8, num_vars=80, comm_vars=4 it is approximately:
#
#     ~0.85 MB per block  x  max_num_blocks  +  ~0.5 GB (buffers/parents/dots)
#
#   max_num_blocks =  8000  ->  ~7 GB/rank
#   max_num_blocks = 24000  -> ~22 GB/rank   <-- this script
#
# max_num_blocks is a PER-RANK limit, so per-rank memory is IDENTICAL across
# 2/4/8/16 nodes. 48G gives ~2x headroom over the ~22 GB working set and is
# far easier to schedule than 96G. If your nodes have <48 GB, lower max_blocks.
#
# WHY max_blocks was raised 8000 -> 24000:
# block.c:71 does a HARD exit(-1) ("Need more blocks") if any rank ever needs
# num_active+8 > max_num_blocks. Because fewer ranks concentrate more blocks
# per rank, the 2-NODE case is the most likely to overflow. 24000 buys ~3x
# headroom for the cost of ~15 GB/rank of otherwise-idle RAM we have anyway.
# -------------------------------------------------------------------------

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
# Design rationale (these params, 100 timesteps):
#
#   checksum Allreduce:  CHECKSUM_FREQ=1 fires every stage
#                        80 vars x 40 stages x 100 ts = 320,000 Allreduce
#                        (vs 64,000 at the default CHECKSUM_FREQ=5)
#
#   refine Allreduce:    REFINE_FREQ=2 -> 50 refine() calls; each call runs
#                        NUM_REFINE_STEP iterations, each with ~12 Allreduce
#                        from the 2:1-balance convergence loops -> ~600+
#                        Allreduce from AMR convergence alone.
#
#   Alltoall:            active objects trigger redistribute_blocks() on each
#                        refine inner iteration -> ~500 Alltoall over the run.
#                        Without objects this count is ZERO -- objects are the
#                        single most important ingredient for collective load.
#
#   P2P (halo exchange): COMM_VARS=4 -> 20 comm rounds/stage, latency-bound.
#
#   STENCIL=7:           identical comm pattern to stencil=27 but avoids the
#                        check_input() divergence warning under non-uniform AMR.
# -------------------------------

NX=8
NY=8
NZ=8

NUM_REFINE=5
MAX_BLOCKS=24000      # per-rank; ~22 GB/rank. See MEMORY ACCOUNTING above.

NUM_TSTEPS=100
STAGES_PER_TS=40
CHECKSUM_FREQ=1
REFINE_FREQ=2

NUM_VARS=80
COMM_VARS=4

STENCIL=7

# WEAK SCALING (default here): each rank starts with exactly 1 block, so the
# global problem grows with node count and per-rank base load stays constant.
# This isolates how COLLECTIVE cost scales with rank count (Allreduce ~log P,
# Alltoall ~P) -- the stated goal of this sweep.
#
# For STRONG scaling instead (fixed global problem, shrinking per-rank work),
# do NOT use init=1; set INIT_X/Y/Z so that (npx*INIT_X)x(npy*INIT_Y)x(npz*INIT_Z)
# is the SAME constant base mesh for every node count, and pass --init_x/y/z.
INIT_X=1
INIT_Y=1
INIT_Z=1

# Two bouncing spheres moving in opposite x-directions.
# bounce=1 reflects at the domain boundary so they move for all 100 ts.
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

    # npx * npy * npz must equal the number of MPI ranks (check_input enforces).
    if [ "${NNODES}" -eq 2 ]; then
        NPX=2; NPY=1; NPZ=1
    elif [ "${NNODES}" -eq 4 ]; then
        NPX=2; NPY=2; NPZ=1
    elif [ "${NNODES}" -eq 8 ]; then
        NPX=2; NPY=2; NPZ=2
    elif [ "${NNODES}" -eq 16 ]; then
        NPX=4; NPY=2; NPZ=2
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
    echo "Nodes: ${NNODES}   MPI ranks: ${NRANKS}   Ranks/node: 1"
    echo "Processor grid: ${NPX} x ${NPY} x ${NPZ}"
    echo "Initial blocks/rank: ${INIT_X} x ${INIT_Y} x ${INIT_Z}"
    echo "Case directory: ${CASE_DIR}"
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
        echo "mpiP enabled (library: ${MPIP_LIB})"
        export MPIP="-k ${CASE_NAME}"

        (
            cd ${MPIP_DIR}
            echo "Running from mpiP output directory: $(pwd)   MPIP=${MPIP}"

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
