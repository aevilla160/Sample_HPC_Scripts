#!/bin/bash
#SBATCH --job-name=miniAMR_smoke_2node
#SBATCH --partition=<partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=0-00:30:00
#SBATCH --output=miniAMR_smoke_2node_%j.stdout
#SBATCH --error=miniAMR_smoke_2node_%j.stderr
#SBATCH --export=ALL
#
# -------------------------------------------------------------------------
# PURPOSE
# Cheap pre-flight for miniAMR_scale_test_v2.sh. The full sweep aborts with a
# hard exit(-1) (block.c:71, "Need more blocks") if any rank ever needs
# num_active+8 > max_num_blocks. max_num_blocks is PER RANK, so the 2-node case
# concentrates the most blocks per rank and is the most likely to overflow.
#
# This runs ONLY the 2-node case with the SAME refinement drivers as the sweep
# but few timesteps, then reports the peak per-rank block count so you can size
# MAX_BLOCKS for the real run before burning a 16-node allocation.
# -------------------------------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

module purge
module load openmpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

MINIAMR_EXE=./miniAMR.x

# -------------------------------
# Smoke-test parameters
# Identical refinement drivers to the sweep (nx/ny/nz, num_refine, num_vars,
# objects) so block growth is representative -- only the timestep count is cut.
# -------------------------------

NX=8
NY=8
NZ=8

NUM_REFINE=5
MAX_BLOCKS=24000      # value you intend to use in the sweep; we test it here

NUM_TSTEPS=20         # short: peak shell size is reached early; 20 > the
                      # refine_freq=2 cadence so refinement fires ~10 times
STAGES_PER_TS=4       # small -- stages do not change the mesh, only comm volume
CHECKSUM_FREQ=1
REFINE_FREQ=2

NUM_VARS=80
COMM_VARS=4

STENCIL=7

# 2-node decomposition (the sweep's worst case for blocks/rank)
NPX=2; NPY=1; NPZ=1

# Weak-scaling base mesh: 1 block/rank (must match the sweep's choice)
INIT_X=1
INIT_Y=1
INIT_Z=1

# Same two bouncing spheres as the sweep.
# type bounce xcen ycen zcen xmov ymov zmov xsize ysize zsize xinc yinc zinc
OBJECTS="--num_objects 2 \
    --object 0 1  0.25 0.5 0.5  0.01 0.0 0.0  0.15 0.15 0.15  0.0 0.0 0.0 \
    --object 0 1  0.75 0.5 0.5 -0.01 0.0 0.0  0.15 0.15 0.15  0.0 0.0 0.0"

RESULT_DIR=miniAMR_smoke_2node_${SLURM_JOB_ID}
mkdir -p ${RESULT_DIR}
OUT=${RESULT_DIR}/miniAMR_smoke_2node.out
ERR=${RESULT_DIR}/miniAMR_smoke_2node.err

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

echo "============================================================"
echo "2-node smoke test"
echo "max_blocks under test: ${MAX_BLOCKS} (per rank)"
echo "Processor grid: ${NPX} x ${NPY} x ${NPZ}   timesteps: ${NUM_TSTEPS}"
echo "miniAMR command:"
echo "${CMD}"
echo "============================================================"

srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    ${CMD} \
    > ${OUT} \
    2> ${ERR}

RC=$?

# -------------------------------
# Verdict
# -------------------------------
echo "============================================================"
echo "SMOKE TEST VERDICT"
echo "============================================================"

if grep -q "Need more blocks" ${OUT} ${ERR}; then
    echo "RESULT: FAIL -- max_blocks=${MAX_BLOCKS} OVERFLOWED."
    echo "The full sweep WILL crash at the 2-node case. Raise MAX_BLOCKS."
    grep "Need more blocks" ${OUT} ${ERR}
elif [ ${RC} -ne 0 ]; then
    echo "RESULT: ERROR -- srun exited ${RC} for a reason other than block"
    echo "overflow. Inspect ${ERR}."
else
    # report_perf bit 8 prints "Total number of blocks at timestep N is M"
    PEAK=$(grep "Total number of blocks at timestep" ${OUT} \
           | awk '{print $NF}' | sort -n | tail -1)
    echo "RESULT: PASS -- no overflow at max_blocks=${MAX_BLOCKS}."
    echo "Peak GLOBAL active blocks observed: ${PEAK:-<not reported>}"
    echo "Peak PER-RANK (>= global/2 at 2 nodes, but load imbalance means a"
    echo "single rank can hold more): check 'total_blocks_ts_max' in the"
    echo "profile output below."
    grep -i "total_blocks_ts_max\|Maximum number of blocks" ${OUT}
    echo ""
    echo "GUIDANCE: set the sweep's MAX_BLOCKS to ~2x the peak per-rank value"
    echo "to cover load imbalance and the longer 100-ts run."
fi

echo "Full output: ${OUT}"
echo "============================================================"
