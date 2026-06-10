#!/bin/bash
#SBATCH --job-name=miniVite_scale_coll
#SBATCH --partition=<partition>
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-02:00:00
#SBATCH --output=miniVite_scale_coll_%j.stdout
#SBATCH --error=miniVite_scale_coll_%j.stderr
#SBATCH --export=ALL

# =====================================================================
# miniVite collective-stress weak-scaling sweep: 2, 4, 8, 16 nodes
# =====================================================================
#
# REQUIRED BUILD (compile before submitting):
#
#   make clean
#   make OPTFLAGS="-O3 -fopenmp -DUSE_MPI_COLLECTIVES -DPRINT_DIST_STATS"
#
#   -DUSE_MPI_COLLECTIVES  replaces all non-blocking P2P in fillRemoteCommunities
#                          with MPI_Alltoallv / MPI_Alltoall collectives.
#                          NOTE: updateRemoteCommunities always uses Irecv/Isend
#                          regardless of this flag -- see dspl.hpp:978.
#   -DPRINT_DIST_STATS     prints per-rank edge distribution at startup.
#
# =====================================================================
# MPI COLLECTIVE PROFILE (USE_MPI_COLLECTIVES, per Louvain iteration):
#
#   Call 1: MPI_Alltoallv  -- community IDs of boundary vertices
#              send: ssz * 8 B,  recv: rsz * 8 B
#              ssz/rsz fixed after exchangeVertexReqs() setup phase
#
#   Call 2: MPI_Alltoall   -- metadata request sizes (how many community
#              IDs each rank will ask for in Call 3)
#              send/recv: P * 8 B  (small, but a global sync barrier)
#
#   Call 3: MPI_Alltoallv  -- community IDs whose metadata is needed
#              stcsz * 8 B sent,  rtcsz * 8 B recv
#              stcsz/rtcsz VARY per iteration; can grow as communities merge
#
#   Call 4: MPI_Alltoallv  -- community metadata (size + degree)
#              CommInfo = {int64, int64, double} = 24 B (64-bit build)
#              rtcsz * 24 B sent,  stcsz * 24 B recv
#              NOTE: send/recv count arrays are intentionally swapped
#              relative to Call 3 (dspl.hpp:792) -- be aware when profiling
#
#   Call 5: MPI_Alltoall   -- remote community update sizes
#              send/recv: P * 8 B
#
#   Call 6: MPI_Irecv/Isend/Waitall  -- community degree/size deltas
#              CommInfo structs: scnt * 24 B sent,  rcnt * 24 B recv
#              NOT a collective -- always point-to-point (flag has no effect)
#
#   Call 7: MPI_Allreduce  -- global modularity (2 doubles = 16 B)
#
#   Total per iteration: 5 global collective barriers + 1 P2P + 1 Allreduce
#
# =====================================================================
# PARAMETER DESIGN RATIONALE (quantitative):
#
#   Graph: Random Geometric Graph (RGG), 1-D vertex distribution.
#          Rank p owns the horizontal strip [p/P, (p+1)/P) of the unit square.
#          Edge exists between vertices i,j if Euclidean_distance(i,j) <= rn,
#          where rn = (rc + rt) / 2:
#              rc = sqrt( log(nv) / (pi * nv) )   [connectivity threshold]
#              rt = sqrt( 2.0736 / nv )
#
#   Weak scaling: 250,000 vertices per rank.
#   Assertion checked by code: (1/P) > rn  (strip height must exceed radius).
#
#   Scale point validation (rn, avg_degree):
#     P= 2, nv=  500,000: rn=0.002463, avg_deg= 9.5  [assert OK]
#     P= 4, nv=1,000,000: rn=0.001769, avg_deg= 9.8  [assert OK]
#     P= 8, nv=2,000,000: rn=0.001269, avg_deg=10.1  [assert OK]
#     P=16, nv=4,000,000: rn=0.000910, avg_deg=10.4  [assert OK]
#
#   MEMORY (per rank, 64-bit build; PEAK is during RGG generation, NOT the
#   Louvain loop -- the temporary edgeList<EdgeTuple,24B> and the final
#   edge_list_<Edge,16B> are alive simultaneously during the CSR copy, and
#   std::vector growth can transiently overcommit ~1.5-2.0x). Figures for
#   the largest case (P=16, nv=4M, nv/rank=250K, -p 2):
#     directed edges/rank ~ 2.65M (RGG ~2.60M + 2% random ~0.05M)
#     edgeList  : 2.65M * 24B * ~1.5 growth     ~  96 MB
#     edge_list_: 2.65M * 16B (CSR, coexists)   ~  42 MB
#     coords    : up to 8 * (nv/P) * 8B doubles ~  16 MB
#     -------------------------------------------------------------
#     PEAK   ~ 158 MB/rank  (worst case ~190 MB at 2.0x vector growth)
#     STEADY ~ 120-160 MB/rank during the Louvain loop
#   --mem=2G gives ~10x headroom over the ~190 MB worst-case peak.
#
#   -p 2  (2% random cross-process edges):
#          Pure RGG boundary is narrow -- only ~3% of P=16 vertices sit
#          within rn of a strip boundary (~7K boundary vertices/rank).
#          Without -p, Call 1 send = O(7K * 8B) ~ 0.06 MB/rank (weak signal).
#          With -p 2 (P=16, nv=4M): adds ~0.42M random edges globally,
#          ~26K per rank pointing to globally random targets. Nearly all
#          targets are on other ranks, so ssz ~ 32K unique remote vertices
#          -> Call 1 Alltoallv send ~0.25 MB/rank (~4x the bare-RGG boundary
#          traffic). Higher -p increases this volume but at O(n^2) generation
#          cost (see RAND_EDGE_PCT note below) -- 2% is the walltime-feasible
#          sweet spot. Call COUNT (latency-bound, the real scale stressor) is
#          set by iteration count via -t, independent of -p.
#
#   -t 1.0E-9  (convergence threshold, default 1.0E-6):
#          Exit condition: currMod - prevMod < thresh (dspl.hpp:1401).
#          Tightening 1000x forces more Louvain iterations, each firing
#          all 7 MPI calls. Expected iteration count: 20-80 (vs 5-15 default).
#          At 20 iterations: 100 Alltoall/v + 20 Allreduce = 120 global syncs.
#          At 80 iterations: 400 Alltoall/v + 80 Allreduce = 480 global syncs.
#
#   -w  (Euclidean edge weights):
#          Edge weight = Euclidean distance (0, rn]. Produces a smoother
#          modularity landscape, typically increasing iteration count vs
#          unit weights. Random cross-process edges get weight drawn
#          uniformly from (0.01, 1.0) since Euclidean coords are unavailable.
#
#   -l  (LCG random number generator):
#          Uses distributed linear congruential generator with parallel
#          prefix (log2(P) matrix-multiply steps) instead of per-process
#          std::default_random_engine. Ensures reproducible coordinates
#          across runs and removes per-process seed divergence.
#
# =====================================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

echo "Max allocated nodes: $SLURM_JOB_NUM_NODES"
echo "Ranks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per rank: $SLURM_CPUS_PER_TASK"

module purge
module load openmpi

# Single-threaded MPI -- OMP parallel regions exist in miniVite but with
# 1 thread they serialize immediately. This isolates MPI collective cost.
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# -----------------------------------------------------------------------
# User paths
# -----------------------------------------------------------------------

MINIVITE_EXE=./bin/miniVite

MPIP_LIB=/path/to/libmpiP.so

USE_MPIP=1

# -----------------------------------------------------------------------
# RGG parameters
# -----------------------------------------------------------------------

# Vertices per rank (weak scaling baseline: 250K/rank).
# nv for a given run = NV_PER_RANK * NRANKS.
#
# WHY 250K AND NOT 1M: miniVite's RGG generator is brute-force O((nv/P)^2)
# -- three serial all-pairs distance loops (local strip + up-ghost + down-ghost,
# graph.hpp:759/816/848). At 1M/rank that is ~1.5e12 distance evals/rank
# (~37 min) PLUS ~25 min of O(n^2) random-edge dedup = ~65 min/run, and the
# 4-run sweep runs sequentially in one job => ~4.3h, busting the 2h walltime.
# Both generation costs scale as (nv/P)^2; the Louvain loop you actually
# measure scales as (nv/P). At 250K/rank: generation+dedup ~4 min/run,
# whole sweep ~16 min, Louvain loop ~1-2s, identical collective pattern.
# NOTE: miniVite reports generation time and the Louvain "Average total time"
# SEPARATELY (main.cpp:142 vs 192) -- use the latter for scaling analysis;
# generation is one-time setup, not part of that metric.
NV_PER_RANK=250000

# 2% random cross-process edges on top of the RGG strip structure.
# WHY ONLY 2%: random-edge insertion (graph.hpp:1013) does a serial O(edgeList)
# std::find_if dedup per random edge -> O(n^2) generation cost on top of the
# already-O((nv/P)^2) RGG construction. Higher -p inflates both generation time
# and Call-1 byte volume; 2% adds meaningful cross-process traffic (~4x the bare
# RGG boundary) while keeping the dedup to ~tens of seconds at 250K/rank.
# Collective CALL COUNT (the dominant scale stressor) is driven by iteration
# count via -t, not by -p.
RAND_EDGE_PCT=2

# Convergence threshold: 1000x tighter than the default 1E-6.
# Drives more Louvain iterations, each firing 5 collectives + 1 P2P.
THRESHOLD=1.0E-9

# -----------------------------------------------------------------------
# Function to run one miniVite scale case
# -----------------------------------------------------------------------

run_minivite_scale_case () {
    local NNODES=$1
    local NRANKS=${NNODES}                           # 1 rank per node
    local NV=$(( NV_PER_RANK * NRANKS ))             # weak-scaled total vertices

    local CASE_NAME=miniVite_nodes_${NNODES}_ranks_${NRANKS}_nv_${NV}
    local CASE_DIR=${RESULT_DIR}/${CASE_NAME}
    local MPIP_DIR=${CASE_DIR}/mpiP

    mkdir -p "${CASE_DIR}"
    mkdir -p "${MPIP_DIR}"

    echo "============================================================"
    echo "Running miniVite scale case: ${CASE_NAME}"
    echo "Nodes:          ${NNODES}"
    echo "MPI ranks:      ${NRANKS}"
    echo "Ranks per node: 1"
    echo "Total vertices: ${NV}  (${NV_PER_RANK} per rank)"
    echo "Random edges:   -p ${RAND_EDGE_PCT}%"
    echo "Threshold:      -t ${THRESHOLD}"
    echo "Case directory: ${CASE_DIR}"
    echo "mpiP directory: ${MPIP_DIR}"
    echo "============================================================"

    # -n nv  : total vertices (RGG generation mode; requires nv % P == 0 and P = 2^k)
    # -l     : LCG RNG for reproducibility
    # -p pct : add pct% random cross-process edges
    # -w     : Euclidean edge weights (not unit weight)
    # -t thr : modularity improvement threshold for loop exit
    CMD="${MINIVITE_EXE} \
        -n ${NV} \
        -l \
        -p ${RAND_EDGE_PCT} \
        -w \
        -t ${THRESHOLD}"

    echo "miniVite command:"
    echo "${CMD}"

    if [ "${USE_MPIP}" -eq 1 ]; then
        echo "mpiP enabled"
        echo "mpiP library: ${MPIP_LIB}"

        # -k sets the mpiP task name (used as prefix for .mpiP output files).
        # mpiP writes output to the working directory, so we cd into MPIP_DIR.
        export MPIP="-k ${CASE_NAME}"

        (
            cd "${MPIP_DIR}"

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
                > "${OLDPWD}/${CASE_DIR}/miniVite_${CASE_NAME}.out" \
                2> "${OLDPWD}/${CASE_DIR}/miniVite_${CASE_NAME}.err"
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
            > "${CASE_DIR}/miniVite_${CASE_NAME}.out" \
            2> "${CASE_DIR}/miniVite_${CASE_NAME}.err"
    fi

    echo "Finished case: ${CASE_NAME}"
    echo ""
}

# -----------------------------------------------------------------------
# Scale sweep: 2, 4, 8, 16 nodes
# -----------------------------------------------------------------------

RESULT_DIR=miniVite_scale_results_${SLURM_JOB_ID}
mkdir -p "${RESULT_DIR}"

echo "Results directory: ${RESULT_DIR}"

run_minivite_scale_case 2
run_minivite_scale_case 4
run_minivite_scale_case 8
# run_minivite_scale_case 16

echo "============================================================"
echo "miniVite collective-stress scale sweep complete"
echo "Results stored in: ${RESULT_DIR}"
echo "============================================================"
