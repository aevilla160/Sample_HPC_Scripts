#!/bin/bash
#SBATCH --job-name=quicksilver_1node_32core
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0-00:15:00
#SBATCH --output=quicksilver_1node_32core_%j.out
#SBATCH --error=quicksilver_1node_32core_%j.err
#SBATCH --export=ALL

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated node:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

module purge
module load openmpi

echo "Using MPI:"
which mpirun
mpirun --version

# Assumes this script is launched from:
# Quicksilver/Examples/CTS2_Benchmark
QS_EXE="../../src/qs"
INPUT="CTS2.inp"

if [ ! -x "$QS_EXE" ]; then
    echo "ERROR: Quicksilver executable not found or not executable: $QS_EXE"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    exit 1
fi

# 1 node x 32 MPI ranks = 32 total ranks
RANKS=32

# I * J * K must equal total MPI ranks.
# 8 x 4 x 1 = 32
I=8
J=4
K=1

# CTS-style weak scaling:
# local mesh per rank = 16 x 16 x 16
# global mesh = local mesh * domain decomposition
X=$((16 * I))   # 128
Y=$((16 * J))   # 64
Z=$((16 * K))   # 16

# CTS-style particle count:
# 40960 particles per rank
PARTICLES=$((40960 * RANKS))

echo "Running Quicksilver on 1 node with 32 MPI ranks"
echo "Ranks: $RANKS"
echo "Domain decomposition: I=$I J=$J K=$K"
echo "Global mesh: X=$X Y=$Y Z=$Z"
echo "Particles: $PARTICLES"

mpirun -np "$RANKS" "$QS_EXE" -i "$INPUT" \
    -X"$X" -Y"$Y" -Z"$Z" \
    -x"$X" -y"$Y" -z"$Z" \
    -I"$I" -J"$J" -K"$K" \
    -n"$PARTICLES"

echo "Done."
