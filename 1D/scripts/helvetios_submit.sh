#! /bin/bash

#SBATCH -J PIPNet_Train
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --mem 180G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH -A lrm

module purge
module load intel intel-mpi intel-mkl

source activate S2C_gpu

srun python -u $1 > ${1%.py}.log
