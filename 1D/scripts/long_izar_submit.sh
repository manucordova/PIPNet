#! /bin/bash

#SBATCH -J PIPNet_Train
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --mem 90G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -A lrm

module purge
module load intel intel-mpi intel-mkl

source activate S2C_gpu

srun python -u $1 > ${1%.py}.log
