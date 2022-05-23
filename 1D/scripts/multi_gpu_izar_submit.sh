#! /bin/bash

#SBATCH -J PIPNet_Train
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --mem 180G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH -A lrm

module purge
module load intel intel-mpi intel-mkl

source activate PIPNet

srun python -u $1 > ${1%.py}.log
