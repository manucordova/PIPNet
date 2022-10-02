#! /bin/bash

#SBATCH -J PIPNet_Train
#SBATCH --time=3-00:00:00
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

source activate PIPNet

#srun python -u $1 > ${1%.py}.log

srun python -u ANALYSE-visualize_training.py ${1%.py} >> ${1%.py}.log

srun python -u ANALYSE-predict_experimental.py ${1%.py} >> ${1%.py}.log

srun python -u ANALYSE-evaluate_model.py ${1%.py} >> ${1%.py}.log
