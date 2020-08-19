#!/bin/bash

#Set the job name and wall time limit
#BSUB -J "daemon"
#BSUB -W 8:00

# Set the output and error output paths.
##BSUB -o  %J.o
#BSUB -e  %J.e

# Set cpu options.
#BSUB -n 1 -R  "rusage[mem=4]"
#BSUB -q cpuqueue
#BSUB -T 24

#quit on first error
set -e
cd $LS_SUBCWD
export PATH="/home/rufad/miniconda3/envs/openmm/bin:$PATH" #enter conda path to YOUR chosen environment
export OPENMM_CPU_THREADS="1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
module load cuda/9.2 #load YOUR version of CUDA

