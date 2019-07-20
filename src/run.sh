#!/bin/sh
#BATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
##SBATCH --socket-per-node=2
##SBATCH --cores-per-socket=14
##SBATCH --cpus-per-task=24
#SBATCH -c 10      # cores requested
#SBATCH --partition=ceashpc # assign task to partition
#SBATCH -w ceashpc-01
##SBATCH --mem-per-cpu=10000
#SBATCH --mem=85000  # memory in Mb
##SBATCH -o outfile  # send stdout to outfile
#SBATCH --mail-type=ALL
#SBATCH -e errfile_4  # send stderr to errfile
#SBATCH --time=3-0:00:00  # time requested in hour:minute:second
#SBATCH --mail-user=aalimu@albany.edu

python adv_coi_sybils_experiment_pipeline_structure.py
       