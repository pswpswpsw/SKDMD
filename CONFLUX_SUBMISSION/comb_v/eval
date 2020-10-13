#!/bin/bash

#BSUB -a cpu                        # tell lsf this is an openmpi job
#BSUB -c 8000000                        # total cpu time for job (sum of all processes ?)
#BSUB -n 20                              # number of tasks in job
#BSUB -R "affinity[core(1):cpubind=core:distribute=balance]"
#BSUB -R rusage[mem=20000]         # amount of total memory in MB for all processes
#BSUB -J combv_eval                   # job name
#BSUB -e errors.%J                  # error file name in which %J is replaced by the job ID
#BSUB -o output.%J                  # output file name in which %J is replaced by the job ID
##BSUB -q normal
#BSUB -q gpu_p100 # normal                 # choose the queue to use: normal or large_memory
#BSUB -B                        # email job start notification
#BSUB -N                        # email job end notification
#BSUB -u shawnpan@umich.edu     # email address to send notifications
#BSUB -R "select[ngpus>0] rusage[ngpus_excl_p=4]" # if I use gpu_p100

cd ../../EXAMPLES/comb_v/

python3 run_apo_combv_ekdmd.py

# test
# mpirun python3 cv_test.py

