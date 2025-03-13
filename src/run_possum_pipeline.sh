#!/bin/bash --login
#SBATCH --job-name=possum_pipeline
#SBATCH --error=%x.e.%j
#SBATCH --output=%x.o.%j
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=148g
#SBATCH --account=scw1985

clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_3_exclusive"

echo '----------------------------------------------------'
echo ' NODE USED = '$SLURM_NODELIST
echo ' SLURM_JOBID = '$SLURM_JOBID
echo ' OMP_NUM_THREADS = '$OMP_NUM_THREADS
echo ' ncores = '$NP
echo ' PPN = ' $PPN
echo '----------------------------------------------------'
#
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
#
echo Number of Processing Elements is $NP
echo Number of mpiprocs per node is $PPN
env

# Setup working directory
WDPATH=/scratch/$USER/possum_pipeline.$SLURM_JOBID
rm -rf ${WDPATH}
mkdir -p ${WDPATH}
cd ${WDPATH}

# Clone the repository
git clone https://github.com/oagn/rumpy_possums_train.git .

# Setup environment
module purge
module load anaconda
source activate keras-jax
conda list

# Run the pipeline
# Use command line arguments to specify stages if needed:
# --stage 0: Run all stages
# --stage 1: Start from wildlife model training
# --stage 2: Start from possum fine-tuning (requires --wildlife_model)
# --stage 3: Start from pseudo-labeling (requires --possum_model)

start="$(date +%s)"
time python src/possum_pipeline.py --config src/possum_config.yaml
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Possum pipeline $SLURM_JOBID Job-Time $finish seconds
echo End Time is `date` 