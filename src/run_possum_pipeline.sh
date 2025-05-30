#!/bin/bash --login
#SBATCH --job-name=possum_pipeline
#SBATCH --error=%x.e.%j
#SBATCH --output=%x.o.%j
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
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
input_dir=$HOME/workarea/rumpy_possums_train/
WDPATH=/scratch/$USER/possum_pipeline.$SLURM_JOBID
rm -rf ${WDPATH}
mkdir -p ${WDPATH}
cd ${WDPATH}

# Copy only the src folder and its contents
echo "Copying src folder from ${input_dir} to ${WDPATH}"
mkdir -p ${WDPATH}/src
cp -r ${input_dir}/src/* ${WDPATH}/src/

# List the contents to verify
echo "Contents of src directory:"
ls -la ${WDPATH}/src

# Setup environment
module purge
module load anaconda
source activate keras-jax
conda list

# Set environment variables to disable XLA and configure TensorFlow
export TF_CPP_MIN_LOG_LEVEL=3
export KERAS_BACKEND=jax
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/apps/languages/cuda/11.7.0"
# Limit JAX to use only one GPU
export CUDA_VISIBLE_DEVICES=0

# Run the pipeline
# Use command line arguments to specify stages if needed:
# --stage 0: Run all stages
# --stage 1: Start from wildlife model training
# --stage 2: Start from possum fine-tuning (requires --wildlife_model)
# --stage 3: Start from pseudo-labeling (requires --possum_model)
# Define path to pre-trained possum model
POSSUM_MODEL="/scratch/c.c1767198/output/rumpy_possum_train/possum_disease_finetuned/ENS/possum_disease_finetuned_ENS.keras"

# Run the pipeline starting from stage 3 (pseudo-labeling)
start="$(date +%s)"
time conda run -n keras-jax python src/possum_pipeline.py --config src/possum_config.yaml --stage 3 --possum_model ${POSSUM_MODEL}
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Possum pipeline $SLURM_JOBID Job-Time $finish seconds
echo End Time is `date` 