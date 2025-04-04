#!/bin/bash --login
#SBATCH --job-name=possum_finetuning
#SBATCH --error=%x.e.%j
#SBATCH --output=%x.o.%j
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
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

# Check for command line arguments
# -w: Use wildlife model (default)
# -i: Use ImageNet weights
# -h: Show help message

# Default values
USE_WILDLIFE=true
MODEL_SOURCE="wildlife"

# Parse command line arguments
while getopts "wih" opt; do
  case $opt in
    w)
      USE_WILDLIFE=true
      MODEL_SOURCE="wildlife"
      ;;
    i)
      USE_WILDLIFE=false
      MODEL_SOURCE="imagenet"
      ;;
    h)
      echo "Usage: $0 [-w] [-i] [-h]"
      echo "  -w: Use wildlife model as starting point (default)"
      echo "  -i: Use ImageNet weights as starting point"
      echo "  -h: Show this help message"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Setup working directory
input_dir=$HOME/workarea/rumpy_possums_train/
WDPATH=/scratch/$USER/possum_finetuning.$SLURM_JOBID
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

# Config file location
CONFIG_FILE="${WDPATH}/src/possum_config.yaml"

# Define path to pre-trained wildlife model
WILDLIFE_MODEL="/scratch/$USER/output/rumpy_possum_train/possum_disease_wildlife/ENS/possum_disease_wildlife_ENS.keras"

# Run the pipeline
start="$(date +%s)"

if $USE_WILDLIFE; then
    echo "Starting fine-tuning from wildlife model: ${WILDLIFE_MODEL}"
    # Run stage 2 (fine-tuning) with wildlife model, passing model source as an environment variable
    # The --stage 2 argument specifies to only run the fine-tuning step without continuing to pseudo-labeling
    MODEL_SOURCE=${MODEL_SOURCE} time conda run -n keras-jax python src/possum_pipeline.py --config ${CONFIG_FILE} --stage 2 --wildlife_model ${WILDLIFE_MODEL}
else
    echo "Starting fine-tuning from ImageNet weights"
    # Run stage 2 (fine-tuning) with ImageNet weights (by not providing a wildlife model)
    # The --stage 2 argument specifies to only run the fine-tuning step without continuing to pseudo-labeling
    MODEL_SOURCE=${MODEL_SOURCE} time conda run -n keras-jax python src/possum_pipeline.py --config ${CONFIG_FILE} --stage 2
fi

stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Possum fine-tuning $SLURM_JOBID Job-Time $finish seconds
echo End Time is `date` 