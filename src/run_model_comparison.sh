#!/bin/bash --login
#SBATCH --job-name=model_comparison
#SBATCH --error=%x.e.%j
#SBATCH --output=%x.o.%j
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
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

# Default values
MODEL1=""
MODEL2=""
TEST_DATA=""
OUTPUT_DIR="comparison_results"
MODEL1_NAME="ImageNet"
MODEL2_NAME="Wildlife"
MODEL1_HISTORY=""
MODEL2_HISTORY=""
BATCH_SIZE=16

# Parse command line arguments
while getopts "1:2:t:o:n:m:h:i:b:" opt; do
  case $opt in
    1)
      MODEL1="$OPTARG"
      ;;
    2)
      MODEL2="$OPTARG"
      ;;
    t)
      TEST_DATA="$OPTARG"
      ;;
    o)
      OUTPUT_DIR="$OPTARG"
      ;;
    n)
      MODEL1_NAME="$OPTARG"
      ;;
    m)
      MODEL2_NAME="$OPTARG"
      ;;
    h)
      MODEL1_HISTORY="$OPTARG"
      ;;
    i)
      MODEL2_HISTORY="$OPTARG"
      ;;
    b)
      BATCH_SIZE="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$MODEL1" ] || [ -z "$MODEL2" ] || [ -z "$TEST_DATA" ]; then
    echo "Error: Required arguments missing"
    echo "Usage: $0 -1 <model1_path> -2 <model2_path> -t <test_data_path> [-o <output_dir>] [-n <model1_name>] [-m <model2_name>] [-h <model1_history>] [-i <model2_history>] [-b <batch_size>]"
    exit 1
fi

# Setup working directory
WDPATH=/scratch/$USER/model_comparison.$SLURM_JOBID
rm -rf ${WDPATH}
mkdir -p ${WDPATH}
cd ${WDPATH}

# Copy necessary files
echo "Setting up working directory at ${WDPATH}"
mkdir -p ${WDPATH}/src
cp -r $SLURM_SUBMIT_DIR/src/*.py ${WDPATH}/src/
cp -r $SLURM_SUBMIT_DIR/src/lib_*.py ${WDPATH}/src/

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Setup environment
module purge
module load anaconda
source activate keras-jax
conda list

# Set environment variables to configure TensorFlow and JAX
export TF_CPP_MIN_LOG_LEVEL=3
export KERAS_BACKEND=jax
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/apps/languages/cuda/11.7.0"
# Limit JAX to use only one GPU
export CUDA_VISIBLE_DEVICES=0

# Construct the command with history paths if provided
CMD="python src/model_comparison.py --model1 $MODEL1 --model2 $MODEL2 --test_data $TEST_DATA --output $OUTPUT_DIR --model1_name $MODEL1_NAME --model2_name $MODEL2_NAME --batch_size $BATCH_SIZE"

if [ ! -z "$MODEL1_HISTORY" ]; then
    CMD="$CMD --model1_history $MODEL1_HISTORY"
fi

if [ ! -z "$MODEL2_HISTORY" ]; then
    CMD="$CMD --model2_history $MODEL2_HISTORY"
fi

# Run the model comparison
echo "Running model comparison..."
echo "Command: $CMD"
start="$(date +%s)"

time conda run -n keras-jax $CMD

stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Model Comparison $SLURM_JOBID Job-Time $finish seconds

# Copy results back to the submit directory
echo "Copying results to ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}"
mkdir -p ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}
cp -r ${OUTPUT_DIR}/* ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}/

echo "Model comparison completed successfully!"
echo End Time is `date` 