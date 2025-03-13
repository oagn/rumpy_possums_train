# Possum Disease Classification Pipeline

This repository contains a multi-stage pipeline for possum disease classification using transfer learning and pseudo-labeling techniques.

## Overview

The pipeline implements a four-stage approach to build an effective classifier for possum health status:

1. **Wildlife Model Training**: Train an initial model on 9 wildlife classes (excluding possums) to develop general feature recognition capabilities.
2. **Possum Disease Fine-tuning**: Fine-tune the wildlife model on labeled possum data with 3 health classes (healthy, diseased, occluded).
3. **Pseudo-labeling**: Generate labels for unlabeled possum data to expand the training dataset.
4. **Combined Training**: Train the model on both labeled and pseudo-labeled possum data for optimal performance.

## Data Structure

The pipeline works with the following data organization:

- **Wildlife data** (9 classes):
  ```
  /scratch/c.c1767198/data/possums/data/
  ├── train/           # 4000 images per class
  │   ├── 1/
  │   ├── 2/
  │   ├── 3/
  │   ├── 5/
  │   ├── 6/
  │   ├── 7/
  │   ├── 11/
  │   ├── 13/
  │   └── 17/
  └── test/            # 1000 images per class
      ├── 1/
      └── ...
  ```

- **Possum disease data** (3 classes):
  ```
  /scratch/c.c1767198/data/possums/near_dupe_sorted/
  ├── train/
  │   ├── 0/    # Healthy possums (3177 images)
  │   ├── 12/   # Visible signs of disease (1329 images)
  │   └── 99/   # Occluded possums (1619 images)
  ├── validate/
  └── test/
  ```

- **Unlabeled possum data**:
  ```
  /scratch/c.c1767198/data/possums/unlabeled_possums/
  ```

## Installation and Requirements

This pipeline is designed to run on a SLURM-based High-Performance Computing (HPC) environment with GPU support.

### Requirements
- Python 3.8+
- TensorFlow with JAX backend
- Keras
- Access to SLURM scheduler with GPU nodes

## Running the Pipeline

### SLURM Job Submission

The complete pipeline can be run via the provided SLURM script:

```bash
sbatch src/run_possum_pipeline.sh
```

### Command-line Arguments

For more granular control, you can run specific stages of the pipeline:

```bash
python src/possum_pipeline.py [--stage STAGE] [--config CONFIG_PATH] [--wildlife_model MODEL_PATH] [--possum_model MODEL_PATH]
```

Options:
- `--stage`: Start from a specific stage (0=all, 1=wildlife, 2=possum, 3=pseudo)
- `--config`: Path to configuration file (default: src/possum_config.yaml)
- `--wildlife_model`: Path to pre-trained wildlife model (to skip Stage 1)
- `--possum_model`: Path to pre-trained possum model (to skip Stage 2)

Examples:

```bash
# Run the entire pipeline from start to finish
python src/possum_pipeline.py

# Start from possum fine-tuning with a pre-trained wildlife model
python src/possum_pipeline.py --stage 2 --wildlife_model /path/to/wildlife_model.keras

# Run only pseudo-labeling and final training
python src/possum_pipeline.py --stage 3 --possum_model /path/to/possum_model.keras
```

## Configuration

The pipeline is configured through `src/possum_config.yaml`. Key settings include:

```yaml
# Model architecture and output paths
MODEL: 'ENS'                                       # EfficientNetV2S
SAVEFILE: 'possum_disease'                         # Base filename for saved models
OUTPUT_PATH: '/scratch/c.c1767198/output/rumpy_possum_train'

# Training hyperparameters
LEARNING_RATE: 1e-4
BATCH_SIZE: 16
FROZ_EPOCH: 10                                     # Epochs with frozen base model
PROG_TOT_EPOCH: 30                                 # Total progressive fine-tuning epochs

# Class-specific sample counts
CLASS_SAMPLES_DEFAULT: 4000                        # For wildlife classes
CLASS_SAMPLES_SPECIFIC:                            # For possum disease classes
  - SAMPLES: 3177                                  # Use all available data
    CLASS: "0"                                     # healthy
  - SAMPLES: 1329                                  # Use all available data 
    CLASS: "12"                                    # visible signs of disease
  - SAMPLES: 1619                                  # Use all available data
    CLASS: "99"                                    # occluded

# Pseudo-labeling settings
PSEUDO_LABEL_CONFIDENCE: 0.7                       # Minimum confidence threshold
```

## Handling Class Imbalance

The pipeline addresses class imbalance in the possum disease data through:

1. **Class Weighting**: Uses weights inversely proportional to class frequency in the loss function
2. **Balanced Augmentation**: Applies more augmentations to minority classes
3. **Full Data Utilization**: Uses all available samples from each class

## Pipeline Components

The pipeline consists of the following key files:

- `src/possum_pipeline.py`: Main pipeline orchestration
- `src/lib_model.py`: Model building, training, and evaluation functions
- `src/lib_data.py`: Data loading and preprocessing utilities
- `src/lib_pseudo.py`: Pseudo-labeling implementation
- `src/possum_config.yaml`: Configuration parameters
- `src/run_possum_pipeline.sh`: SLURM job script

## Output and Results

After pipeline execution, the following outputs are generated in the specified output path:

- **Trained Models**: `.keras` files for each stage of the pipeline
- **Class Mappings**: YAML files containing class-to-index mappings
- **Evaluation Metrics**: Confusion matrices and classification reports
- **Training History**: CSV files with training metrics for analysis
- **Pseudo-labeled Data**: Generated in the designated output directory

The final model combines the strength of wildlife feature recognition with specific possum disease characteristics, enhanced by additional pseudo-labeled data.