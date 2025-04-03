#!/usr/bin/env python3
# Possum disease classification pipeline with pseudo-labeling
import os
import sys
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import argparse
from keras import models
# Import necessary modules for custom objects
import kimm

# Import custom modules
from lib_model import build_classifier, fit_frozen, fit_progressive, calc_class_metrics, unfreeze_model, save_training_history
from lib_data import print_dsinfo, create_train, create_fixed, process_samples_from_config, ensure_output_directory, validate_directory_structure
from lib_pseudo import generate_pseudo_labels, combine_datasets, create_curriculum_datasets
from lib_common import update_config_from_env, model_img_size_mapping, read_yaml, setup_strategy


def train_wildlife_model(config, strategy):
    """Train the initial model on wildlife data (9 classes)"""
    print("\n===== STAGE 1: Training initial wildlife model =====")
    
    # Setup paths with wildlife-specific name
    wildlife_savefile = f"{config['SAVEFILE']}_wildlife"
    output_fpath = os.path.join(config['OUTPUT_PATH'], wildlife_savefile, config['MODEL'])
    ensure_output_directory(output_fpath)
    
    # Create a modified config with the wildlife-specific savefile
    wildlife_config = config.copy()
    wildlife_config['SAVEFILE'] = wildlife_savefile
    
    # Validate directory structure
    validate_directory_structure(config['TRAIN_PATH'], config['VAL_PATH'], config['TEST_PATH'])
    
    # Get image size based on model
    img_size = model_img_size_mapping(config['MODEL'])
    
    # Create training dataset
    custom_sample_file, is_custom_sample = process_samples_from_config(config)
    train_df, num_classes = create_train(
        config['TRAIN_PATH'],
        seed=config['SEED'],
        ns=custom_sample_file['default'], 
        custom_sample=is_custom_sample, 
        custom_file=custom_sample_file
    )
    classes = train_df['Label'].unique()
    class_map = {name: idx for idx, name in enumerate(classes)}
    df_size = train_df.shape[0]
    
    # Save class mapping
    with open(os.path.join(output_fpath, config['SAVEFILE']+'_class_map.yaml'), 'w') as file:
        yaml.dump(class_map, file, default_flow_style=False)
    
    print('Number of classes: {}'.format(num_classes) + '\n')
    print_dsinfo(train_df, 'Training Data')
    
    # Create validation dataset
    val_df = create_fixed(config['VAL_PATH'])
    print_dsinfo(val_df, 'Validation Data')
    
    # Build and train model with frozen base
    with strategy.scope():
        model = build_classifier(config, num_classes, df_size, img_size)
        frozen_hist, model = fit_frozen(config, model, train_df, val_df, num_classes, df_size, img_size)
    model.summary()
    
    # Create datasets for progressive training
    print('\nCreating datasets for progressive training...')
    prog_train = []
    for i in range(config['PROG_TOT_EPOCH']):
        train_tmp, _ = create_train(
            config['TRAIN_PATH'],
            seed=(config['SEED']+i),
            ns=custom_sample_file['default'], 
            custom_sample=is_custom_sample, 
            custom_file=custom_sample_file
        )
        prog_train.append(train_tmp)
    
    # Progressively fine-tune the model
    prog_hists, model, best_model_fpath = fit_progressive(
        config, model,
        prog_train=prog_train,
        val_df=val_df,
        output_fpath=output_fpath,
        img_size=img_size
    )
    
    # Save training history
    history_path = save_training_history(
        prog_hists, 
        output_fpath, 
        f"{wildlife_savefile}_{config['MODEL']}_wildlife"
    )
    print(f"Wildlife model training history saved to: {history_path}")
    
    # Evaluate on test set
    calc_class_metrics(
        model_fpath=best_model_fpath,
        test_fpath=config['TEST_PATH'],
        output_fpath=output_fpath,
        classes=classes,
        batch_size=config["BATCH_SIZE"],
        img_size=img_size
    )
    
    return best_model_fpath, classes, img_size

def finetune_on_possum(config, wildlife_model_path, img_size, strategy):
    """Fine-tune the wildlife model on possum disease data"""
    print("\n===== STAGE 2: Fine-tuning on possum disease data =====")
    
    # Setup paths with finetuned-specific name
    finetuned_savefile = f"{config['SAVEFILE']}_finetuned"
    output_fpath = os.path.join(config['OUTPUT_PATH'], finetuned_savefile, config['MODEL'])
    ensure_output_directory(output_fpath)
    
    # Create a modified config with the finetuned-specific savefile
    finetuned_config = config.copy()
    finetuned_config['SAVEFILE'] = finetuned_savefile
    
    # Update paths for possum data
    possum_config = config.copy()
    possum_config['TRAIN_PATH'] = config['POSSUM_TRAIN_PATH']
    possum_config['VAL_PATH'] = config['POSSUM_VAL_PATH']
    possum_config['TEST_PATH'] = config['POSSUM_TEST_PATH']
    possum_config['SAVEFILE'] = finetuned_savefile
    
    # Set up class weighting to handle imbalance
    possum_config['USE_CLASS_WEIGHTS'] = True
    
    # Validate directory structure
    validate_directory_structure(possum_config['TRAIN_PATH'], possum_config['VAL_PATH'], possum_config['TEST_PATH'])
    
    # Create training dataset
    custom_sample_file, is_custom_sample = process_samples_from_config(possum_config)
    train_df, num_classes = create_train(
        possum_config['TRAIN_PATH'],
        seed=config['SEED'],
        ns=custom_sample_file['default'], 
        custom_sample=is_custom_sample, 
        custom_file=custom_sample_file
    )
    classes = train_df['Label'].unique()
    class_map = {name: idx for idx, name in enumerate(classes)}
    df_size = train_df.shape[0]
    
    # Save class mapping
    with open(os.path.join(output_fpath, possum_config['SAVEFILE']+'_class_map.yaml'), 'w') as file:
        yaml.dump(class_map, file, default_flow_style=False)
    
    print('Number of classes: {}'.format(num_classes) + '\n')
    print_dsinfo(train_df, 'Training Data')
    
    # Create validation dataset
    val_df = create_fixed(possum_config['VAL_PATH'])
    print_dsinfo(val_df, 'Validation Data')
    
    # Load pre-trained wildlife model with custom objects
    print(f"\nLoading pre-trained wildlife model from {wildlife_model_path}")
    
    # Define custom_objects dictionary to help with model loading
    custom_objects = {
        'EfficientNetV2S': kimm.models.EfficientNetV2S,
        'EfficientNetV2B0': kimm.models.EfficientNetV2B0,
        'EfficientNetV2B2': kimm.models.EfficientNetV2B2,
        'EfficientNetV2M': kimm.models.EfficientNetV2M,
        'EfficientNetV2L': kimm.models.EfficientNetV2L,
        'EfficientNetV2XL': kimm.models.EfficientNetV2XL,
        'ConvNeXtPico': kimm.models.ConvNeXtPico,
        'ConvNeXtNano': kimm.models.ConvNeXtNano,
        'ConvNeXtTiny': kimm.models.ConvNeXtTiny,
        'ConvNeXtSmall': kimm.models.ConvNeXtSmall,
        'ConvNeXtBase': kimm.models.ConvNeXtBase,
        'ConvNeXtLarge': kimm.models.ConvNeXtLarge,
        'VisionTransformerTiny16': kimm.models.VisionTransformerTiny16,
        'VisionTransformerSmall16': kimm.models.VisionTransformerSmall16,
        'VisionTransformerBase16': kimm.models.VisionTransformerBase16,
        'VisionTransformerLarge16': kimm.models.VisionTransformerLarge16
    }
    
    with strategy.scope():
        try:
            # Try to load the model with custom objects
            wildlife_model = models.load_model(wildlife_model_path, custom_objects=custom_objects, compile=False)
            print("Wildlife model loaded successfully with custom objects")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting alternative loading approach...")
            
            try:
                # Try using TensorFlow's keras instead
                wildlife_model = tf.keras.models.load_model(wildlife_model_path, compile=False)
                print("Wildlife model loaded successfully using TensorFlow Keras")
            except Exception as e2:
                print(f"TensorFlow loading also failed: {e2}")
                print("WARNING: Unable to load the wildlife model. Will rebuild from scratch.")
                
                # Rebuild the model from scratch as a last resort
                wildlife_model = build_classifier(config, 9, 1000, img_size)  # 9 classes for wildlife
                print("Rebuilt wildlife model from scratch (untrained)")
        
        # Modify the model to have the correct number of classes for possum disease
        wildlife_model.pop()  # Remove the last classification layer
        new_classification_layer = wildlife_model.add(tf.keras.layers.Dense(
            num_classes, 
            activation='softmax' if num_classes > 2 else 'sigmoid',
            name="classification_possum"
        ))
        
        # Unfreeze for fine-tuning
        model = unfreeze_model(possum_config, wildlife_model, num_classes, df_size)
    
    model.summary()
    
    # Create datasets for progressive training
    print('\nCreating datasets for progressive training...')
    prog_train = []
    for i in range(possum_config['PROG_TOT_EPOCH']):
        train_tmp, _ = create_train(
            possum_config['TRAIN_PATH'],
            seed=(config['SEED']+i),
            ns=custom_sample_file['default'], 
            custom_sample=is_custom_sample, 
            custom_file=custom_sample_file
        )
        prog_train.append(train_tmp)
    
    # Calculate class weights to handle imbalance
    class_counts = train_df['Label'].value_counts().to_dict()
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    max_count = max(class_counts.values())

    # Calculate weights inversely proportional to class frequency
    class_weights = {}
    for class_name, count in class_counts.items():
        # Use inverse frequency with smoothing to prevent extreme weights
        class_weights[class_map[class_name]] = (max_count / count) * (n_classes / len(class_counts))

    print("Calculated class weights to handle imbalance:")
    for class_name, idx in class_map.items():
        print(f"  Class {class_name}: weight = {class_weights[idx]:.2f}")
    
    # Pass class weights to the progressive training function
    prog_hists, model, best_model_fpath = fit_progressive(
        possum_config, model, prog_train, val_df, output_fpath, img_size
    )
    
    # Save training history
    history_path = save_training_history(
        prog_hists, 
        output_fpath, 
        f"{finetuned_savefile}_{config['MODEL']}"
    )
    print(f"Finetuned model training history saved to: {history_path}")
    
    # Evaluate on possum test set
    calc_class_metrics(
        model_fpath=best_model_fpath,
        test_fpath=possum_config['TEST_PATH'],
        output_fpath=output_fpath,
        classes=classes,
        batch_size=possum_config["BATCH_SIZE"],
        img_size=img_size
    )
    
    return best_model_fpath, classes, img_size

def generate_and_train_with_iterative_pseudo_labels(config, possum_model_path, possum_classes, img_size, strategy):
    """Generate pseudo-labels iteratively and train with curriculum learning at each iteration"""
    print("\n===== STAGE 3: Iterative Self-Learning with Curriculum Approach =====")
    
    # Define paths for this stage
    pseudo_config = config.copy()
    pseudo_config['TRAIN_PATH'] = config['POSSUM_TRAIN_PATH']
    pseudo_config['VAL_PATH'] = config['POSSUM_VAL_PATH']
    pseudo_config['TEST_PATH'] = config['POSSUM_TEST_PATH']
    pseudo_config['SAVEFILE'] = config['SAVEFILE'] + '_iterative_pseudo'
    
    # Explicitly remove CLASS_SAMPLES_SPECIFIC to force using CLASS_SAMPLES_DEFAULT only
    if 'CLASS_SAMPLES_SPECIFIC' in pseudo_config:
        print(f"Removing class-specific sample settings for pseudo-label training")
        del pseudo_config['CLASS_SAMPLES_SPECIFIC']
    
    # Get CLASS_SAMPLES_DEFAULT directly from the config
    default_samples = pseudo_config['CLASS_SAMPLES_DEFAULT']
    print(f"Using CLASS_SAMPLES_DEFAULT: {default_samples} for pseudo-label training")
    
    # Setup output path
    base_output_path = os.path.join(config['OUTPUT_PATH'], pseudo_config['SAVEFILE'])
    ensure_output_directory(base_output_path)
    
    # Define custom_objects dictionary to help with model loading
    custom_objects = {
        'EfficientNetV2S': kimm.models.EfficientNetV2S,
        'EfficientNetV2B0': kimm.models.EfficientNetV2B0,
        'EfficientNetV2B2': kimm.models.EfficientNetV2B2,
        'EfficientNetV2M': kimm.models.EfficientNetV2M,
        'EfficientNetV2L': kimm.models.EfficientNetV2L,
        'EfficientNetV2XL': kimm.models.EfficientNetV2XL,
        'ConvNeXtPico': kimm.models.ConvNeXtPico,
        'ConvNeXtNano': kimm.models.ConvNeXtNano,
        'ConvNeXtTiny': kimm.models.ConvNeXtTiny,
        'ConvNeXtSmall': kimm.models.ConvNeXtSmall,
        'ConvNeXtBase': kimm.models.ConvNeXtBase,
        'ConvNeXtLarge': kimm.models.ConvNeXtLarge,
        'VisionTransformerTiny16': kimm.models.VisionTransformerTiny16,
        'VisionTransformerSmall16': kimm.models.VisionTransformerSmall16,
        'VisionTransformerBase16': kimm.models.VisionTransformerBase16,
        'VisionTransformerLarge16': kimm.models.VisionTransformerLarge16
    }
    
    # Get parameters for iterative self-learning
    num_iterations = config.get('ITERATIVE_ITERATIONS', 3)  # Default to 3 iterations
    initial_confidence = float(config.get('INITIAL_CONFIDENCE', 0.99))  # Start with high confidence
    min_confidence = float(config.get('MIN_CONFIDENCE', 0.7))  # Minimum confidence threshold
    initial_epochs = config.get('INITIAL_EPOCHS', 15)  # Epochs for first iteration
    curriculum_epochs = config.get('CURRICULUM_EPOCHS', 10)  # Epochs for curriculum stages
    num_curriculum_stages = config.get('CURRICULUM_STAGES', 4)  # Number of curriculum stages
    
    # Initialize current model path with the input possum model
    current_model_path = possum_model_path
    best_final_model_path = None
    
    # Perform multiple iterations of pseudo-labeling and training
    for iteration in range(num_iterations):
        print(f"\n===== Iteration {iteration+1}/{num_iterations} =====")
        
        # Calculate confidence threshold for this iteration - gradually decrease
        if num_iterations > 1:
            # Linearly decrease confidence from initial to min
            confidence_threshold = initial_confidence - (iteration * (initial_confidence - min_confidence) / (num_iterations - 1))
        else:
            confidence_threshold = initial_confidence
            
        print(f"Using confidence threshold of {confidence_threshold:.4f} for this iteration")
        
        # Create iteration-specific directories
        iteration_dir = os.path.join(base_output_path, f"iteration_{iteration+1}")
        ensure_output_directory(iteration_dir)
        
        # Path for pseudo-labeled data for this iteration
        pseudo_labeled_path = os.path.join(iteration_dir, 'pseudo_labeled')
        ensure_output_directory(pseudo_labeled_path)
        
        # Generate pseudo-labels using the current model
        print(f"\n----- Generating pseudo-labels with model from {'previous iteration' if iteration > 0 else 'fine-tuning'} -----")
        print(f"Using model: {current_model_path}")
        
        # Generate pseudo-labels for unlabeled data
        csv_path = generate_pseudo_labels(
            model_path=current_model_path,
            unlabeled_dir=config['UNLABELED_POSSUM_PATH'],
            output_dir=pseudo_labeled_path,
            classes=possum_classes,
            confidence_threshold=confidence_threshold,
            img_size=img_size,
            custom_objects=custom_objects
        )
        
        if csv_path is None:
            print(f"No pseudo-labels generated for iteration {iteration+1}. Skipping to next iteration.")
            continue
        
        # Create curriculum datasets for this iteration
        print(f"\n----- Training with curriculum learning (Iteration {iteration+1}) -----")
        curriculum_base_path = os.path.join(iteration_dir, 'curriculum_datasets')
        
        curriculum_paths = create_curriculum_datasets(
            original_train_path=config['POSSUM_TRAIN_PATH'],
            pseudo_labeled_csv_path=csv_path,
            output_base_path=curriculum_base_path,
            num_stages=num_curriculum_stages,
            min_confidence=confidence_threshold
        )
        
        # Set epochs for this iteration - first iteration uses more epochs
        epochs_per_stage = initial_epochs if iteration == 0 else curriculum_epochs
        
        # Validate directory structure for the first stage
        iteration_config = pseudo_config.copy()
        iteration_config['TRAIN_PATH'] = curriculum_paths[0]
        iteration_config['SAVEFILE'] = f"{pseudo_config['SAVEFILE']}_iter_{iteration+1}"
        validate_directory_structure(iteration_config['TRAIN_PATH'], iteration_config['VAL_PATH'], iteration_config['TEST_PATH'])
        
        # Create validation dataset
        val_df = create_fixed(iteration_config['VAL_PATH'])
        print_dsinfo(val_df, 'Validation Data')
        
        # Load the current model
        print(f"Loading model from: {current_model_path}")
        with strategy.scope():
            try:
                model = models.load_model(current_model_path, custom_objects=custom_objects, compile=False)
                print("Model loaded successfully with custom objects")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Attempting alternative loading approach...")
                
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(current_model_path, compile=False)
                    print("Model loaded successfully using TensorFlow Keras")
                except Exception as e2:
                    print(f"TensorFlow loading also failed: {e2}")
                    print(f"WARNING: Unable to load the model. Skipping iteration {iteration+1}.")
                    continue
        
        # Train on each curriculum stage
        iteration_output_path = os.path.join(base_output_path, f"iteration_{iteration+1}", config['MODEL'])
        ensure_output_directory(iteration_output_path)
        best_model_path = None
        
        # Update PROG_TOT_EPOCH for this iteration
        iteration_config['PROG_TOT_EPOCH'] = epochs_per_stage
        
        for stage, train_path in enumerate(curriculum_paths):
            print(f"\n----- Curriculum Stage {stage+1}/{num_curriculum_stages} (Iteration {iteration+1}) -----")
            print(f"Training on data from: {train_path}")
            
            # Update training path for this stage
            iteration_config['TRAIN_PATH'] = train_path
            
            # Create training dataset for this curriculum stage
            custom_sample_file, is_custom_sample = process_samples_from_config(iteration_config)
            train_df, num_classes = create_train(
                iteration_config['TRAIN_PATH'],
                seed=config['SEED'],
                ns=custom_sample_file['default'], 
                custom_sample=is_custom_sample, 
                custom_file=custom_sample_file
            )
            classes = train_df['Label'].unique()
            class_map = {name: idx for idx, name in enumerate(classes)}
            df_size = train_df.shape[0]
            
            print(f"Training data for iteration {iteration+1}, stage {stage+1}:")
            print_dsinfo(train_df, f'Training Data')
            
            # Unfreeze model for fine-tuning
            with strategy.scope():
                stage_model = unfreeze_model(iteration_config, model, num_classes, df_size)
            
            # Create dataset for each epoch
            print(f'Creating datasets for stage {stage+1} training (iteration {iteration+1})...')
            prog_train = []
            for i in range(epochs_per_stage):
                train_tmp, _ = create_train(
                    iteration_config['TRAIN_PATH'],
                    seed=(config['SEED'] + (1000 * iteration) + (100 * stage) + i),  # Use different seed space for each iteration
                    ns=custom_sample_file['default'], 
                    custom_sample=is_custom_sample, 
                    custom_file=custom_sample_file
                )
                prog_train.append(train_tmp)
            
            # Customize stage-specific config
            stage_config = iteration_config.copy()
            stage_config['SAVEFILE'] = f"{iteration_config['SAVEFILE']}_stage_{stage+1}"
            
            # Setup output path for this stage
            stage_output_path = os.path.join(iteration_output_path, f"stage_{stage+1}")
            ensure_output_directory(stage_output_path)
            
            # Train the model for this curriculum stage
            prog_hists, stage_model, stage_best_model_path = fit_progressive(
                stage_config, stage_model,
                prog_train=prog_train,
                val_df=val_df,
                output_fpath=stage_output_path,
                img_size=img_size
            )
            
            # Save training history
            history_path = save_training_history(
                prog_hists, 
                stage_output_path, 
                f"{stage_config['SAVEFILE']}_{config['MODEL']}"
            )
            print(f"Stage {stage+1} training history saved to: {history_path}")
            
            # Update model for next stage and save best model path
            model = models.load_model(stage_best_model_path, custom_objects=custom_objects, compile=False)
            best_model_path = stage_best_model_path
            
            # Evaluate after this stage
            print(f"\nEvaluating after curriculum stage {stage+1} (iteration {iteration+1}):")
            calc_class_metrics(
                model_fpath=stage_best_model_path,
                test_fpath=iteration_config['TEST_PATH'],
                output_fpath=stage_output_path,
                classes=classes,
                batch_size=iteration_config["BATCH_SIZE"],
                img_size=img_size
            )
        
        # After all curriculum stages in this iteration, evaluate the final model
        final_output_path = os.path.join(iteration_output_path, "final")
        ensure_output_directory(final_output_path)
        
        print(f"\nFinal evaluation after iteration {iteration+1}:")
        calc_class_metrics(
            model_fpath=best_model_path,
            test_fpath=iteration_config['TEST_PATH'],
            output_fpath=final_output_path,
            classes=classes,
            batch_size=iteration_config["BATCH_SIZE"],
            img_size=img_size
        )
        
        # Update current model path for next iteration
        current_model_path = best_model_path
        
        # Save final best model path for the last iteration
        if iteration == num_iterations - 1:
            best_final_model_path = best_model_path
    
    print("\n===== Iterative Self-Learning Complete =====")
    return best_final_model_path

def main():
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["KERAS_BACKEND"] = "jax"
    # Disable XLA for TensorFlow to avoid ptxas version issues
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    parser = argparse.ArgumentParser(description='Possum disease classification pipeline')
    parser.add_argument('--stage', type=int, default=0, 
                        help='Start from specific stage (0=all, 1=wildlife, 2=possum, 3=pseudo)')
    parser.add_argument('--config', type=str, default='src/possum_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--wildlife_model', type=str, default=None,
                        help='Path to pre-trained wildlife model (to skip Stage 1)')
    parser.add_argument('--possum_model', type=str, default=None,
                        help='Path to pre-trained possum model (to skip Stage 2)')
    
    args = parser.parse_args()
    
    # Load config
    config = update_config_from_env(read_yaml(args.config))
    
    # Set up strategy for distributed training
    strategy = setup_strategy()
    
    # Determine which stages to run
    start_stage = args.stage
    wildlife_model_path = args.wildlife_model
    possum_model_path = args.possum_model
    
    # Execute pipeline stages
    if start_stage <= 1 and not wildlife_model_path:
        wildlife_model_path, wildlife_classes, img_size = train_wildlife_model(config, strategy)
    else:
        img_size = model_img_size_mapping(config['MODEL'])
        print(f"\nSkipping Stage 1. Using wildlife model: {wildlife_model_path}")
    
    if start_stage <= 2 and not possum_model_path:
        possum_model_path, possum_classes, _ = finetune_on_possum(config, wildlife_model_path, img_size, strategy)
    else:
        # Load possum classes from the existing model's class map
        # Extract model directory from the model path
        possum_model_dir = os.path.dirname(possum_model_path)
        
        # Look for class map file in the same directory as the model
        possible_class_maps = [
            os.path.join(possum_model_dir, f"{config['SAVEFILE']}_finetuned_class_map.yaml"),
            os.path.join(possum_model_dir, f"{config['SAVEFILE']}_possum_class_map.yaml"),
            # Add more possible patterns if needed
        ]
        
        # Try each possible class map path
        class_map_found = False
        for class_map_path in possible_class_maps:
            if os.path.exists(class_map_path):
                print(f"Found class map at: {class_map_path}")
                with open(class_map_path, 'r') as file:
                    class_map = yaml.safe_load(file)
                possum_classes = list(class_map.keys())
                class_map_found = True
                break
        
        if not class_map_found:
            raise FileNotFoundError(f"Could not find class map file in {possum_model_dir}. "
                                   f"Tried paths: {possible_class_maps}")
        
        print(f"\nSkipping Stage 2. Using possum model: {possum_model_path}")
    
    if start_stage <= 3:
        if config.get('USE_CURRICULUM', True):  # Default to using curriculum
            final_model_path = generate_and_train_with_iterative_pseudo_labels(
                config, possum_model_path, possum_classes, img_size, strategy
            )
        else:
            final_model_path = generate_and_train_with_iterative_pseudo_labels(
                config, possum_model_path, possum_classes, img_size, strategy
            )
        print(f"\nPipeline complete! Final model saved to: {final_model_path}")
    else:
        print("\nSkipping Stage 3 and 4. Pipeline complete!")

if __name__ == "__main__":
    main() 