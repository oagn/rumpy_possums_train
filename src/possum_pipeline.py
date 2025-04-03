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
from lib_pseudo import generate_pseudo_labels, combine_datasets
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
    finetuned_savefile = f"{config['SAVEFILE']}_frozen_finetuned"  # Changed to include "frozen" in the name
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
    
    with strategy.scope():
        # Check if we're starting from a wildlife model or from scratch with ImageNet weights
        if wildlife_model_path and os.path.exists(wildlife_model_path):
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
                    print("WARNING: Unable to load the wildlife model. Will rebuild from scratch with ImageNet weights.")
                    
                    # Fall back to ImageNet weights
                    wildlife_model_path = None
            
            if wildlife_model_path:  # If wildlife model was successfully loaded
                # Modify the model to have the correct number of classes for possum disease
                wildlife_model.pop()  # Remove the last classification layer
                model = wildlife_model.add(tf.keras.layers.Dense(
                    num_classes, 
                    activation='softmax' if num_classes > 2 else 'sigmoid',
                    name="classification_possum"
                ))
            else:
                # Wildlife model loading failed, fallback to ImageNet weights
                print("\nStarting from scratch with ImageNet weights")
                model = build_classifier(possum_config, num_classes, df_size, img_size)
        else:
            # Starting from scratch with ImageNet weights
            print("\nStarting fine-tuning from ImageNet weights")
            model = build_classifier(possum_config, num_classes, df_size, img_size)
        
        # First, train with frozen base model to stabilize the new classification layer
        print("\n===== STEP 2.1: Frozen training to stabilize classification layer =====")
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
            
        # Apply fit_frozen to stabilize the new classification layer
        frozen_hist, model = fit_frozen(possum_config, model, train_df, val_df, num_classes, df_size, img_size)
        
        # Save the frozen trained model
        frozen_model_path = os.path.join(output_fpath, f"{possum_config['SAVEFILE']}_frozen_{possum_config['MODEL']}.keras")
        model.save(frozen_model_path)
        print(f"Frozen trained model saved to: {frozen_model_path}")
        
        # Continue with progressive fine-tuning (model is already unfrozen by fit_frozen)
        print("\n===== STEP 2.2: Progressive fine-tuning =====")
    
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
    
    # Pass class weights to the progressive training function
    prog_hists, model, best_model_fpath = fit_progressive(
        possum_config, model,
        prog_train=prog_train,
        val_df=val_df,
        output_fpath=output_fpath,
        img_size=img_size,
        class_weights=class_weights
    )
    
    # Save training history
    history_path = save_training_history(
        prog_hists, 
        output_fpath, 
        f"{possum_config['SAVEFILE']}_{possum_config['MODEL']}"
    )
    print(f"Fine-tuning history saved to: {history_path}")
    
    # Evaluate on test set
    calc_class_metrics(
        model_fpath=best_model_fpath,
        test_fpath=possum_config['TEST_PATH'],
        output_fpath=output_fpath,
        classes=classes,
        batch_size=possum_config["BATCH_SIZE"],
        img_size=img_size
    )
    
    return best_model_fpath

def generate_and_train_with_pseudo_labels(config, possum_model_path, possum_classes, img_size, strategy):
    """Generate pseudo-labels for unlabeled possum data and train on combined dataset"""
    print("\n===== STAGE 3: Generating pseudo-labels for unlabeled data =====")
    
    # Define paths for this stage
    pseudo_config = config.copy()
    pseudo_config['TRAIN_PATH'] = config['POSSUM_TRAIN_PATH']
    pseudo_config['VAL_PATH'] = config['POSSUM_VAL_PATH']
    pseudo_config['TEST_PATH'] = config['POSSUM_TEST_PATH']
    pseudo_config['SAVEFILE'] = config['SAVEFILE'] + '_pseudo'
    
    # Explicitly remove CLASS_SAMPLES_SPECIFIC to force using CLASS_SAMPLES_DEFAULT only
    if 'CLASS_SAMPLES_SPECIFIC' in pseudo_config:
        print(f"Removing class-specific sample settings for pseudo-label training")
        del pseudo_config['CLASS_SAMPLES_SPECIFIC']
    
    # Get CLASS_SAMPLES_DEFAULT directly from the config
    default_samples = pseudo_config['CLASS_SAMPLES_DEFAULT']
    print(f"Using CLASS_SAMPLES_DEFAULT: {default_samples} for pseudo-label training")
    
    # Setup output path
    output_fpath = os.path.join(config['OUTPUT_PATH'], pseudo_config['SAVEFILE'], config['MODEL'])
    ensure_output_directory(output_fpath)
    
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
    
    # Generate pseudo-labels for unlabeled data
    unlabeled_path = config['UNLABELED_POSSUM_PATH']
    pseudo_labeled_path = os.path.join(config['OUTPUT_PATH'], 'pseudo_labeled')
    ensure_output_directory(pseudo_labeled_path)
    
    confidence_threshold = float(config.get('PSEUDO_LABEL_CONFIDENCE', 0.7))
    
    # Load the possum model
    csv_path = generate_pseudo_labels(
        model_path=possum_model_path,
        unlabeled_dir=unlabeled_path,
        output_dir=pseudo_labeled_path,
        classes=possum_classes,
        confidence_threshold=confidence_threshold,
        img_size=img_size,
        custom_objects=custom_objects
    )
    
    if csv_path is None:
        print("No pseudo-labels generated. Cannot proceed with Stage 4.")
        return None
    
    print("\n===== STAGE 4: Training on combined labeled + pseudo-labeled data =====")
    
    # Combine original labeled data with pseudo-labeled data
    combined_train_path = os.path.join(config['OUTPUT_PATH'], 'combined_train')
    pseudo_config['TRAIN_PATH'] = combined_train_path
    
    # Create combined dataset using the CSV file instead of copying files again
    combine_datasets(
        original_train_path=config['POSSUM_TRAIN_PATH'],
        pseudo_labeled_csv_path=csv_path,
        output_path=combined_train_path
    )
    
    # Validate directory structure
    validate_directory_structure(pseudo_config['TRAIN_PATH'], pseudo_config['VAL_PATH'], pseudo_config['TEST_PATH'])
    
    # Create training dataset from combined data
    custom_sample_file, is_custom_sample = process_samples_from_config(pseudo_config)
    train_df, num_classes = create_train(
        pseudo_config['TRAIN_PATH'],
        seed=config['SEED'],
        ns=custom_sample_file['default'], 
        custom_sample=is_custom_sample, 
        custom_file=custom_sample_file
    )
    classes = train_df['Label'].unique()
    class_map = {name: idx for idx, name in enumerate(classes)}
    df_size = train_df.shape[0]
    
    # Save class mapping
    with open(os.path.join(output_fpath, pseudo_config['SAVEFILE']+'_class_map.yaml'), 'w') as file:
        yaml.dump(class_map, file, default_flow_style=False)
    
    print('Number of classes: {}'.format(num_classes) + '\n')
    print_dsinfo(train_df, 'Training Data (Combined Original + Pseudo-labeled)')
    
    # Create validation dataset
    val_df = create_fixed(pseudo_config['VAL_PATH'])
    print_dsinfo(val_df, 'Validation Data')
    
    # Load fine-tuned possum model
    print(f"\nLoading fine-tuned possum model from {possum_model_path}")
    with strategy.scope():
        try:
            # Try to load the model with custom objects
            model = models.load_model(possum_model_path, custom_objects=custom_objects, compile=False)
            print("Possum model loaded successfully with custom objects")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting alternative loading approach...")
            
            try:
                # Try using TensorFlow's keras instead
                import tensorflow as tf
                model = tf.keras.models.load_model(possum_model_path, compile=False)
                print("Possum model loaded successfully using TensorFlow Keras")
            except Exception as e2:
                print(f"TensorFlow loading also failed: {e2}")
                print("WARNING: Unable to load the possum model. Cannot proceed with training.")
                return None
        
        # Unfreeze for fine-tuning on combined data
        model = unfreeze_model(pseudo_config, model, num_classes, df_size)
    
    model.summary()
    
    # Create datasets for progressive training
    print('\nCreating datasets for progressive training...')
    prog_train = []
    for i in range(pseudo_config['PROG_TOT_EPOCH']):
        train_tmp, _ = create_train(
            pseudo_config['TRAIN_PATH'],
            seed=(config['SEED']+i),
            ns=custom_sample_file['default'], 
            custom_sample=is_custom_sample, 
            custom_file=custom_sample_file
        )
        prog_train.append(train_tmp)
    
    # Progressively fine-tune the model
    prog_hists, model, best_model_fpath = fit_progressive(
        pseudo_config, model,
        prog_train=prog_train,
        val_df=val_df,
        output_fpath=output_fpath,
        img_size=img_size
    )
    
    # Save training history
    history_path = save_training_history(
        prog_hists, 
        output_fpath, 
        f"{pseudo_config['SAVEFILE']}_{config['MODEL']}"
    )
    print(f"Pseudo trained model training history saved to: {history_path}")

    # Evaluate on possum test set
    calc_class_metrics(
        model_fpath=best_model_fpath,
        test_fpath=pseudo_config['TEST_PATH'],
        output_fpath=output_fpath,
        classes=classes,
        batch_size=pseudo_config["BATCH_SIZE"],
        img_size=img_size
    )
    
    return best_model_fpath

def main():
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Possum Disease Classification Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3], 
                        help='Starting stage (0=all, 1=wildlife, 2=finetune, 3=pseudo)')
    parser.add_argument('--wildlife_model', type=str, default=None, 
                        help='Path to pre-trained wildlife model for stage 2')
    parser.add_argument('--possum_model', type=str, default=None, 
                        help='Path to fine-tuned possum model for stage 3')
    args = parser.parse_args()
    
    # Load configuration
    config = read_yaml(args.config)
    config = update_config_from_env(config)
    strategy = setup_strategy()
    
    # Execute pipeline stages based on argument
    wildlife_model_path = None
    possum_model_path = None
    classes = None
    img_size = model_img_size_mapping(config['MODEL'])
    
    # Stage 1: Train wildlife model
    if args.stage <= 1:
        wildlife_model_path, classes, img_size = train_wildlife_model(config, strategy)
    else:
        # Get wildlife model path from args if provided and at stage 2
        if args.stage == 2 and args.wildlife_model:
            wildlife_model_path = args.wildlife_model
            
    # Stage 2: Fine-tune on possum disease data
    if args.stage <= 2:
        # If we're starting at stage 2 and no wildlife model is provided, we'll use ImageNet weights
        possum_model_path = finetune_on_possum(config, wildlife_model_path, img_size, strategy)
    else:
        # Get possum model path from args if provided and at stage 3
        if args.stage == 3 and args.possum_model:
            possum_model_path = args.possum_model
    
    # Stage 3: Generate and train with pseudo-labels
    if args.stage <= 3:
        if possum_model_path is None and args.possum_model:
            possum_model_path = args.possum_model
            
        if possum_model_path:
            # Since we don't have possum_classes from earlier stages, load them from the model directory
            try:
                model_dir = os.path.dirname(possum_model_path)
                class_map_file = os.path.join(model_dir, config['SAVEFILE'] + '_class_map.yaml')
                with open(class_map_file, 'r') as file:
                    class_map = yaml.safe_load(file)
                possum_classes = list(class_map.keys())
            except Exception as e:
                print(f"Error loading class map: {e}")
                print("Using default classes: ['0', '12', '99']")
                possum_classes = ['0', '12', '99']
                
            generate_and_train_with_pseudo_labels(config, possum_model_path, possum_classes, img_size, strategy)
        else:
            print("ERROR: Possum model path is required for pseudo-labeling stage.")
            sys.exit(1)
    else:
        print("\nSkipping Stage 3 and 4. Pipeline complete!")

if __name__ == "__main__":
    main() 