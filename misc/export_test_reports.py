#!/usr/bin/env python3
# Script to directly export test set classification reports for each stage
import os
import sys
import argparse
import yaml
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import kimm
from keras import models
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Import custom modules (adjust paths as needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib_model import calc_class_metrics
from lib_common import model_img_size_mapping, read_yaml, update_config_from_env


def find_models(base_dir, pattern="**/*saved_model*"):
    """Find all saved models in the given directory."""
    models_found = []
    for model_path in glob.glob(os.path.join(base_dir, pattern), recursive=True):
        if os.path.isdir(model_path) and any(f.endswith('.pb') for f in os.listdir(model_path)):
            models_found.append(model_path)
    return models_found


def extract_stage_info(model_path):
    """Extract stage and iteration information from model path."""
    path_parts = model_path.split(os.sep)
    
    # Default values
    iteration = "unknown"
    stage = "unknown"
    
    # Try to extract iteration and stage information
    for part in path_parts:
        if part.startswith("iteration_"):
            iteration = part.split("_")[1]
        elif part.startswith("stage_"):
            stage = part.split("_")[1]
        elif part == "wildlife":
            stage = "wildlife"
        elif part == "finetuned":
            stage = "possum"
    
    return iteration, stage


def evaluate_model(model_path, test_path, output_dir, batch_size=32, 
                  config_path=None, custom_objects=None):
    """Evaluate a model on the test set and save the classification report."""
    
    print(f"\n===== Evaluating model: {model_path} =====")
    
    # Extract iteration and stage information
    iteration, stage = extract_stage_info(model_path)
    print(f"Identified as iteration {iteration}, stage {stage}")
    
    # Determine img_size from config or use default
    img_size = 224  # Default
    if config_path:
        try:
            config = update_config_from_env(read_yaml(config_path))
            img_size = model_img_size_mapping(config['MODEL'])
            print(f"Using image size: {img_size} from config")
        except Exception as e:
            print(f"Warning: Could not get image size from config: {e}")
            print(f"Using default image size: {img_size}")
    
    # Define custom_objects dictionary if not provided
    if custom_objects is None:
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
    
    # Create output directory for this evaluation
    stage_iteration_dir = os.path.join(output_dir, f"iteration_{iteration}_stage_{stage}")
    os.makedirs(stage_iteration_dir, exist_ok=True)
    
    # Try to load model
    try:
        model = models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting alternative loading approach...")
        
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully using TensorFlow Keras")
        except Exception as e2:
            print(f"TensorFlow loading also failed: {e2}")
            print(f"ERROR: Unable to load the model from {model_path}.")
            return None
    
    # Get list of classes from test directory
    classes = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    print(f"Found {len(classes)} classes in test set: {classes}")
    
    # Perform evaluation with direct capturing of metrics
    class_map = {name: idx for idx, name in enumerate(classes)}
    inv_class = {v: k for k, v in class_map.items()}
    class_ids = sorted(class_map.keys())
    
    # Initialize y_pred and y_true
    y_pred = []
    y_true = []
    
    # Evaluate each class
    for i, spp_class in enumerate(classes):
        print(f"\nEvaluating class '{spp_class}' ({(i+1)}/{len(classes)})")
        img_generator = tf.keras.utils.image_dataset_from_directory(
            os.path.join(test_path, spp_class),
            labels=None,
            label_mode=None,
            batch_size=batch_size, 
            image_size=(img_size, img_size),
            shuffle=False
        )
        preds = model.predict(img_generator)
        y_pred_tmp = [class_ids[pred.argmax()] for pred in preds]
        y_pred.extend(y_pred_tmp)
        y_true.extend([spp_class] * len(y_pred_tmp))
    
    # Generate classification report
    report = metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)
    report_text = metrics.classification_report(y_true, y_pred, digits=3)
    
    # Print the report
    print("\nClassification report:")
    print(report_text)
    
    # Convert report to DataFrame for export
    report_rows = []
    
    # Add class rows
    for class_name, metrics_dict in report.items():
        if class_name in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        report_rows.append({
            'class': class_name,
            'precision': metrics_dict['precision'],
            'recall': metrics_dict['recall'],
            'f1-score': metrics_dict['f1-score'],
            'support': metrics_dict['support'],
            'iteration': iteration,
            'stage': stage
        })
    
    # Add accuracy, macro avg, and weighted avg
    for avg_type in ('accuracy', 'macro avg', 'weighted avg'):
        if avg_type in report:
            row = {
                'class': avg_type,
                'precision': report[avg_type].get('precision'),
                'recall': report[avg_type].get('recall'),
                'f1-score': report[avg_type]['f1-score'],
                'support': report[avg_type].get('support'),
                'iteration': iteration,
                'stage': stage
            }
            report_rows.append(row)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(report_rows)
    csv_path = os.path.join(stage_iteration_dir, 'classification_report.csv')
    df.to_csv(csv_path, index=False)
    print(f"Classification report saved to: {csv_path}")
    
    # Generate and save confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize="true")
    rcParams.update({'figure.autolayout': True})
    if len(classes) > 20:  # adjust font size with many classes
        font_size = 7 if len(classes) < 35 else 5
        rcParams.update({'font.size': font_size})
    
    plt.figure(figsize=(10, 8))
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, 
        display_labels=class_ids
    )
    cm_display.plot(cmap=plt.cm.Blues, include_values=len(class_ids) < 8, values_format='.2g')
    plt.xticks(rotation=90, ha='center')
    plt.title(f"Iteration {iteration}, Stage {stage}")
    
    cm_path = os.path.join(stage_iteration_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Export test set classification reports for each stage')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Base directory containing model directories from different stages')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to the test dataset')
    parser.add_argument('--output_dir', type=str, default='test_reports',
                        help='Directory to save the classification reports (default: test_reports)')
    parser.add_argument('--config', type=str, default='src/possum_config.yaml',
                        help='Path to configuration file for image size information')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for model predictions (default: 32)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_dir):
        parser.error(f"Model directory not found: {args.model_dir}")
    
    if not os.path.exists(args.test_path):
        parser.error(f"Test path not found: {args.test_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all models
    model_paths = find_models(args.model_dir)
    if not model_paths:
        parser.error(f"No saved models found in {args.model_dir}")
    
    print(f"Found {len(model_paths)} models to evaluate")
    
    # Evaluate each model and collect results
    all_results = []
    for model_path in model_paths:
        try:
            results = evaluate_model(
                model_path=model_path,
                test_path=args.test_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                config_path=args.config
            )
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
    
    # Combine all results into a single DataFrame
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.output_dir, 'all_classification_reports.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"\nAll classification reports combined and saved to: {combined_path}")
    else:
        print("\nNo successful model evaluations to combine.")


if __name__ == "__main__":
    main() 