# Example comparison script structure
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
import kimm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from statsmodels.stats.contingency_tables import mcnemar
import seaborn as sns
import yaml
import argparse
from pathlib import Path

# Import from your existing modules
from lib_data import create_tensorset
from lib_common import model_img_size_mapping

def load_model_safely(model_path):
    """Load a model with proper error handling and custom objects."""
    print(f"Loading model from {model_path}")
    
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
        model = models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully with custom objects")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting alternative loading approach...")
        
        try:
            # Try using TensorFlow's keras instead
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully using TensorFlow Keras")
        except Exception as e2:
            print(f"TensorFlow loading also failed: {e2}")
            raise ValueError(f"Unable to load model from {model_path}")
    
    return model

def get_predictions(model, test_data_path, class_names, img_size, batch_size=16):
    """Get predictions for all images in the test data path."""
    y_true = []
    y_pred = []
    y_scores = []
    
    # Create dataset from the test directory
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_data_path,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False
    )
    
    # Get ground truth labels
    y_true_batches = []
    for images, labels in test_dataset:
        y_true_batches.append(labels.numpy())
    
    if y_true_batches:
        y_true_onehot = np.concatenate(y_true_batches)
        y_true = np.argmax(y_true_onehot, axis=1)
    
    # Get model predictions
    y_scores = model.predict(test_dataset)
    y_pred = np.argmax(y_scores, axis=1)
    
    # Convert indices back to class names
    y_true_names = [class_names[idx] for idx in y_true]
    y_pred_names = [class_names[idx] for idx in y_pred]
    
    return y_true_names, y_pred_names, y_scores

def calculate_metrics(y_true, y_pred, classes):
    """Calculate various classification metrics."""
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Format into nice dataframe
    metrics_data = {
        'Metric': ['Accuracy'],
        'Value': [accuracy]
    }
    
    # Add per-class metrics
    for class_name in classes:
        if class_name in report:
            class_report = report[class_name]
            metrics_data['Metric'].extend([
                f'Precision ({class_name})',
                f'Recall ({class_name})',
                f'F1-score ({class_name})'
            ])
            metrics_data['Value'].extend([
                class_report['precision'],
                class_report['recall'],
                class_report['f1-score']
            ])
    
    # Add macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            metrics_data['Metric'].extend([
                f'{avg_type.capitalize()} Precision',
                f'{avg_type.capitalize()} Recall',
                f'{avg_type.capitalize()} F1-score'
            ])
            metrics_data['Value'].extend([
                report[avg_type]['precision'],
                report[avg_type]['recall'],
                report[avg_type]['f1-score']
            ])
    
    return pd.DataFrame(metrics_data)

def perform_mcnemar_test(y_pred_model1, y_pred_model2, y_true):
    """Perform McNemar's test to check if the difference between models is statistically significant."""
    # Convert to numpy arrays if they're not already
    y_pred_model1 = np.array(y_pred_model1)
    y_pred_model2 = np.array(y_pred_model2)
    y_true = np.array(y_true)
    
    # Create contingency table
    correct_model1 = (y_pred_model1 == y_true)
    correct_model2 = (y_pred_model2 == y_true)
    
    # Contingency table
    b = np.sum(correct_model1 & ~correct_model2)  # model1 correct, model2 wrong
    c = np.sum(~correct_model1 & correct_model2)  # model1 wrong, model2 correct
    
    # McNemar's test statistic and p-value
    statistic = ((b - c)**2) / (b + c) if (b + c) > 0 else 0
    
    # Perform McNemar's test with continuity correction
    result = mcnemar(np.array([[0, b], [c, 0]]), exact=False, correction=True)
    
    return {
        'statistic': statistic,
        'p_value': result.pvalue,
        'b': b,  # model1 correct, model2 wrong
        'c': c,  # model1 wrong, model2 correct
        'significant': result.pvalue < 0.05
    }

def plot_confusion_matrices(y_true_model1, y_pred_model1, y_true_model2, y_pred_model2, 
                            classes, titles, output_path):
    """Plot confusion matrices for two models side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    
    # Normalize confusion matrices
    cm1 = confusion_matrix(y_true_model1, y_pred_model1, labels=classes, normalize='true')
    cm2 = confusion_matrix(y_true_model2, y_pred_model2, labels=classes, normalize='true')
    
    # Plot first confusion matrix
    sns.heatmap(cm1, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, 
                yticklabels=classes, ax=axs[0])
    axs[0].set_title(f'{titles[0]} Confusion Matrix')
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')
    
    # Plot second confusion matrix
    sns.heatmap(cm2, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, 
                yticklabels=classes, ax=axs[1])
    axs[1].set_title(f'{titles[1]} Confusion Matrix')
    axs[1].set_ylabel('True Label')
    axs[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrices_comparison.png'))
    plt.close()

def plot_learning_curves(history_model1, history_model2, titles, output_path):
    """Plot learning curves for two models."""
    metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in history_model1.columns and metric in history_model2.columns:
            axs[i].plot(history_model1['epoch'], history_model1[metric], 'b-', label=titles[0])
            axs[i].plot(history_model2['epoch'], history_model2[metric], 'r-', label=titles[1])
            axs[i].set_title(f'Model {metric.capitalize()}')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(metric.capitalize())
            axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'learning_curves_comparison.png'))
    plt.close()

def plot_per_class_metrics(metrics_df1, metrics_df2, model_names, output_path):
    """Create a bar chart showing per-class performance for both models."""
    # Filter metrics to include only class-specific ones
    class_metrics1 = metrics_df1[metrics_df1['Metric'].str.contains(r'Precision \(|Recall \(|F1-score \(')].copy()
    class_metrics1['Model'] = model_names[0]
    
    class_metrics2 = metrics_df2[metrics_df2['Metric'].str.contains(r'Precision \(|Recall \(|F1-score \(')].copy()
    class_metrics2['Model'] = model_names[1]
    
    # Combine the dataframes
    combined = pd.concat([class_metrics1, class_metrics2])
    
    # Extract class and metric type for better plotting
    combined['Class'] = combined['Metric'].str.extract(r'\((.*?)\)')
    combined['Metric Type'] = combined['Metric'].str.extract(r'(Precision|Recall|F1-score)')
    
    # Plot
    plt.figure(figsize=(15, 10))
    g = sns.catplot(
        data=combined, 
        x='Class', 
        y='Value', 
        hue='Model', 
        col='Metric Type',
        kind='bar',
        height=6, 
        aspect=0.8
    )
    
    g.set_axis_labels('Class', 'Score')
    g.set_titles('{col_name}')
    g.fig.suptitle('Per-class Performance Comparison', y=1.05, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'per_class_metrics_comparison.png'))
    plt.close()

def compare_models(model1_path, model2_path, test_data_path, output_path, 
                   model1_history_path=None, model2_history_path=None,
                   model1_name="ImageNet", model2_name="Wildlife", batch_size=16):
    """
    Compare two models and generate comprehensive comparison metrics and visualizations.
    
    Parameters:
    -----------
    model1_path : str
        Path to the first model file (.keras)
    model2_path : str
        Path to the second model file (.keras)
    test_data_path : str
        Path to the test data directory with class subdirectories
    output_path : str
        Path to save comparison results
    model1_history_path : str, optional
        Path to training history CSV for model1
    model2_history_path : str, optional
        Path to training history CSV for model2
    model1_name : str, default="ImageNet"
        Name to use for model1 in visualizations
    model2_name : str, default="Wildlife"
        Name to use for model2 in visualizations
    batch_size : int, default=16
        Batch size for predictions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Starting comparison of {model1_name} vs {model2_name} models...")
    
    # Load models
    model1 = load_model_safely(model1_path)
    model2 = load_model_safely(model2_path)
    
    # Determine model type and image size
    try:
        model_type = Path(model1_path).name.split('_')[-1].split('.')[0]
        img_size = model_img_size_mapping(model_type.upper())
    except:
        print("Could not determine model type. Using default image size of 224.")
        img_size = 224
    
    print(f"Using image size: {img_size}")
    
    # Try to load class map from the same directory as the model
    model1_dir = os.path.dirname(model1_path)
    class_map_candidates = [
        os.path.join(model1_dir, f) for f in os.listdir(model1_dir) 
        if f.endswith('_class_map.yaml') or f == 'class_map.yaml'
    ]
    
    if class_map_candidates:
        class_map_path = class_map_candidates[0]
        print(f"Using class map from: {class_map_path}")
        with open(class_map_path, 'r') as f:
            class_map = yaml.safe_load(f)
        class_names = list(class_map.keys())
    else:
        # If no class map found, infer classes from test data directory
        class_names = [d for d in os.listdir(test_data_path) 
                 if os.path.isdir(os.path.join(test_data_path, d))]
        class_names = sorted(class_names)
        print(f"Inferred classes from test data: {class_names}")
    
    # Get predictions from both models
    print("\nGenerating predictions for model 1...")
    y_true1, y_pred1, y_scores1 = get_predictions(
        model1, test_data_path, class_names, img_size, batch_size)
    
    print("\nGenerating predictions for model 2...")
    y_true2, y_pred2, y_scores2 = get_predictions(
        model2, test_data_path, class_names, img_size, batch_size)
    
    # Calculate metrics for both models
    print("\nCalculating metrics for model 1...")
    metrics_model1 = calculate_metrics(y_true1, y_pred1, class_names)
    metrics_model1.columns = ['Metric', model1_name]
    
    print("Calculating metrics for model 2...")
    metrics_model2 = calculate_metrics(y_true2, y_pred2, class_names)
    metrics_model2.columns = ['Metric', model2_name]
    
    # Merge metrics into a comparison dataframe
    comparison_df = pd.merge(metrics_model1, metrics_model2, on='Metric')
    
    # Add difference column
    comparison_df['Difference'] = comparison_df[model2_name] - comparison_df[model1_name]
    
    # Save comparison to CSV
    comparison_csv_path = os.path.join(output_path, 'model_comparison_metrics.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Comparison metrics saved to: {comparison_csv_path}")
    
    # Perform McNemar's test
    print("\nPerforming McNemar's test for statistical significance...")
    mcnemar_result = perform_mcnemar_test(y_pred1, y_pred2, y_true1)
    
    with open(os.path.join(output_path, 'mcnemar_test_results.txt'), 'w') as f:
        f.write(f"McNemar's Test Results:\n")
        f.write(f"Statistic: {mcnemar_result['statistic']}\n")
        f.write(f"p-value: {mcnemar_result['p_value']}\n")
        f.write(f"Significant difference (p < 0.05): {mcnemar_result['significant']}\n\n")
        
        f.write("Contingency Details:\n")
        f.write(f"{model1_name} correct & {model2_name} incorrect: {mcnemar_result['b']}\n")
        f.write(f"{model1_name} incorrect & {model2_name} correct: {mcnemar_result['c']}\n")
        
        # Interpretation
        if mcnemar_result['significant']:
            if mcnemar_result['b'] > mcnemar_result['c']:
                f.write(f"\nInterpretation: {model1_name} performs SIGNIFICANTLY BETTER than {model2_name}\n")
            else:
                f.write(f"\nInterpretation: {model2_name} performs SIGNIFICANTLY BETTER than {model1_name}\n")
        else:
            f.write(f"\nInterpretation: No statistically significant difference between {model1_name} and {model2_name}\n")
    
    # Plot confusion matrices
    print("\nGenerating confusion matrix visualizations...")
    plot_confusion_matrices(
        y_true1, y_pred1, y_true2, y_pred2, 
        class_names, [model1_name, model2_name], output_path
    )
    
    # Plot per-class metrics comparison
    print("\nGenerating per-class metrics comparison...")
    plot_per_class_metrics(
        metrics_model1, metrics_model2,
        [model1_name, model2_name], output_path
    )
    
    # Plot learning curves if history is available
    if model1_history_path and model2_history_path:
        print("\nPlotting learning curves...")
        history1 = pd.read_csv(model1_history_path)
        history2 = pd.read_csv(model2_history_path)
        
        plot_learning_curves(
            history1, history2, 
            [model1_name, model2_name], output_path
        )
    
    # Generate summary visualization
    print("\nGenerating summary visualization...")
    plt.figure(figsize=(14, 8))
    
    metrics_to_plot = comparison_df[
        comparison_df['Metric'].str.contains('Accuracy|weighted avg')
    ].copy()
    
    plot_data = pd.melt(
        metrics_to_plot, 
        id_vars=['Metric'],
        value_vars=[model1_name, model2_name]
    )
    
    sns.barplot(
        data=plot_data,
        x='Metric', y='value', hue='variable'
    )
    
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'performance_summary.png'))
    plt.close()
    
    print("\nComparison completed successfully!")
    return comparison_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two trained models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('--model1_name', type=str, default='ImageNet', help='Name for model 1')
    parser.add_argument('--model2_name', type=str, default='Wildlife', help='Name for model 2')
    parser.add_argument('--model1_history', type=str, help='Path to training history CSV for model 1')
    parser.add_argument('--model2_history', type=str, help='Path to training history CSV for model 2')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for predictions')
    
    args = parser.parse_args()
    
    compare_models(
        args.model1, args.model2, args.test_data, args.output,
        args.model1_history, args.model2_history,
        args.model1_name, args.model2_name, args.batch_size
    )