# Example comparison script structure
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
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
import traceback

# Import from your existing modules
from lib_data import create_tensorset
from lib_common import model_img_size_mapping

# Try importing from lib_model, handle potential errors
try:
    from lib_model import (
        build_sequential_model,
        model_constructors,
        calc_class_metrics as lib_calc_class_metrics # Import with an alias
    )
except ImportError:
    print("Warning: Could not import functions from lib_model. Ensure it's accessible.")
    # Define necessary components locally if import fails or is undesired (as before)
    # ... (fallback definitions for model_constructors, build_sequential_model) ...

    # Define a placeholder if calc_class_metrics cannot be imported
    def lib_calc_class_metrics(*args, **kwargs):
        print("WARNING: lib_model.calc_class_metrics could not be imported. Skipping direct comparison.")
        # Optionally return None or raise an error if strict comparison is needed
        return None

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
    model.trainable = False
    
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
    """Perform McNemar's test on OVERALL predictions."""
    y_pred_model1_arr = np.array(y_pred_model1)
    y_pred_model2_arr = np.array(y_pred_model2)
    y_true_arr = np.array(y_true)

    correct_model1 = (y_pred_model1_arr == y_true_arr)
    correct_model2 = (y_pred_model2_arr == y_true_arr)

    b = np.sum(correct_model1 & ~correct_model2) # M1 correct, M2 wrong
    c = np.sum(~correct_model1 & correct_model2) # M1 wrong, M2 correct
    a = np.sum(correct_model1 & correct_model2)  # Both correct
    d = np.sum(~correct_model1 & ~correct_model2) # Both wrong
    cont_table = np.array([[a, b], [c, d]])

    # Use exact test if discordant pairs are few, otherwise chi-square
    n_discordant = b + c
    if n_discordant == 0:
         # Avoid division by zero or running test on no disagreements
         print("Warning: No discordant pairs found for overall McNemar test.")
         stat = 0
         pval = 1.0
    elif n_discordant < 25:
        print(f"Using McNemar's exact test (b+c = {n_discordant} < 25)")
        result = mcnemar(cont_table, exact=True) # correction=False is implied for exact=True
        stat = result.statistic # May be NaN for exact test, focus on p-value
        pval = result.pvalue
    else:
        print(f"Using McNemar's chi-square test (b+c = {n_discordant} >= 25)")
        result = mcnemar(cont_table, exact=False, correction=False) # Chi-square without correction
        stat = result.statistic
        pval = result.pvalue

    return {
        'statistic': stat,
        'p_value': pval,
        'b': b,
        'c': c,
        'n_discordant': n_discordant,
        'significant': pval < 0.05 if pval is not None else False
    }

def perform_per_class_mcnemar(y_pred_model1, y_pred_model2, y_true, classes):
    """Perform McNemar's test for each true class."""
    y_pred_model1_arr = np.array(y_pred_model1)
    y_pred_model2_arr = np.array(y_pred_model2)
    y_true_arr = np.array(y_true)
    results = {}

    print("\n--- Per-Class McNemar's Test ---")
    for target_class in classes:
        print(f"Testing for TRUE class: {target_class}")
        # Filter data for the current true class
        indices = np.where(y_true_arr == target_class)[0]
        if len(indices) == 0:
            print(f"  No instances found for class {target_class}. Skipping.")
            continue

        y_true_subset = y_true_arr[indices]
        y_pred1_subset = y_pred_model1_arr[indices]
        y_pred2_subset = y_pred_model2_arr[indices]

        # Check correctness *within this subset*
        # Correct means predicting the target_class
        correct_model1 = (y_pred1_subset == target_class)
        correct_model2 = (y_pred2_subset == target_class)

        b = np.sum(correct_model1 & ~correct_model2) # M1 correct, M2 wrong
        c = np.sum(~correct_model1 & correct_model2) # M1 wrong, M2 correct
        a = np.sum(correct_model1 & correct_model2)  # Both correct
        d = np.sum(~correct_model1 & ~correct_model2) # Both wrong (predict wrong class)
        cont_table = np.array([[a, b], [c, d]])

        n_discordant = b + c
        if n_discordant == 0:
             print(f"  No discordant pairs found for class {target_class}.")
             stat = 0
             pval = 1.0
             test_type = "N/A (No Discordance)"
        elif n_discordant < 25:
            print(f"  Using McNemar's exact test (b+c = {n_discordant} < 25)")
            result = mcnemar(cont_table, exact=True)
            stat = result.statistic # May be NaN for exact test
            pval = result.pvalue
            test_type = "Exact Binomial"
        else:
            print(f"  Using McNemar's chi-square test (b+c = {n_discordant} >= 25)")
            result = mcnemar(cont_table, exact=False, correction=False)
            stat = result.statistic
            pval = result.pvalue
            test_type = "Chi-Square (no correction)"

        results[target_class] = {
            'statistic': stat,
            'p_value': pval,
            'b (M1✓, M2X)': b,
            'c (M1X, M2✓)': c,
            'n_discordant': n_discordant,
            'test_type': test_type,
            'significant': pval < 0.05 if pval is not None else False
        }
        print(f"  Results: p={pval:.4f}, b={b}, c={c}, Significant={results[target_class]['significant']}")

    print("--- End Per-Class McNemar's Test ---")
    return results

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
    Compare two models using architecture rebuild + weight loading,
    and include a direct call to lib_model.calc_class_metrics for validation.
    """
    # Use Path object for easier manipulation
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True) # Use Path's mkdir
    print(f"Results will be saved to: {output_path.resolve()}") # Show absolute path

    print(f"Starting comparison of {model1_name} vs {model2_name} models...")

    # --- Determine Model Config (as before) ---
    # ... (Infer classes, model keys, img_size) ...
    if not os.path.isdir(test_data_path):
         raise ValueError(f"Test data path not found or not a directory: {test_data_path}")
    try:
        class_names = sorted([d for d in os.listdir(test_data_path)
                          if os.path.isdir(os.path.join(test_data_path, d))])
        if not class_names:
             raise ValueError(f"No subdirectories found in {test_data_path} to infer classes.")
        print(f"Inferred classes from test data: {class_names}")
        num_classes = len(class_names)
    except Exception as e:
        print(f"Error inferring class names: {e}")
        raise

    try:
        model1_filename = Path(model1_path).name
        model2_filename = Path(model2_path).name
        model1_key = model1_filename.split('_')[-1].split('.')[0].upper()
        model2_key = model2_filename.split('_')[-1].split('.')[0].upper()
        img_size1 = model_img_size_mapping(model1_key)
        img_size2 = model_img_size_mapping(model2_key)
        if img_size1 != img_size2:
            print(f"Warning: Models have different image sizes ({img_size1} vs {img_size2}). Using size for Model 1: {img_size1}")
        img_size = img_size1
    except Exception as e:
        print(f"Could not determine model type/image size from filenames: {e}")
        print("Using default image size of 224. Ensure this is correct for BOTH models.")
        img_size = 224
        raise ValueError("Unable to determine model keys from filenames. Adjust parsing logic.")

    print(f"Using image size: {img_size}")
    print(f"Model 1 Key: {model1_key}, Model 2 Key: {model2_key}")
    dropout_rate = 0.1 # Assuming default

    # --- Load Models by Rebuilding ---
    model1 = load_model_safely(model1_path)
    model2 = load_model_safely(model2_path)

    # --- Get Predictions (model_comparison.py internal method) ---
    print("\nGenerating predictions for model 1 (model_comparison method)...")
    y_true1_names, y_pred1_names, y_scores1 = get_predictions(
        model1, test_data_path, class_names, img_size, batch_size)

    print("\nGenerating predictions for model 2 (model_comparison method)...")
    y_true2_names, y_pred2_names, y_scores2 = get_predictions(
        model2, test_data_path, class_names, img_size, batch_size)

    # --- Calculate Metrics (model_comparison.py internal method) ---
    print("\nCalculating metrics for model 1 (model_comparison method)...")
    metrics_model1 = calculate_metrics(y_true1_names, y_pred1_names, class_names)
    metrics_model1.columns = ['Metric', f"{model1_name}_CompScript"] # Add suffix

    print("Calculating metrics for model 2 (model_comparison method)...")
    metrics_model2 = calculate_metrics(y_true2_names, y_pred2_names, class_names)
    metrics_model2.columns = ['Metric', f"{model2_name}_CompScript"] # Add suffix

    # --- Call lib_model.calc_class_metrics for Sanity Check ---
    print(f"\n--- Running Sanity Check using lib_model.calc_class_metrics ---")
    lib_metrics_output_path = output_path / "lib_model_metrics_output" # Subdir for clarity
    lib_metrics_output_path.mkdir(exist_ok=True)

    print(f"\nRunning lib_calc_class_metrics for {model1_name}...")
    # Note: lib_calc_class_metrics might print its own report and save plots.
    # We are calling it primarily to see if its internal logic produces different results
    # on the *exact same loaded model object* (model1) and test data path.
    # It likely expects model path, not object, so we pass the path again.
    # It also generates its own plots/reports in its output path.
    try:
        # Pass the model *path* as lib_calc_class_metrics expects,
        # along with other necessary args. It will reload the model internally.
        # We save its output to a different subdir to avoid overwriting.
         _ = lib_calc_class_metrics(
             model_fpath=model1_path, # Pass path
             test_fpath=test_data_path,
             output_fpath=str(lib_metrics_output_path), # Convert Path to string
             classes=class_names,
             batch_size=batch_size,
             img_size=img_size
         )
         print(f"lib_calc_class_metrics completed for {model1_name}. Check output in {lib_metrics_output_path}")
    except Exception as e:
         print(f"ERROR running lib_calc_class_metrics for {model1_name}: {e}")
         traceback.print_exc()


    print(f"\nRunning lib_calc_class_metrics for {model2_name}...")
    try:
         _ = lib_calc_class_metrics(
             model_fpath=model2_path, # Pass path
             test_fpath=test_data_path,
             output_fpath=str(lib_metrics_output_path), # Convert Path to string
             classes=class_names,
             batch_size=batch_size,
             img_size=img_size
         )
         print(f"lib_calc_class_metrics completed for {model2_name}. Check output in {lib_metrics_output_path}")
    except Exception as e:
         print(f"ERROR running lib_calc_class_metrics for {model2_name}: {e}")
         traceback.print_exc()

    print(f"--- End of lib_model.calc_class_metrics Sanity Check ---")


    # --- Compare Metrics (from model_comparison.py methods) ---
    # Keep using the metrics calculated by *this* script's functions for comparison/plots
    comparison_df = pd.merge(metrics_model1, metrics_model2, on='Metric', how='outer')
    comparison_df['Difference'] = comparison_df[f"{model2_name}_CompScript"].sub(comparison_df[f"{model1_name}_CompScript"], fill_value=np.nan)

    comparison_csv_path = output_path / 'model_comparison_metrics.csv' # Use Path object
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparison metrics (from model_comparison.py) saved to: {comparison_csv_path}")
    print("Metrics calculated by model_comparison.py:")
    print(comparison_df)

    # --- Perform McNemar's Test (Overall) ---
    print("\nPerforming McNemar's test for OVERALL statistical significance...")
    # Use the consistent true labels (e.g., y_true1_names)
    overall_mcnemar_result = perform_mcnemar_test(y_pred1_names, y_pred2_names, y_true1_names)

    mcnemar_file = output_path / 'mcnemar_test_results.txt' # Use Path object
    with open(mcnemar_file, 'w') as f:
        f.write(f"--- Overall McNemar's Test Results ({model1_name} vs {model2_name}) ---\n")
        f.write(f"Model 1 correct & Model 2 incorrect (b): {overall_mcnemar_result['b']}\n")
        f.write(f"Model 1 incorrect & Model 2 correct (c): {overall_mcnemar_result['c']}\n")
        f.write(f"Total Discordant Pairs (b+c): {overall_mcnemar_result['n_discordant']}\n\n")
        f.write(f"Statistic: {overall_mcnemar_result['statistic']:.4f}\n") # Chi2 or NaN
        f.write(f"p-value: {overall_mcnemar_result['p_value']:.4f}\n")
        f.write(f"Significant difference (p < 0.05): {overall_mcnemar_result['significant']}\n\n")
        # Interpretation
        if overall_mcnemar_result['p_value'] < 0.05:
             if overall_mcnemar_result['b'] > overall_mcnemar_result['c']:
                 interpretation = f"{model1_name} performs SIGNIFICANTLY BETTER overall than {model2_name}"
             elif overall_mcnemar_result['c'] > overall_mcnemar_result['b']:
                 interpretation = f"{model2_name} performs SIGNIFICANTLY BETTER overall than {model1_name}"
             else:
                 interpretation = "Significant difference overall, but discordant counts b and c are equal."
        else:
             interpretation = f"No statistically significant difference overall between {model1_name} and {model2_name}"
        f.write(f"Overall Interpretation: {interpretation}\n")
    print(f"Overall McNemar's test results saved to: {mcnemar_file}")
    print(interpretation)

    # --- Perform Per-Class McNemar's Test ---
    per_class_mcnemar_results = perform_per_class_mcnemar(y_pred1_names, y_pred2_names, y_true1_names, class_names)

    # Append per-class results to the same file
    with open(mcnemar_file, 'a') as f: # Open in append mode
        f.write("\n\n--- Per-Class McNemar's Test Results ---\n")
        for target_class, result in per_class_mcnemar_results.items():
            f.write(f"\nTrue Class: {target_class}\n")
            f.write(f"  Test Type: {result['test_type']}\n")
            f.write(f"  Model 1 correct & Model 2 incorrect (b): {result['b (M1✓, M2X)']}\n")
            f.write(f"  Model 1 incorrect & Model 2 correct (c): {result['c (M1X, M2✓)']}\n")
            f.write(f"  Discordant Pairs (b+c): {result['n_discordant']}\n")
            f.write(f"  Statistic: {result['statistic']:.4f}\n") # May be NaN for exact
            f.write(f"  p-value: {result['p_value']:.4f}\n")
            f.write(f"  Significant difference (p < 0.05): {result['significant']}\n")
            # Per-class interpretation
            if result['p_value'] < 0.05:
                 if result['b (M1✓, M2X)'] > result['c (M1X, M2✓)']:
                     class_interpretation = f"  Interpretation: {model1_name} is significantly better than {model2_name} for class '{target_class}'."
                 elif result['c (M1X, M2✓)'] > result['b (M1✓, M2X)']:
                     class_interpretation = f"  Interpretation: {model2_name} is significantly better than {model1_name} for class '{target_class}'."
                 else:
                     class_interpretation = f"  Interpretation: Significant difference for class '{target_class}', but discordant counts are equal."
            else:
                 class_interpretation = f"  Interpretation: No significant difference between models for class '{target_class}'."
                 if result['n_discordant'] < 5: # Add warning for low power
                      class_interpretation += " (Note: Low number of discordant pairs limits test power)."
            f.write(class_interpretation + "\n")
    print(f"Per-class McNemar's test results appended to: {mcnemar_file}")


    # --- Generate Plots (using this script's predictions/metrics) ---
    print("\nGenerating confusion matrix visualizations...")
    plot_confusion_matrices(
        y_true1_names, y_pred1_names, y_true2_names, y_pred2_names,
        class_names, [model1_name, model2_name], output_path
    )
    # ... (Call other plotting functions as before, using output_path) ...
    print("Generating per-class metrics comparison...")
    plot_per_class_metrics(
        metrics_model1.rename(columns={f"{model1_name}_CompScript": model1_name}), # Rename for plot func
        metrics_model2.rename(columns={f"{model2_name}_CompScript": model2_name}),
        [model1_name, model2_name], output_path
    )
    # ... (Plot learning curves) ...
    # ... (Plot summary) ...


    print("\nComparison completed successfully!")
    # Return the comparison df calculated by this script
    return comparison_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two trained models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model (.keras file)')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model (.keras file)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory (can be absolute or relative)')
    parser.add_argument('--model1_name', type=str, default='Model1', help='Name for model 1')
    parser.add_argument('--model2_name', type=str, default='Model2', help='Name for model 2')
    parser.add_argument('--model1_history', type=str, help='Path to training history CSV for model 1')
    parser.add_argument('--model2_history', type=str, help='Path to training history CSV for model 2')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for predictions')

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.model1): raise FileNotFoundError(f"Model 1 file not found: {args.model1}")
    if not os.path.exists(args.model2): raise FileNotFoundError(f"Model 2 file not found: {args.model2}")
    if not os.path.isdir(args.test_data): raise NotADirectoryError(f"Test data path is not a directory: {args.test_data}")

    compare_models(
        args.model1, args.model2, args.test_data, args.output,
        args.model1_history, args.model2_history,
        args.model1_name, args.model2_name, args.batch_size
    )