# Library for generating and managing pseudo-labels
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras import models
import shutil
from tqdm import tqdm

def load_img(file_path, img_size):
    """Load and preprocess an image"""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    img = tf.image.resize(img, size=(img_size, img_size))
    return img

def generate_pseudo_labels(model_path, unlabeled_dir, output_dir, classes, confidence_threshold=0.7, img_size=384, batch_size=32, custom_objects=None):
    """
    Generate pseudo-labels for unlabeled data using a trained model
    
    Args:
        model_path: Path to the trained model file
        unlabeled_dir: Directory containing unlabeled images
        output_dir: Directory to save pseudo-labeled images
        classes: List of class names
        confidence_threshold: Minimum confidence to accept a prediction
        img_size: Image size required by the model
        batch_size: Batch size for prediction
        custom_objects: Custom objects for model loading
        
    Returns:
        Path to CSV file containing pseudo-labeled data information
    """
    print(f"Generating pseudo-labels for unlabeled data in {unlabeled_dir}")
    print(f"Using model: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Clear output directory if it exists
    if os.path.exists(output_dir):
        print(f"Clearing existing pseudo-label directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create output directories for each class
    os.makedirs(output_dir, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # Load the model
    try:
        # Try to load with custom objects first
        if custom_objects:
            model = models.load_model(model_path, custom_objects=custom_objects, compile=False)
        else:
            model = models.load_model(model_path, compile=False)
        model.trainable = False  # Set to inference mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Get all image files from the unlabeled directory
    unlabeled_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        unlabeled_files.extend(list(Path(unlabeled_dir).glob(f"**/{ext}")))
    
    if len(unlabeled_files) == 0:
        raise ValueError(f"No image files found in {unlabeled_dir}")
    
    print(f"Found {len(unlabeled_files)} unlabeled images")
    
    # Create batches for prediction
    all_files = [str(f) for f in unlabeled_files]
    num_batches = (len(all_files) + batch_size - 1) // batch_size
    
    pseudo_labeled_files = []
    pseudo_labels = []
    pseudo_confidences = []
    
    for batch_idx in tqdm(range(num_batches), desc="Generating pseudo-labels"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_files))
        batch_files = all_files[start_idx:end_idx]
        
        # Load and preprocess images
        batch_images = np.stack([load_img(f, img_size).numpy() for f in batch_files])
        
        # Predict
        predictions = model.predict(batch_images, verbose=0)
        
        # Process predictions
        for i, (file_path, pred) in enumerate(zip(batch_files, predictions)):
            confidence = np.max(pred)
            if confidence >= confidence_threshold:
                class_idx = np.argmax(pred)
                class_name = classes[class_idx]
                
                # Copy the file to the output directory with the pseudo-label
                dest_path = os.path.join(output_dir, class_name, os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                
                # Record the pseudo-label
                pseudo_labeled_files.append(dest_path)
                pseudo_labels.append(class_name)
                pseudo_confidences.append(confidence)
    
    # Create a DataFrame with pseudo-labeled information
    if len(pseudo_labeled_files) > 0:
        pseudo_df = pd.DataFrame({
            'File': pseudo_labeled_files,
            'Label': pseudo_labels,
            'Confidence': pseudo_confidences
        })
        
        # Print statistics
        print(f"\nPseudo-labeling complete!")
        print(f"Labeled {len(pseudo_df)} out of {len(all_files)} images ({100*len(pseudo_df)/len(all_files):.2f}%)")
        print("\nDistribution of pseudo-labels:")
        print(pseudo_df['Label'].value_counts())
        
        # Save DataFrame to CSV
        csv_path = os.path.join(output_dir, 'pseudo_labels.csv')
        pseudo_df.to_csv(csv_path, index=False)
        print(f"\nPseudo-label information saved to {csv_path}")
        
        return csv_path
    else:
        print(f"\nNo images met the confidence threshold of {confidence_threshold}")
        return None

def combine_datasets(original_train_path, pseudo_labeled_csv_path, output_path):
    """
    Combine original labeled data with pseudo-labeled data using the CSV file
    
    Args:
        original_train_path: Path to original labeled training data
        pseudo_labeled_csv_path: Path to CSV file containing pseudo-label information
        output_path: Path to save the combined dataset
    """
    print(f"Combining original labeled data with pseudo-labeled data")
    
    # Clear output directory if it exists
    if os.path.exists(output_path):
        print(f"Clearing existing combined dataset directory: {output_path}")
        shutil.rmtree(output_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy original labeled data
    print("Copying original labeled data...")
    for class_dir in os.listdir(original_train_path):
        class_path = os.path.join(original_train_path, class_dir)
        if os.path.isdir(class_path):
            # Create output class directory
            output_class_dir = os.path.join(output_path, class_dir)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Copy original files
            file_count = 0
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(class_path, file_name)
                    dst_file = os.path.join(output_class_dir, f"original_{file_name}")
                    shutil.copy2(src_file, dst_file)
                    file_count += 1
            print(f"  - Copied {file_count} images from class {class_dir}")
    
    # Read and process pseudo-labeled data from CSV
    if not os.path.exists(pseudo_labeled_csv_path):
        print(f"Warning: Pseudo-label CSV file not found: {pseudo_labeled_csv_path}")
        return
    
    print(f"Adding pseudo-labeled data from: {pseudo_labeled_csv_path}")
    try:
        import pandas as pd
        pseudo_df = pd.read_csv(pseudo_labeled_csv_path)
        
        # Verify CSV structure
        required_columns = ['File', 'Label', 'Confidence']
        if not all(col in pseudo_df.columns for col in required_columns):
            print(f"Error: CSV file does not contain required columns: {required_columns}")
            return
        
        # Process each pseudo-labeled image
        copied_count = 0
        class_counts = {}
        
        # Convert all labels to strings (fixes the int join error)
        pseudo_df['Label'] = pseudo_df['Label'].astype(str)
        
        for idx, row in pseudo_df.iterrows():
            source_file = row['File']
            label = str(row['Label'])  # Ensure label is a string
            confidence = row['Confidence']
            
            # Ensure label directory exists
            output_class_dir = os.path.join(output_path, label)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Extract filename and create destination path
            file_name = os.path.basename(source_file)
            dst_file = os.path.join(output_class_dir, f"pseudo_{confidence:.2f}_{file_name}")
            
            # Copy file
            if os.path.exists(source_file):
                shutil.copy2(source_file, dst_file)
                copied_count += 1
                
                # Track counts per class
                class_counts[label] = class_counts.get(label, 0) + 1
            else:
                print(f"Warning: Source file not found: {source_file}")
                
            # Print progress periodically
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(pseudo_df)} pseudo-labeled images...")
        
        print(f"Added {copied_count} pseudo-labeled images to the combined dataset")
        print("Pseudo-labeled class distribution:")
        for label, count in class_counts.items():
            print(f"  Class {label}: {count} added")
        
    except Exception as e:
        import traceback
        print(f"Error processing pseudo-label CSV: {e}")
        traceback.print_exc()
    
    # Count files in combined dataset
    total_files = 0
    class_counts = {}
    for class_dir in os.listdir(output_path):
        class_path = os.path.join(output_path, class_dir)
        if os.path.isdir(class_path):
            file_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_dir] = file_count
            total_files += file_count
    
    print(f"\nCombined dataset created at {output_path}")
    print(f"Total files: {total_files}")
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} files")

# Add this new function for curriculum-based dataset creation
def create_curriculum_datasets(original_train_path, pseudo_labeled_csv_path, output_base_path, num_stages=4, min_confidence=0.7):
    """
    Create a series of datasets for curriculum learning with progressively more pseudo-labels
    
    Args:
        original_train_path: Path to original labeled training data
        pseudo_labeled_csv_path: Path to CSV file containing pseudo-label information
        output_base_path: Base path to save the curriculum datasets
        num_stages: Number of curriculum stages
        min_confidence: Minimum confidence threshold for including any pseudo-label
        
    Returns:
        List of paths to the curriculum datasets
    """
    print(f"Creating curriculum datasets for pseudo-label training")
    
    # Make sure the base output directory exists
    os.makedirs(output_base_path, exist_ok=True)
    
    # Load the pseudo-labeled data
    import pandas as pd
    pseudo_df = pd.read_csv(pseudo_labeled_csv_path)
    
    # Verify CSV structure
    required_columns = ['File', 'Label', 'Confidence']
    if not all(col in pseudo_df.columns for col in required_columns):
        raise ValueError(f"CSV file does not contain required columns: {required_columns}")
    
    # Convert all labels to strings
    pseudo_df['Label'] = pseudo_df['Label'].astype(str)
    
    # Filter by minimum confidence
    pseudo_df = pseudo_df[pseudo_df['Confidence'] >= min_confidence]
    if len(pseudo_df) == 0:
        raise ValueError(f"No pseudo-labels with confidence >= {min_confidence}")
    
    # Sort by confidence (highest first)
    pseudo_df = pseudo_df.sort_values(by='Confidence', ascending=False)
    
    # Calculate the number of samples to include in each stage
    total_samples = len(pseudo_df)
    samples_per_stage = [int(total_samples * (i+1) / num_stages) for i in range(num_stages)]
    print(f"Total pseudo-labeled samples: {total_samples}")
    print(f"Samples per curriculum stage: {samples_per_stage}")
    
    # Fix 2: Save the CSV to a temporary directory first
    temp_dir = os.path.join(output_base_path, "temp_csv")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create datasets for each curriculum stage
    curriculum_paths = []
    for stage, num_samples in enumerate(samples_per_stage):
        stage_path = os.path.join(output_base_path, f"stage_{stage+1}")
        os.makedirs(stage_path, exist_ok=True)
        
        # Select the top N samples for this stage
        stage_df = pseudo_df.iloc[:num_samples]
        min_conf_in_stage = stage_df['Confidence'].min()
        print(f"Stage {stage+1}: Using {num_samples} samples with confidence >= {min_conf_in_stage:.4f}")
        
        # Create a temporary CSV for this stage in the temp directory
        stage_csv_path = os.path.join(temp_dir, f"pseudo_labels_stage_{stage+1}.csv")
        stage_df.to_csv(stage_csv_path, index=False)
        
        # Combine original data with this stage's pseudo-labels
        combine_datasets(original_train_path, stage_csv_path, stage_path)
        curriculum_paths.append(stage_path)
    
    return curriculum_paths 