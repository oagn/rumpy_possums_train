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

def generate_pseudo_labels(model_path, unlabeled_dir, output_dir, classes, confidence_threshold=0.7, img_size=384, batch_size=32):
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
        
    Returns:
        DataFrame containing pseudo-labeled data information
    """
    print(f"Generating pseudo-labels for unlabeled data in {unlabeled_dir}")
    print(f"Using model: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Load the model
    model = models.load_model(model_path, compile=False)
    model.trainable = False  # Set to inference mode
    
    # Get all image files from the unlabeled directory
    unlabeled_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        unlabeled_files.extend(list(Path(unlabeled_dir).glob(f"**/{ext}")))
    
    if len(unlabeled_files) == 0:
        raise ValueError(f"No image files found in {unlabeled_dir}")
    
    print(f"Found {len(unlabeled_files)} unlabeled images")
    
    # Create output directories for each class
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
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
        
        return pseudo_df
    else:
        print(f"\nNo images met the confidence threshold of {confidence_threshold}")
        return pd.DataFrame(columns=['File', 'Label', 'Confidence'])

def combine_datasets(original_train_path, pseudo_labeled_path, output_path):
    """
    Combine original labeled data with pseudo-labeled data
    
    Args:
        original_train_path: Path to original labeled training data
        pseudo_labeled_path: Path to pseudo-labeled data
        output_path: Path to save the combined dataset
    """
    print(f"Combining original labeled data with pseudo-labeled data")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy original labeled data
    for class_dir in os.listdir(original_train_path):
        class_path = os.path.join(original_train_path, class_dir)
        if os.path.isdir(class_path):
            # Create output class directory
            output_class_dir = os.path.join(output_path, class_dir)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Copy original files
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(class_path, file_name)
                    dst_file = os.path.join(output_class_dir, f"original_{file_name}")
                    shutil.copy2(src_file, dst_file)
    
    # Copy pseudo-labeled data
    for class_dir in os.listdir(pseudo_labeled_path):
        class_path = os.path.join(pseudo_labeled_path, class_dir)
        if os.path.isdir(class_path):
            # Create output class directory if it doesn't exist
            output_class_dir = os.path.join(output_path, class_dir)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Copy pseudo-labeled files
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(class_path, file_name)
                    dst_file = os.path.join(output_class_dir, f"pseudo_{file_name}")
                    shutil.copy2(src_file, dst_file)
    
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