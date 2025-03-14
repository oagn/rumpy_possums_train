import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import numpy as np
import os
import tempfile
import shutil
import pandas as pd
from pathlib import Path

class TestLibPseudo(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.unlabeled_dir = os.path.join(self.test_dir, 'unlabeled')
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.model_path = os.path.join(self.test_dir, 'model.keras')
        
        # Create dummy unlabeled directory structure
        os.makedirs(self.unlabeled_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create dummy image files
        for i in range(5):
            with open(os.path.join(self.unlabeled_dir, f'image_{i}.jpg'), 'wb') as f:
                f.write(b'dummy image content')
        
        print(f"\nTest setup complete. Created test directories:")
        print(f"- Unlabeled dir: {self.unlabeled_dir}")
        print(f"- Output dir: {self.output_dir}")
        print(f"- Model path: {self.model_path}")
    
    def tearDown(self):
        # Clean up temporary directories
        try:
            shutil.rmtree(self.test_dir)
            print(f"Test cleanup complete. Removed test directory: {self.test_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up test directory: {e}")
    
    @patch('lib_pseudo.models.load_model')
    @patch('lib_pseudo.load_img')
    def test_generate_pseudo_labels(self, mock_load_img, mock_load_model):
        from lib_pseudo import generate_pseudo_labels
        
        print("\n=== Running test_generate_pseudo_labels ===")
        
        # Mock the model
        mock_model = MagicMock()
        predictions = np.array([
            [0.1, 0.8, 0.1],  # High confidence for class 1
            [0.2, 0.2, 0.6],  # Medium confidence for class 2
            [0.4, 0.3, 0.3],  # Low confidence, should be rejected
            [0.1, 0.1, 0.8],  # High confidence for class 2
            [0.7, 0.2, 0.1],  # Medium confidence for class 0
        ])
        mock_model.predict.return_value = predictions
        mock_load_model.return_value = mock_model
        print(f"Mocked model with {len(predictions)} predictions")
        
        # Mock image loading
        mock_img = tf.zeros((224, 224, 3))
        mock_load_img.return_value = mock_img
        
        # Define test classes
        classes = ['healthy', 'disease', 'occluded']
        
        # Run the function with mocked dependencies
        try:
            pseudo_df = generate_pseudo_labels(
                self.model_path, 
                self.unlabeled_dir, 
                self.output_dir, 
                classes, 
                confidence_threshold=0.7,
                img_size=224, 
                batch_size=32
            )
            
            print(f"Function returned DataFrame with shape: {pseudo_df.shape}")
            print(f"DataFrame columns: {pseudo_df.columns.tolist()}")
            if not pseudo_df.empty:
                print(f"Sample of pseudo labels:\n{pseudo_df.head()}")
            
            # Check that the DataFrame has the expected structure
            self.assertIn('File', pseudo_df.columns)
            self.assertIn('Label', pseudo_df.columns)
            self.assertIn('Confidence', pseudo_df.columns)
            
            # Only images with confidence >= 0.7 should be included
            self.assertEqual(len(pseudo_df), 2)  # Only 2 of the 5 images had confidence >= 0.7
            
            print("Test completed successfully")
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_combine_datasets(self):
        from lib_pseudo import combine_datasets
        
        print("\n=== Running test_combine_datasets ===")
        
        # Create directories for original and pseudo-labeled data
        original_train_path = os.path.join(self.test_dir, 'original')
        pseudo_labeled_path = os.path.join(self.test_dir, 'pseudo')
        combined_path = os.path.join(self.test_dir, 'combined')
        
        # Create class subdirectories
        for path in [original_train_path, pseudo_labeled_path]:
            for class_name in ['class1', 'class2']:
                class_dir = os.path.join(path, class_name)
                os.makedirs(class_dir, exist_ok=True)
                print(f"Created directory: {class_dir}")
                
                # Create dummy image files
                for i in range(3):
                    img_path = os.path.join(class_dir, f'image_{i}.jpg')
                    with open(img_path, 'wb') as f:
                        f.write(b'dummy image content')
                    print(f"Created dummy image: {img_path}")
        
        # Run the function
        try:
            combine_datasets(original_train_path, pseudo_labeled_path, combined_path)
            
            # Check that the combined dataset has the expected structure
            for class_name in ['class1', 'class2']:
                class_path = os.path.join(combined_path, class_name)
                self.assertTrue(os.path.exists(class_path))
                
                # Each class should have original + pseudo files
                files = os.listdir(class_path)
                print(f"Combined class {class_name} has {len(files)} files: {files}")
                self.assertEqual(len(files), 6)  # 3 original + 3 pseudo files
                
            print("Test completed successfully")
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    unittest.main()
