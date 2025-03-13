import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import numpy as np
import os
import tempfile
import shutil
from lib_pseudo import generate_pseudo_labels, combine_datasets

class TestLibPseudo(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.unlabeled_dir = os.path.join(self.test_dir, 'unlabeled')
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.model_path = os.path.join(self.test_dir, 'model.keras')
        
        # Create dummy unlabeled directory structure
        os.makedirs(self.unlabeled_dir)
        
        # Create dummy image files
        for i in range(5):
            with open(os.path.join(self.unlabeled_dir, f'image_{i}.jpg'), 'w') as f:
                f.write('dummy image content')
    
    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.test_dir)
    
    @patch('lib_pseudo.models.load_model')
    @patch('lib_pseudo.load_img')
    def test_generate_pseudo_labels(self, mock_load_img, mock_load_model):
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([
            [0.1, 0.8, 0.1],  # High confidence for class 1
            [0.2, 0.2, 0.6],  # Medium confidence for class 2
            [0.4, 0.3, 0.3],  # Low confidence, should be rejected
            [0.1, 0.1, 0.8],  # High confidence for class 2
            [0.7, 0.2, 0.1],  # Medium confidence for class 0
        ])
        mock_load_model.return_value = mock_model
        
        # Mock image loading
        mock_load_img.return_value = tf.zeros((224, 224, 3))
        
        # Run the function with mocked dependencies
        classes = ['healthy', 'disease', 'occluded']
        pseudo_df = generate_pseudo_labels(
            self.model_path, 
            self.unlabeled_dir, 
            self.output_dir, 
            classes, 
            confidence_threshold=0.7,
            img_size=224, 
            batch_size=32
        )
        
        # Check that the DataFrame has the expected structure
        self.assertIn('File', pseudo_df.columns)
        self.assertIn('Label', pseudo_df.columns)
        self.assertIn('Confidence', pseudo_df.columns)
        
        # Only images with confidence >= 0.7 should be included
        self.assertEqual(len(pseudo_df), 2)  # Only 2 of the 5 images had confidence >= 0.7
    
    def test_combine_datasets(self):
        # Create directories for original and pseudo-labeled data
        original_train_path = os.path.join(self.test_dir, 'original')
        pseudo_labeled_path = os.path.join(self.test_dir, 'pseudo')
        output_path = os.path.join(self.test_dir, 'combined')
        
        # Create class subdirectories
        for path in [original_train_path, pseudo_labeled_path]:
            for class_name in ['class1', 'class2']:
                os.makedirs(os.path.join(path, class_name), exist_ok=True)
                
                # Create dummy image files
                for i in range(3):
                    with open(os.path.join(path, class_name, f'image_{i}.jpg'), 'w') as f:
                        f.write('dummy image content')
        
        # Run the function
        combine_datasets(original_train_path, pseudo_labeled_path, output_path)
        
        # Check that the combined dataset has the expected structure
        for class_name in ['class1', 'class2']:
            class_path = os.path.join(output_path, class_name)
            self.assertTrue(os.path.exists(class_path))
            
            # Each class should have original + pseudo files
            files = os.listdir(class_path)
            self.assertEqual(len(files), 6)  # 3 original + 3 pseudo files
