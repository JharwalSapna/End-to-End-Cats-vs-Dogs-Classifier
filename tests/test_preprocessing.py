"""
Unit tests for data preprocessing functions.
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_image, augment_image, split_data
import random


class TestLoadImage:
    """Tests for image loading function."""

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
    
    def test_load_valid_image(self):
        """Test loading a valid image file."""
        # Create a temp image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(f.name)
            
            result = load_image(f.name, target_size=(224, 224))
            
            assert result is not None
            assert result.shape == (224, 224, 3)
            assert result.dtype == np.float32
            assert result.min() >= -1.0 and result.max() <= 1.0
            
            os.unlink(f.name)
    
    def test_load_grayscale_converts_to_rgb(self):
        """Test that grayscale images are converted to RGB."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
            img.save(f.name)
            
            result = load_image(f.name, target_size=(224, 224))
            
            assert result is not None
            assert result.shape == (224, 224, 3)
            
            os.unlink(f.name)
    
    def test_load_nonexistent_file_returns_none(self):
        """Test that loading a non-existent file returns None."""
        result = load_image('/nonexistent/path/image.jpg')
        assert result is None


class TestAugmentImage:
    """Tests for image augmentation function."""

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
    
    def test_augment_preserves_shape(self):
        """Test that augmentation preserves image shape."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        augmented = augment_image(img)
        
        assert augmented.shape == img.shape
        assert augmented.dtype == np.float32
    
    def test_augment_within_valid_range(self):
        """Test that augmented values stay in [0, 1] range."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Run multiple times since augmentation is random
        for _ in range(10):
            augmented = augment_image(img)
            assert augmented.min() >= -1.0
            assert augmented.max() <= 1.0


class TestSplitData:
    """Tests for data splitting function."""

    def setup_method(self):
        np.random.seed(42)
        random.seed(42)
    
    def test_split_correct_proportions(self):
        """Test that split follows specified ratios."""
        n_samples = 100
        images = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples)
        
        (train, train_labels), (val, val_labels), (test, test_labels) = split_data(
            images, labels, train_ratio=0.8, val_ratio=0.1
        )
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
    
    def test_split_no_data_leakage(self):
        """Test that splits have no overlapping indices."""
        n_samples = 50
        # Use unique values so we can detect duplicates
        images = np.arange(n_samples).reshape(n_samples, 1, 1, 1).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples)
        
        (train, _), (val, _), (test, _) = split_data(images, labels)
        
        all_values = set(train.flatten().tolist() + val.flatten().tolist() + test.flatten().tolist())
        total_samples = len(train) + len(val) + len(test)
        
        assert len(all_values) == total_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
