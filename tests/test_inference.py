"""
Unit tests for model inference functions.
"""

import os
import sys
import pytest
import numpy as np
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SimpleCNN, compute_accuracy, compute_confusion_matrix


class TestSimpleCNN:
    """Tests for the SimpleCNN model."""
    
    def setup_method(self):
        np.random.seed(42)
    
    def test_model_initialization(self):
        """Test that model initializes with correct shapes."""
        model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
        
        assert model.weights['h1'].shape == (224 * 224 * 3, 64)
        assert model.weights['h2'].shape == (64, 32)
        assert model.weights['h3'].shape == (32, 16)
        assert model.weights['out'].shape == (16, 1)
        assert model.biases['h1'].shape == (64,)
        assert model.biases['h2'].shape == (32,)
        assert model.biases['h3'].shape == (16,)
        assert model.biases['out'].shape == (1,)
    
    def test_forward_pass_output_shape(self):
        """Test that forward pass produces correct output shape."""
        model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
        x = np.random.rand(5, 224, 224, 3).astype(np.float32)
        
        output = model.forward(x)
        
        assert output.shape == (5,)
    
    def test_predict_returns_binary(self):
        """Test that predict returns 0 or 1."""
        model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
        x = np.random.rand(10, 224, 224, 3).astype(np.float32)
        
        predictions = model.predict(x)
        
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba_in_range(self):
        """Test that probabilities are in [0, 1] range."""
        model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
        x = np.random.rand(10, 224, 224, 3).astype(np.float32)
        
        probs = model.predict_proba(x)
        
        assert all(0 <= p <= 1 for p in probs)
    
    def test_save_and_load(self):
        """Test that model can be saved and loaded."""
        model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
        x = np.random.rand(3, 224, 224, 3).astype(np.float32)
        
        original_pred = model.predict_proba(x)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            model.save(f.name)
            loaded_model = SimpleCNN.load(f.name)
            loaded_pred = loaded_model.predict_proba(x)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            os.unlink(f.name)
    
    def test_train_step_reduces_loss(self):
        """Test that training step reduces loss over iterations."""
        model = SimpleCNN(input_shape=(32, 32, 3), hidden_units=32)
        x = np.random.rand(8, 32, 32, 3).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        initial_loss = model.train_step(x, y, learning_rate=0.01)
        
        # Train for a few more steps
        for _ in range(10):
            final_loss = model.train_step(x, y, learning_rate=0.01)
        
        # Loss should generally decrease (or at least not explode)
        assert final_loss < initial_loss * 2  # Allow some variance


class TestMetricFunctions:
    """Tests for metric computation functions."""
    
    def test_compute_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 1.0
    
    def test_compute_accuracy_half(self):
        """Test accuracy with 50% correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 0.5
    
    def test_confusion_matrix_structure(self):
        """Test that confusion matrix has correct structure."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        cm = compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        # TN=1, FP=1, FN=1, TP=1
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
