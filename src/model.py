"""
Simple CNN model for Cats vs Dogs binary classification.
"""

import numpy as np


class SimpleCNN:
    """
    A basic CNN-like model using only NumPy.
    This is a simplified version that flattens images and uses dense layers.
    For production, you'd use TensorFlow/PyTorch, but this demonstrates the concept.
    """
    
    def __init__(self, input_shape=(224, 224, 3), hidden_units=128):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.hidden_units = hidden_units
        
        # Initialize weights with Xavier initialization
        self.weights = {
            'hidden': np.random.randn(self.input_size, hidden_units) * np.sqrt(2.0 / self.input_size),
            'output': np.random.randn(hidden_units, 1) * np.sqrt(2.0 / hidden_units)
        }
        self.biases = {
            'hidden': np.zeros(hidden_units),
            'output': np.zeros(1)
        }
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Hidden layer
        hidden = self.relu(np.dot(x_flat, self.weights['hidden']) + self.biases['hidden'])
        
        # Output layer
        output = self.sigmoid(np.dot(hidden, self.weights['output']) + self.biases['output'])
        
        return output.flatten()
    
    def predict(self, x):
        """Get class predictions (0 or 1)."""
        probs = self.forward(x)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, x):
        """Get probability of class 1 (dog)."""
        return self.forward(x)
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def train_step(self, x, y, learning_rate=0.001):
        """One training step with basic gradient descent."""
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Forward pass
        hidden_input = np.dot(x_flat, self.weights['hidden']) + self.biases['hidden']
        hidden_output = self.relu(hidden_input)
        
        output_input = np.dot(hidden_output, self.weights['output']) + self.biases['output']
        y_pred = self.sigmoid(output_input).flatten()
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward pass
        d_output = (y_pred - y).reshape(-1, 1) / batch_size
        
        d_weights_output = np.dot(hidden_output.T, d_output)
        d_biases_output = np.sum(d_output, axis=0)
        
        d_hidden = np.dot(d_output, self.weights['output'].T)
        d_hidden[hidden_input <= 0] = 0  # ReLU derivative
        
        d_weights_hidden = np.dot(x_flat.T, d_hidden)
        d_biases_hidden = np.sum(d_hidden, axis=0)
        
        # Update weights
        self.weights['output'] -= learning_rate * d_weights_output
        self.biases['output'] -= learning_rate * d_biases_output
        self.weights['hidden'] -= learning_rate * d_weights_hidden
        self.biases['hidden'] -= learning_rate * d_biases_hidden
        
        return loss
    
    def save(self, filepath):
        """Save model weights to file."""
        np.savez(filepath, 
                 weights_hidden=self.weights['hidden'],
                 weights_output=self.weights['output'],
                 biases_hidden=self.biases['hidden'],
                 biases_output=self.biases['output'],
                 input_shape=self.input_shape,
                 hidden_units=self.hidden_units)
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        data = np.load(filepath)
        model = cls(tuple(data['input_shape']), int(data['hidden_units']))
        model.weights['hidden'] = data['weights_hidden']
        model.weights['output'] = data['weights_output']
        model.biases['hidden'] = data['biases_hidden']
        model.biases['output'] = data['biases_output']
        return model


def compute_accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


if __name__ == "__main__":
    # Quick test
    print("Testing model...")
    model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
    dummy_input = np.random.rand(4, 224, 224, 3).astype(np.float32)
    dummy_labels = np.array([0, 1, 0, 1])
    
    pred = model.predict(dummy_input)
    probs = model.predict_proba(dummy_input)
    print(f"Predictions: {pred}")
    print(f"Probabilities: {probs}")
    print("Model working correctly!")
