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
        
        # Architecture: Input -> h1 -> h2 -> h3 -> Output
        # hidden_units arg sets the size of the FIRST hidden layer
        self.h1_size = hidden_units
        self.h2_size = hidden_units // 2
        self.h3_size = hidden_units // 4
        
        # Initialize weights with He initialization
        self.weights = {
            'h1': np.random.randn(self.input_size, self.h1_size) * np.sqrt(2.0 / self.input_size),
            'h2': np.random.randn(self.h1_size, self.h2_size) * np.sqrt(2.0 / self.h1_size),
            'h3': np.random.randn(self.h2_size, self.h3_size) * np.sqrt(2.0 / self.h2_size),
            'out': np.random.randn(self.h3_size, 1) * np.sqrt(2.0 / self.h3_size)
        }
        self.biases = {
            'h1': np.zeros(self.h1_size),
            'h2': np.zeros(self.h2_size),
            'h3': np.zeros(self.h3_size),
            'out': np.zeros(1)
        }
        
        # Momentum state
        self.velocity = {
            'w_h1': np.zeros_like(self.weights['h1']),
            'w_h2': np.zeros_like(self.weights['h2']),
            'w_h3': np.zeros_like(self.weights['h3']),
            'w_out': np.zeros_like(self.weights['out']),
            'b_h1': np.zeros_like(self.biases['h1']),
            'b_h2': np.zeros_like(self.biases['h2']),
            'b_h3': np.zeros_like(self.biases['h3']),
            'b_out': np.zeros_like(self.biases['out'])
        }
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Hidden layer 1
        self.z1 = np.dot(x_flat, self.weights['h1']) + self.biases['h1']
        self.a1 = self.relu(self.z1)
        
        # Hidden layer 2
        self.z2 = np.dot(self.a1, self.weights['h2']) + self.biases['h2']
        self.a2 = self.relu(self.z2)
        
        # Hidden layer 3
        self.z3 = np.dot(self.a2, self.weights['h3']) + self.biases['h3']
        self.a3 = self.relu(self.z3)
        
        # Output layer
        self.z4 = np.dot(self.a3, self.weights['out']) + self.biases['out']
        output = self.sigmoid(self.z4)
        
        return output.flatten()
    
    def predict(self, x):
        """Get class predictions (0 or 1)."""
        probs = self.forward(x)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, x):
        """Get probability of class 1 (dog)."""
        return self.forward(x)
    
    def compute_loss(self, y_true, y_pred, reg_lambda=0, weights=None):
        """Binary cross-entropy loss with L2 regularization."""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if reg_lambda > 0 and weights is not None:
            l2_loss = 0
            for w in weights.values():
                l2_loss += np.sum(w**2)
            return bce_loss + 0.5 * reg_lambda * l2_loss
            
        return bce_loss
    
    def train_step(self, x, y, learning_rate=0.001, reg_lambda=0.001, dropout_rate=0.2):
        """One training step with gradient descent, momentum, dropout, and L2 regularization."""
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Forward pass with Dropout
        z1 = np.dot(x_flat, self.weights['h1']) + self.biases['h1']
        a1 = self.relu(z1)
        # Dropout 1
        mask1 = (np.random.rand(*a1.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
        a1 *= mask1
        
        z2 = np.dot(a1, self.weights['h2']) + self.biases['h2']
        a2 = self.relu(z2)
        # Dropout 2
        mask2 = (np.random.rand(*a2.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
        a2 *= mask2
        
        z3 = np.dot(a2, self.weights['h3']) + self.biases['h3']
        a3 = self.relu(z3)
        # Dropout 3
        mask3 = (np.random.rand(*a3.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
        a3 *= mask3
        
        z4 = np.dot(a3, self.weights['out']) + self.biases['out']
        y_pred = self.sigmoid(z4).flatten()
        
        # Loss
        loss = self.compute_loss(y, y_pred, reg_lambda, self.weights)
        
        # Backward pass
        # Output layer gradients
        d_z4 = (y_pred - y).reshape(-1, 1) / batch_size
        
        d_w_out = np.dot(a3.T, d_z4) + reg_lambda * self.weights['out']
        d_b_out = np.sum(d_z4, axis=0)
        
        # Hidden layer 3 gradients
        d_a3 = np.dot(d_z4, self.weights['out'].T)
        d_a3 *= mask3 # Apply dropout mask
        d_z3 = d_a3 * (z3 > 0)
        
        d_w_h3 = np.dot(a2.T, d_z3) + reg_lambda * self.weights['h3']
        d_b_h3 = np.sum(d_z3, axis=0)
        
        # Hidden layer 2 gradients
        d_a2 = np.dot(d_z3, self.weights['h3'].T)
        d_a2 *= mask2 # Apply dropout mask
        d_z2 = d_a2 * (z2 > 0)
        
        d_w_h2 = np.dot(a1.T, d_z2) + reg_lambda * self.weights['h2']
        d_b_h2 = np.sum(d_z2, axis=0)
        
        # Hidden layer 1 gradients
        d_a1 = np.dot(d_z2, self.weights['h2'].T)
        d_a1 *= mask1 # Apply dropout mask
        d_z1 = d_a1 * (z1 > 0)
        
        d_w_h1 = np.dot(x_flat.T, d_z1) + reg_lambda * self.weights['h1']
        d_b_h1 = np.sum(d_z1, axis=0)
        
        # Updates with Momentum
        beta = 0.9
        
        # Output layer
        self.velocity['w_out'] = beta * self.velocity['w_out'] - learning_rate * d_w_out
        self.velocity['b_out'] = beta * self.velocity['b_out'] - learning_rate * d_b_out
        self.weights['out'] += self.velocity['w_out']
        self.biases['out'] += self.velocity['b_out']
        
        # Hidden layer 3
        self.velocity['w_h3'] = beta * self.velocity['w_h3'] - learning_rate * d_w_h3
        self.velocity['b_h3'] = beta * self.velocity['b_h3'] - learning_rate * d_b_h3
        self.weights['h3'] += self.velocity['w_h3']
        self.biases['h3'] += self.velocity['b_h3']
        
        # Hidden layer 2
        self.velocity['w_h2'] = beta * self.velocity['w_h2'] - learning_rate * d_w_h2
        self.velocity['b_h2'] = beta * self.velocity['b_h2'] - learning_rate * d_b_h2
        self.weights['h2'] += self.velocity['w_h2']
        self.biases['h2'] += self.velocity['b_h2']
        
        # Hidden layer 1
        self.velocity['w_h1'] = beta * self.velocity['w_h1'] - learning_rate * d_w_h1
        self.velocity['b_h1'] = beta * self.velocity['b_h1'] - learning_rate * d_b_h1
        self.weights['h1'] += self.velocity['w_h1']
        self.biases['h1'] += self.velocity['b_h1']
        
        return loss
    
    def save(self, filepath):
        """Save model weights to file."""
        np.savez(filepath, 
                 w_h1=self.weights['h1'], b_h1=self.biases['h1'],
                 w_h2=self.weights['h2'], b_h2=self.biases['h2'],
                 w_h3=self.weights['h3'], b_h3=self.biases['h3'],
                 w_out=self.weights['out'], b_out=self.biases['out'],
                 input_shape=self.input_shape,
                 hidden_units=self.h1_size)
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        data = np.load(filepath)
        model = cls(tuple(data['input_shape']), int(data['hidden_units']))
        model.weights['h1'] = data['w_h1']
        model.biases['h1'] = data['b_h1']
        model.weights['h2'] = data['w_h2']
        model.biases['h2'] = data['b_h2']
        model.weights['h3'] = data['w_h3']
        model.biases['h3'] = data['b_h3']
        model.weights['out'] = data['w_out']
        model.biases['out'] = data['b_out']
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
