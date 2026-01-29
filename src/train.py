"""
Training script with MLflow experiment tracking.
"""

import os
import sys
import numpy as np
import mlflow
import mlflow.pyfunc
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_dataset, split_data, create_augmented_batch
from src.model import SimpleCNN, compute_accuracy, compute_confusion_matrix



def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Cat', 'Dog']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def train_model(data_dir='data/raw', epochs=10, batch_size=16, learning_rate=0.0005, hidden_units=64):
    """Main training function with MLflow tracking."""
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cats-vs-dogs-classification")
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("input_shape", "224x224x3")
        
        print("Loading dataset...")
        # download_sample_data(data_dir) # handled by dvc

        images, labels = load_dataset(data_dir)
        
        print(f"Loaded {len(images)} images")
        mlflow.log_param("total_samples", len(images))
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(images, labels)
        
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Initialize model
        model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=hidden_units)
        
        train_losses = []
        val_losses = []
        best_val_acc = 0
        
        # Training loop
        print("\nStarting training...")
        for epoch in range(epochs):
            # Train on batches
            epoch_losses = []
            batch_gen = create_augmented_batch(X_train, y_train, batch_size, augment=True)
            n_batches = len(X_train) // batch_size
            
            for batch_idx in range(n_batches):
                batch_x, batch_y = next(batch_gen)
                loss = model.train_step(batch_x, batch_y, learning_rate)
                epoch_losses.append(loss)
            
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)
            
            # Validation
            val_pred = model.predict(X_val)
            val_probs = model.predict_proba(X_val)
            val_loss = model.compute_loss(y_val, val_probs)
            val_losses.append(val_loss)
            val_acc = compute_accuracy(y_val, val_pred)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Final evaluation on test set
        test_pred = model.predict(X_test)
        test_acc = compute_accuracy(y_test, test_pred)
        cm = compute_confusion_matrix(y_test, test_pred)
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        # Save plots
        os.makedirs('models', exist_ok=True)
        
        loss_plot_path = plot_training_curves(train_losses, val_losses, 'models/training_curves.png')
        mlflow.log_artifact(loss_plot_path)
        
        cm_plot_path = plot_confusion_matrix(cm, 'models/confusion_matrix.png')
        mlflow.log_artifact(cm_plot_path)
        
        # Save model
        model_path = 'models/cats_dogs_model.npz'
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        
        # Save metrics to JSON for DVC
        import json
        metrics = {
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc
        }
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"\nModel saved to {model_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        return model, test_acc


if __name__ == "__main__":
    model, accuracy = train_model(
        data_dir='data/raw',
        epochs=1,
        batch_size=16,
        learning_rate=0.0005,
        hidden_units=64
    )
    print(f"\nTraining complete! Final test accuracy: {accuracy:.4f}")
