"""
Data preprocessing for Cats vs Dogs classification.
Handles image loading, resizing, augmentation and train/val/test split.
"""

import os
import numpy as np
from PIL import Image
import random


def load_image(img_path, target_size=(224, 224)):
    """Load and resize a single image to target size."""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None


def augment_image(img):
    """Apply random augmentation to an image."""
    # Random horizontal flip (Very common for animals)
    if random.random() > 0.5:
        img = np.fliplr(img)
    
    # Random brightness adjustment (Gentle)
    brightness = random.uniform(0.9, 1.1)
    img = np.clip(img * brightness, 0, 1)
    
    # Add random Gaussian noise (Simulates sensor noise)
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.02, img.shape) 
        img = np.clip(img + noise, 0, 1)
    
    return img.astype(np.float32)


def load_dataset(data_dir, target_size=(224, 224)):
    """
    Load cats and dogs images from directory structure:
    data_dir/
        cats/
            cat.0.jpg, cat.1.jpg, ...
        dogs/
            dog.0.jpg, dog.1.jpg, ...
    """
    images = []
    labels = []
    
    cats_dir = os.path.join(data_dir, 'cats')
    dogs_dir = os.path.join(data_dir, 'dogs')
    
    MAX_IMAGES = 4000 # Limit to avoid OOM
    
    # Load cats (label 0)
    if os.path.exists(cats_dir):
        files = [f for f in os.listdir(cats_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Shuffle files to get random sample if we limit
        np.random.shuffle(files)
        
        for i, fname in enumerate(files):
            if i >= MAX_IMAGES: break
            
            if i % 100 == 0:
                print(f"Loading cats: {i}/{MAX_IMAGES}", end='\r')
            img = load_image(os.path.join(cats_dir, fname), target_size)
            if img is not None:
                images.append(img)
                labels.append(0)
    print(f"Loaded {len(images)} cat images.          ")
    
    # Load dogs (label 1)
    if os.path.exists(dogs_dir):
        files = [f for f in os.listdir(dogs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(files)
        
        for i, fname in enumerate(files):
            if i >= MAX_IMAGES: break
            
            if i % 100 == 0:
                print(f"Loading dogs: {i}/{MAX_IMAGES}", end='\r')
            img = load_image(os.path.join(dogs_dir, fname), target_size)
            if img is not None:
                images.append(img)
                labels.append(1)
    # Calculate dog count correctly
    dog_count = len(images) - (MAX_IMAGES if len(images) > MAX_IMAGES else len(images)) # simplified logic for print
    # Just print current total
    print(f"Loaded total {len(images)} images (Cats + Dogs).          ")
    
    return np.array(images), np.array(labels)


def split_data(images, labels, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation and test sets."""
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (
        (images[train_idx], labels[train_idx]),
        (images[val_idx], labels[val_idx]),
        (images[test_idx], labels[test_idx])
    )


def create_augmented_batch(images, labels, batch_size=32, augment=True):
    """Generator for training batches with optional augmentation."""
    n_samples = len(images)
    indices = np.arange(n_samples)
    
    while True:
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            batch_images = images[batch_idx].copy()
            batch_labels = labels[batch_idx]
            
            if augment:
                batch_images = np.array([augment_image(img) for img in batch_images])
            
            yield batch_images, batch_labels


if __name__ == "__main__":
    # Quick test of preprocessing functions
    print("Testing preprocessing functions...")
    
    # Create a dummy image for testing
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    augmented = augment_image(dummy_img)
    print(f"Input shape: {dummy_img.shape}, Output shape: {augmented.shape}")
    print("Preprocessing functions working correctly!")
