"""
Data preparation script for DVC pipeline.
Downloads and extracts the Cats vs Dogs dataset.
"""
import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Direct link to Microsoft's hosted Kaggle Cats vs Dogs dataset
DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

def download_file(url, target_path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(target_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Data") as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)

def is_valid_image(file_path):
    """Check if image file is valid and readable."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def prepare_data(data_dir='data/raw'):
    """
    Download and prepare the real dataset.
    """
    raw_dir = Path(data_dir)
    cats_dir = raw_dir / 'cats'
    dogs_dir = raw_dir / 'dogs'
    
    # Check if data already looks populated (simple check)
    if cats_dir.exists() and dogs_dir.exists():
        num_cats = len(list(cats_dir.glob('*.jpg')))
        num_dogs = len(list(dogs_dir.glob('*.jpg')))
        if num_cats > 1000 and num_dogs > 1000:
            print(f"Data already exists: {num_cats} cats, {num_dogs} dogs")
            return

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download zip if not exists
    zip_path = raw_dir / "dataset.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        try:
            download_file(DATASET_URL, zip_path)
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)
            
    # Extract
    print("Extracting dataset...")
    temp_extract_dir = raw_dir / "temp_extract"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    # Move and Clean
    # The zip contains 'PetImages/Cat' and 'PetImages/Dog'
    source_root = temp_extract_dir / "PetImages"
    
    for class_name in ['Cat', 'Dog']:
        source_class_dir = source_root / class_name
        dest_class_dir = cats_dir if class_name == 'Cat' else dogs_dir
        dest_class_dir.mkdir(exist_ok=True)
        
        print(f"Processing {class_name} images...")
        files = list(source_class_dir.glob('*'))
        
    
        valid_count = 0
        for f in tqdm(files, desc=f"Validating {class_name}"):
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                if is_valid_image(f):
                    shutil.move(str(f), str(dest_class_dir / f.name))
                    valid_count += 1
                else:
                    print(f"Skipping corrupt image: {f.name}")
                    
    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_extract_dir)
    os.remove(zip_path)
    
    print(f"Data preparation complete in {data_dir}")

if __name__ == "__main__":
    prepare_data()
