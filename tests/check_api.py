
import os
import glob
import base64
import requests
import random
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ /health passed:", response.json())
            # Verify new keys
            data = response.json()
            if "is_model_loaded" in data and "loaded_at" in data:
                 print("   Verified updated field names.")
            else:
                 print("   WARNING: New field names not found.")
        else:
            print("❌ /health failed:", response.status_code, response.text)
    except Exception as e:
        print("❌ /health exception:", e)

def test_metrics():
    print("\nTesting /metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            print(f"✅ /metrics passed (Length: {len(response.text)})")
        else:
            print("❌ /metrics failed:", response.status_code)
    except Exception as e:
        print("❌ /metrics exception:", e)

def test_predict_url():
    print("\nTesting /predict with URL...")
    payload = {
        "image_url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            print("✅ /predict (URL) passed:", response.json())
        else:
            print("❌ /predict (URL) failed:", response.status_code, response.text)
    except Exception as e:
        print("❌ /predict (URL) exception:", e)

def test_predict_local_files():
    print("\nTesting /predict with local assignment data (data/raw)...")
    
    # Get some random files
    cats = glob.glob("data/raw/cats/*.jpg")
    dogs = glob.glob("data/raw/dogs/*.jpg")
    
    files_to_test = []
    if cats: files_to_test.append(random.choice(cats))
    if dogs: files_to_test.append(random.choice(dogs))
    
    if not files_to_test:
        print("⚠️ No local images found in data/raw/cats or data/raw/dogs")
        return

    for file_path in files_to_test:
        print(f"Testing file: {file_path}")
        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            payload = {"image": encoded_string}
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            
            if response.status_code == 200:
                print(f"✅ /predict (Local File) passed for {os.path.basename(file_path)}:", response.json())
            else:
                print(f"❌ /predict (Local File) failed for {os.path.basename(file_path)}:", response.status_code, response.text)
        except Exception as e:
            print(f"❌ /predict (Local File) exception for {file_path}:", e)

if __name__ == "__main__":
    test_health()
    test_metrics()
    test_predict_url()
    test_predict_local_files()
