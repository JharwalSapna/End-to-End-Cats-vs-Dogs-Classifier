
import requests

BASE_URL = "http://localhost:8000/predict"

candidates = [
    # Cats
    ("Cat 1", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"),
    ("Cat 2", "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"),
    ("Cat 3", "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"),
    
    # Dogs
    ("Dog 1", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Siberian_Husky_pho.jpg/320px-Siberian_Husky_pho.jpg"),
    ("Dog 2", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Black_Labrador_Retriever_-_Male_IMG_3323.jpg/320px-Black_Labrador_Retriever_-_Male_IMG_3323.jpg"),

    # Neutral URLs (Unsplash)
    ("Unsplash Cat 1", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"),
    ("Unsplash Dog 1", "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"),
]

for name, url in candidates:
    try:
        resp = requests.post(BASE_URL, json={"image_url": url})
        if resp.status_code == 200:
            print(f"{name}: {resp.json()}")
        else:
            print(f"{name}: Failed {resp.status_code}")
    except Exception as e:
        print(f"{name}: Error {e}")
