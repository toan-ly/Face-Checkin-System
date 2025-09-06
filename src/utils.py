import matplotlib.pyplot as plt
from PIL import Image
from .search import search_similar_features
from .config import *
import os

def display_query_and_top_matches(query_image_path, df, top_k=5):
    query_image = Image.open(query_image_path).resize((IMAGE_SIZE, IMAGE_SIZE))

    plt.figure(figsize=(5, 5))
    plt.imshow(query_image)
    plt.axis('off')
    plt.title("Query Image")
    plt.show()

    matches = search_similar_features(query_image_path, top_k=top_k)

    top_matches = []
    for name, score, idx in matches:
        img_path = df[df['label'] == name]['image_path'].values[0]
        top_matches.append((img_path, score))

    plt.figure(figsize=(15, 5))
    for i, (name, score, img_path) in enumerate(top_matches):
        match_image = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        plt.subplot(1, top_k, i + 1)
        plt.imshow(match_image)
        plt.axis('off')
        plt.title(f"{name}\nScore: {score:.2f}")

    plt.tight_layout()
    plt.show()

def visualize_embeddings(query_embedding=None, matches=None):
    pass

def get_avt_img(employee_name):
    """
    Return avatar image path for an employee given their name
    """
    for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
        img_path = os.path.join(DATASET_PATH, f"avt_{employee_name}{ext}")
        if os.path.exists(img_path):
            return img_path
    return "https://via.placeholder.com/300?text=No+Photo"
