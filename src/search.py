import faiss
import numpy as np
from .config import *
from .embeddings import *

# def search_similar_images(query_image_path, top_k=TOP_K):
#     # Load the FAISS index
#     index = faiss.read_index(FEATURE_INDEX_PATH)
#     label_map = np.load(FEATURE_LABEL_MAP)

#     face_recognition_model = load_facenet_model()

#     query_vector = img_to_feature(query_image_path, model=face_recognition_model)
#     query_vector = np.array([query_vector])  # FAISS expects a 2D array

#     similarities, indices = index.search(query_vector, top_k)

#     results = []
#     for i in range(len(indices[0])):
#         employee_name = label_map[indices[0][i]]
#         similarity_score = similarities[0][i]
#         results.append((employee_name, similarity_score))

#     return results

def search_similar_features(query_embedding, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    index = faiss.read_index(FEATURE_INDEX_PATH)
    label_map = np.load(FEATURE_LABEL_MAP)

    query_vector = np.array([query_embedding]) 
    similarities, indices = index.search(query_vector, top_k)

    results = []
    for i in range(len(indices[0])):
        sim = similarities[0][i]
        if sim >= threshold:
            employee_name = label_map[indices[0][i]]
            results.append((employee_name, sim, indices[0][i]))

    return results