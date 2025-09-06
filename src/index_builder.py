import faiss
import numpy as np
from tqdm import tqdm
from config import *
from embeddings import img_to_feature

def build_feature_index(df, model):
    index = faiss.IndexFlatIP(VECTOR_DIM_FEATURE)
    label_map = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
        try:
            features = img_to_feature(row['image_path'], model=model)
            index.add(np.array([features]))
            label_map.append(row['label'])
        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
        
    faiss.write_index(index, FEATURE_INDEX_PATH)
    np.save(FEATURE_LABEL_MAP, np.array(label_map))
    print(f"Feature index built with {index.ntotal} vectors.")

