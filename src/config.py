import os

DATASET_PATH: str = os.path.join("data", "Dataset")
RAW_INDEX_PATH: str = os.path.join('data', 'employee_images.index')
RAW_LABEL_MAP: str = os.path.join('data', 'label_map.npy')

FEATURE_INDEX_PATH: str = os.path.join('data', 'facenet_features.index')
FEATURE_LABEL_MAP: str = os.path.join('data', 'facenet_label_map.npy')

IMAGE_SIZE: int = 300
VECTOR_DIM_RAW: int = IMAGE_SIZE * IMAGE_SIZE * 3
VECTOR_DIM_FEATURE: int = 512

SIMILARITY_THRESHOLD: float = 0.3
TOP_K: int = 5