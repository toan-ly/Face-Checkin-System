import os

DATASET_PATH = os.path.join("data", "Dataset")
RAW_INDEX_PATH = os.path.join('data', 'employee_images.index')
RAW_LABEL_MAP = os.path.join('data', 'label_map.npy')

FEATURE_INDEX_PATH = os.path.join('data', 'facenet_features.index')
FEATURE_LABEL_MAP = os.path.join('data', 'facenet_label_map.npy')

IMAGE_SIZE = 300
VECTOR_DIM_RAW = IMAGE_SIZE * IMAGE_SIZE * 3
VECTOR_DIM_FEATURE = 512

SIMILARITY_THRESHOLD = 0.3