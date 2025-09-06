import os
import pandas as pd

def load_dataset(dataset_path):
    """
    Load dataset: image paths + labels
    """
    img_paths = []
    labels = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            img_paths.append(os.path.join(dataset_path, filename))
            filename_no_ext = filename.split('.')[0]
            label = filename_no_ext[4:] # Remove "avt_" prefix
            labels.append(label)

    return pd.DataFrame({'image_path': img_paths, 'label': labels})
