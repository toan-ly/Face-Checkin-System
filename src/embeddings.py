import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from .config import IMAGE_SIZE

# Pretrained FaceNet model
# face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def img_to_feature(image_path, model):
    img_rgb = Image.open(image_path).convert('RGB')
    img_tensor = _transform(img_rgb).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()

def crop_center_square(image):
    """
    Crop the center square of an image and resize 
    """
    width, height = image.size
    cut = min(width, height)

    left = (width - cut) // 2
    top = (height - cut) // 2
    right = (width + cut) // 2
    bottom = (height + cut) // 2
    img = image.crop((left, top, right, bottom))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return img