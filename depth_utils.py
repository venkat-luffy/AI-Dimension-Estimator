# depth_utils.py
import cv2
import torch
import numpy as np
from PIL import Image

def load_depth_model():
    """Loads the MiDaS model and transformation pipeline."""
    model_type = "MiDaS_small"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
    
    return model, transform, device

def get_depth_map(image, model, transform, device):
    """Takes a PIL image and returns a normalized depth map."""

    # --- FIX IS HERE ---
    # Convert the PIL image to a NumPy array before transforming
    image_np = np.array(image)

    img_transformed = transform(image_np).to(device) # Use the numpy array here
    with torch.no_grad():
        prediction = model(img_transformed)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

