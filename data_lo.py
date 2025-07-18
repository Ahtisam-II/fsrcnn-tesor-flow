import cv2
import os
import numpy as np
from glob import glob

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generate_lr_hr_pair(img, scale=3, hr_size=(768, 768)):
    """
    Resize HR image to fixed size hr_size,
    then downscale by scale factor to generate LR.
    """
    # Resize HR image to fixed size
    hr = cv2.resize(img, hr_size, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(hr,ksize=(3,3),sigmaX=1.0)

    # Downscale HR to get LR image
    lr_size = (hr_size[0] // scale, hr_size[1] // scale)
    lr = cv2.resize(blurred, lr_size, interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    lr = lr.astype(np.float32) / 255.0
    hr = hr.astype(np.float32) / 255.0

    return lr, hr

def load_dataset_from_folder(folder_path, scale=3, hr_size=(768, 768)):
    lr_images = []
    hr_images = []

    image_paths = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))

    for img_path in image_paths:
        try:
            img = load_image(img_path)
            lr, hr = generate_lr_hr_pair(img, scale=scale, hr_size=hr_size)
            lr_images.append(lr)
            hr_images.append(hr)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # Return lists (not np arrays to avoid shape issues)
    return lr_images, hr_images
