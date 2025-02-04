import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.io import imread

# Load image
image_data = imread("2_19468.png")  # Replace with your PNG file

# If image has an alpha channel (RGBA), remove it
if image_data.shape[-1] == 4:
    image_data = image_data[:, :, :3]  # Keep only RGB channels

# Convert to grayscale if needed
if image_data.ndim == 3:  
    image_data = rgb2gray(image_data)

# Threshold and label objects
threshold = np.percentile(image_data, 99)  # Adjust as needed
binary_mask = image_data > threshold
labeled = label(binary_mask)

# Measure galaxy properties
regions = regionprops(labeled, intensity_image=image_data)

print("hi")

# Extract major axis orientation
for region in regions:
    print(f"Measured Position Angle: {region.orientation * 180 / np.pi} degrees")