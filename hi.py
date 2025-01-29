import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from astropy.io import fits

# Load image
image_data = fits.getdata("2_19468.fits")  # Replace with your FITS file

# Threshold and label objects
threshold = np.percentile(image_data, 99)  # Adjust as needed
binary_mask = image_data > threshold
labeled = label(binary_mask)

# Measure galaxy properties
regions = regionprops(labeled, intensity_image=image_data)

# Extract major axis orientation
for region in regions:
    print(f"Measured Position Angle: {region.orientation * 180 / np.pi} degrees")