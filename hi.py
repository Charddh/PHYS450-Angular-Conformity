import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.io import imread

axis_bins = np.array([-45, 45, 135])
axis_bin_centres = (axis_bins[:-1] + axis_bins[1:]) / 2  # This gives bin centres
print(axis_bin_centres)  # Should contain at least 3 values