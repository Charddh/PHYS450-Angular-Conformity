import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.io import imread

bins = np.arange(0, 181, 45)
bin_centres = (bins[:-1] + bins[1:]) / 2
print(bins)  # Should contain at least 3 values