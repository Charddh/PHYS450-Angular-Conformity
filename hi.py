import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.io import imread

theta_raw = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])  # Example angles
theta_clockwise = (2*np.pi - (theta_raw + np.pi)) % (2 * np.pi)
theta_limited = np.where(theta_clockwise > np.pi, theta_clockwise - np.pi, theta_clockwise)

print(315 % 180)
