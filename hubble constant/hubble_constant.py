import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Replace 'filename.dat' with the path to your file
ceph_df = pd.read_csv(os.path.join('hubble constant', 'ceph_parallax.dat'), delimiter=',')

ceph_df["distance_pc"] = 1 / ceph_df["parallax_angle(arcsec)"]

stars = []
avg_ms = []
periods = []

# Loop through files C1_lightcurve.dat to C19_lightcurve.dat
for i in range(1, 21):
    filename = f'C{i}_lightcurve.dat'
    df = pd.read_csv(os.path.join('hubble constant', filename), delimiter=',')
    
    star = f'C{i}'
    stars.append(star)
    
    # Mean apparent magnitude
    avg_m = df["app_magnitude_V"].mean()
    avg_ms.append(avg_m)
    
    # Find period (difference between first two peaks)
    time = df['time(days)'].values
    mag = df['app_magnitude_V'].values
    peaks = []
    
    for j in range(1, len(mag) - 1):
        if mag[j] > mag[j - 1] and mag[j] > mag[j + 1]:
            peaks.append(time[j])
    
    if len(peaks) >= 2:
        period = peaks[1] - peaks[0]
        periods.append(period)
    else:
        periods.append(None)

periods = [34,1.4,2.2,2.6,20,16,1.4,23,20,4.6,8,25,14.5,8,2,1.05,15.5,26,40,1.3]
print(len(periods))

# Create DataFrame with quality indicators
lightcurve_df = pd.DataFrame({
    'Star': stars,
    'Average Magnitude': avg_ms,
    'Period (days)': periods
})

merged_df = pd.merge(lightcurve_df, ceph_df, on='Star')

print("head", merged_df.head())

"""i = 5
filename = f'C{i}_lightcurve.dat'
df = pd.read_csv(os.path.join('hubble constant', filename), delimiter=',')
plt.figure(figsize=(8, 4))
plt.scatter(df['time(days)'], df["app_magnitude_V"], label="Data")
# Add MAJOR grid lines (darker, more prominent)
plt.grid(True, which='major', axis='x', alpha=0.7, linestyle='-', linewidth=0.8)
plt.grid(True, which='major', axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
# Add MINOR grid lines (lighter, more frequent)
plt.grid(True, which='minor', axis='x', alpha=0.2, linestyle=':', linewidth=0.5)
plt.grid(True, which='minor', axis='y', alpha=0.2, linestyle=':', linewidth=0.3)
# Set custom tick locations for MORE gridlines
from matplotlib.ticker import MultipleLocator
# Adjust these values based on your time range:
time_range = df['time(days)'].max() - df['time(days)'].min()
if time_range <= 10:
    major_interval = 1    # Major grid every 1 day
    minor_interval = 0.2  # Minor grid every 0.2 days (5 per major division)
elif time_range <= 50:
    major_interval = 5    # Major grid every 5 days
    minor_interval = 1    # Minor grid every 1 day
else:
    major_interval = 10   # Major grid every 10 days
    minor_interval = 2    # Minor grid every 2 days
plt.gca().xaxis.set_major_locator(MultipleLocator(major_interval))
plt.gca().xaxis.set_minor_locator(MultipleLocator(minor_interval))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))   # Major grid every 0.5 mag
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))   # Minor grid every 0.1 mag
plt.xlabel("Time (Days)")
plt.ylabel("Apparent Magnitude V")
plt.title(f"Light Curve for C{i}")
plt.legend()
plt.gca().invert_yaxis()  # Important for magnitude scale
plt.tight_layout()
plt.show()"""

"""df = pd.read_csv(os.path.join('hubble constant', 'C2_lightcurve.dat'), delimiter=',')

plt.figure(figsize=(3, 4))
plt.plot(df["time(days)"].values, df["app_magnitude_V"].values)
plt.xlabel("Time (days)")
plt.ylabel("m")
plt.title("C2")
plt.show()"""

merged_df = merged_df[merged_df["Period (days)"].notna()]

merged_df["Absolute Magnitude"] = (merged_df["Average Magnitude"] - (5*np.log10(merged_df["distance_pc"])) + 5)

merged_df["log10(P)"] = np.log10(merged_df["Period (days)"])

x = merged_df["log10(P)"].values - 1
y = merged_df["Absolute Magnitude"].values

# Perform least squares fit (degree 1 for a line)
A, B = np.polyfit(x, y, 1)

print("Gradient (A):", A)
print("Y-intercept (B):", B)
"""
# Plot data
plt.figure(figsize=(3, 4))
plt.scatter(x, y, label="Data")

# Plot best-fit line
plt.plot(x, A*x + B, color='red', label=f'Fit: M = {A:.2f}*(log10(P)-1) + {B:.2f}')
plt.xlabel("log10(P) - 1")
plt.ylabel("M")
plt.title("Leavitt Law")
plt.legend()
plt.show()"""

# Plot to visualize the fit
plt.figure(figsize=(8, 6))
plt.scatter(merged_df["log10(P)"], merged_df["Absolute Magnitude"], label="Cepheid Data")
x_fit = np.linspace(merged_df["log10(P)"].min(), merged_df["log10(P)"].max(), 100)
plt.plot(x_fit, A*(x_fit-1) + B, 'r-', label=f'Fit: M = {A:.2f}×(logP - 1) + {B:.2f}')
plt.xlabel("log10(P)")
plt.ylabel("Absolute Magnitude M")
plt.title("Leavitt Law Calibration")
plt.legend()
plt.gca().invert_yaxis()  # Brighter = lower values
plt.show()

#𝑀 = 𝐴 (log10(𝑃) − 1) + B

local_gals = pd.read_csv(os.path.join('hubble constant', 'localgroup.dat'), delimiter=',')

# Calculate for all galaxies in local_gals
logP_all = np.log10(local_gals["Ceph_period(days)"])
logP_minus_1_all = logP_all - 1
M_all = A * logP_minus_1_all + B

local_gals["Ceph_av_abs_magnitude_V"] = M_all
local_gals["distance_modulus"] = local_gals["Ceph_av_app_magnitude_V"] - local_gals["Ceph_av_abs_magnitude_V"]
local_gals["distance(pc)"] = 10 ** ((local_gals["distance_modulus"] + 5) / 5)

gal_df = pd.read_csv(os.path.join('hubble constant', 'SN_lowz.dat'), delimiter=',')

galaxies = []
oiii = []

# Loop through files C1_lightcurve.dat to C19_lightcurve.dat
for i in range(1, 20):
    filename = f'G{i}_spec.dat'
    df = pd.read_csv(os.path.join('hubble constant', filename), delimiter=',')
    
    galaxy = f'G{i}'
    galaxies.append(galaxy)
    
    oiii.append(df.loc[df['flux(rel)'].idxmax(), 'wavelength(Ang)'])

lightcurve_df = pd.DataFrame({
    'GalaxyName': galaxies,
    'oiii': oiii
})

merged_df = pd.merge(lightcurve_df, gal_df, on='GalaxyName')

merged_df["z"] = (merged_df["oiii"] - 5007)/ 5007
merged_df["v(kms)"] = 3e5 * merged_df["z"]
SN_absVmag = 5.13 - 5*np.log10(local_gals.loc[local_gals["Galaxy"] == "Andromeda", "distance(pc)"]) + 5
SN_absVmag_value = SN_absVmag.iloc[0]  # Extract the first (and only) value
print("SN Absolute Magnitude:", SN_absVmag_value)

# FIXED: Use correct distance modulus formula: m - M = 5*log10(d) - 5
# So: d = 10^((m - M + 5)/5)
merged_df["distance(pc)"] = 10 ** ((merged_df["SN_apparentVmag"] - SN_absVmag_value + 5) / 5)

merged_df["distance(Mpc)"] = merged_df["distance(pc)"] / 1e6

x = merged_df["distance(Mpc)"].values
y = merged_df["v(kms)"].values

print("x",x)
print("y",y)

# Perform least squares fit (degree 1 for a line)
A, B = np.polyfit(x, y, 1)

print("Gradient (A):", A)
print("Y-intercept (B):", B)

# Plot data
plt.figure(figsize=(3, 4))
plt.scatter(x, y, label="Data")

# Plot best-fit line
plt.plot(x, A*x + B, color='red', label=f'Fit: v = {A:.2f}*d + {B:.2f}')
plt.xlabel("Distance [Mpc]")
plt.ylabel("Velocity [kms^-1]")
plt.title("Hubble Diagram")
plt.legend()
plt.show()
