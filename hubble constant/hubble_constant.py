import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Replace 'filename.dat' with the path to your file
ceph_df = pd.read_csv(os.path.join('hubble constant', 'ceph_parallax.dat'), delimiter=',')

ceph_df["distance_pc"] = 0.01748 / ceph_df["parallax_angle(arcsec)"]

stars = []
avg_ms = []
periods = []

# Loop through files C1_lightcurve.dat to C19_lightcurve.dat
for i in range(1, 20):
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

# Combine results into a single DataFrame
lightcurve_df = pd.DataFrame({
    'Star': stars,
    'Average Magnitude': avg_ms,
    'Period (days)': periods
})

merged_df = pd.merge(lightcurve_df, ceph_df, on='Star')

"""df = pd.read_csv(os.path.join('hubble constant', 'C2_lightcurve.dat'), delimiter=',')

plt.figure(figsize=(3, 4))
plt.plot(df["time(days)"].values, df["app_magnitude_V"].values)
plt.xlabel("Time (days)")
plt.ylabel("m")
plt.title("C2")
plt.show()"""

merged_df = merged_df[merged_df["Period (days)"].notna()]

merged_df["Absolute Magnitude"] = (merged_df["Average Magnitude"] - (5*np.log(merged_df["distance_pc"])) + 5)

merged_df["log10(P)"] = np.log(merged_df["Period (days)"])

"""x = merged_df["log10(P)"].values - 1
y = merged_df["Absolute Magnitude"].values

# Perform least squares fit (degree 1 for a line)
A, B = np.polyfit(x, y, 1)

print("Gradient (A):", A)
print("Y-intercept (B):", B)

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

#𝑀 = 𝐴 (log10(𝑃) − 1) + B

local_gals = pd.read_csv(os.path.join('hubble constant', 'localgroup.dat'), delimiter=',')

local_gals["Ceph_av_abs_magnitude_V"] = -1.3474 * (np.log(local_gals["Ceph_period(days)"] - 1) -0.2278)
local_gals["distance(pc)"] = 10 ** ((local_gals["Ceph_av_app_magnitude_V"] - local_gals["Ceph_av_abs_magnitude_V"] + 5) / 5)

print(local_gals)