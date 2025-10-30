import matplotlib.pyplot as plt
import pandas as pd

# Assuming your dataframe is named df
# with columns "RA" (degrees) and "Dec" (degrees)

cluster_data = pd.read_csv("JA+A688A187mcxcii.csv")
df_clus = pd.DataFrame(cluster_data)

gal_data = pd.read_csv("table.csv")
df_gal = pd.DataFrame(gal_data)

plt.figure(figsize=(8, 6))
plt.scatter(df_clus['RAJ2000'], df_clus['DEJ2000'], s=5)  # s controls marker size
plt.xlabel('Right Ascension (degrees)')
plt.ylabel('Declination (degrees)')
plt.xlim(50,60)
plt.ylim(-20,-30)
plt.title('Sky Positions of Clusters')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df_gal['coord_ra'], df_gal['coord_dec'], s=5)  # s controls marker size
plt.xlabel('Right Ascension (degrees)')
plt.ylabel('Declination (degrees)')
plt.xlim(50,60)
plt.ylim(-20,-30)
plt.title('Sky Positions of LSST DP1 Galaxies')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()