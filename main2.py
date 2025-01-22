import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.io import ascii #astropy is an astronomy module but here we are just importing its data read in functionality. If astropy not included in your installation you can see how to get it here https://www.astropy.org/
from astropy.table import Table, Column, MaskedColumn# this is astropy's way to make a data table, for reading out data
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import glob #allows you to read the names of files in folders 
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

max_z = 0.1
min_n = 30

# Open the FITS file and retrieve the data from the second HDU (Header Data Unit)
cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data
# Convert the structured array to a dictionary with byte-swapping and endian conversion for each column
cluster_df = pd.DataFrame({
    name: cluster_data[name].byteswap().newbyteorder()  # Apply byte-swapping and endian conversion to each field
    for name in cluster_data.dtype.names  # Iterate over each field name in the structured array
})

gz_df = pd.DataFrame(fits.open("GZDR1SFRMASS.fits")[1].data)
bcg_df = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

print(gz_df.columns)
#print(bcg_df.columns)

cluster_df2 = cluster_df[(cluster_df['SCREEN_CLUZSPEC'] < max_z) & (cluster_df['SCREEN_NMEMBERS_W'] > min_n)]

# Extract relevant columns from the filtered clusters DataFrame
cluster_id = cluster_df2['CLUS_ID'].values
cluster_ra = cluster_df2['RA'].values
cluster_dec = cluster_df2['DEC'].values
cluster_z = cluster_df2['SCREEN_CLUZSPEC'].values

# Extract relevant columns from the BCG DataFrame (match by CLUS_ID)
bcg_df2 = bcg_df[bcg_df['CLUS_ID'].isin(cluster_id)]
reduced_clusters_id = bcg_df2['CLUS_ID'].values
reduced_clusters_ra = bcg_df2['RA_BCG'].values
reduced_clusters_dec = bcg_df2['DEC_BCG'].values
reduced_clusters_z = bcg_df2['CLUZSPEC'].values

# Extract relevant columns from the Galaxy Zoo DataFrame
gz_id = gz_df['SPECOBJID_1'].values
gz_ra = gz_df['RA_1'].values
gz_dec = gz_df['DEC_1'].values
gz_z = gz_df['Z'].values

# Calculate the angular separation and redshift difference using vectorized operations
ra_diff = reduced_clusters_ra[:, None] - gz_ra  # Broadcast the RA difference calculation
dec_diff = reduced_clusters_dec[:, None] - gz_dec  # Broadcast the Dec difference calculation
z_diff = np.abs(reduced_clusters_z[:, None] - gz_z)  # Compute absolute redshift difference

# Compute the angular separation using the Haversine formula and the proper scaling
angular_separation = np.sqrt((ra_diff ** 2) * (np.cos(np.radians(reduced_clusters_dec[:, None])) ** 2) + dec_diff ** 2)

# Convert the angular separation to physical separation in kpc
phys_sep_kpc = (1000 / 3600) * cosmo.arcsec_per_kpc_proper(reduced_clusters_z[:, None]).value

# Apply the selection criteria (angular separation and redshift difference)
selected_galaxies_mask = (angular_separation < phys_sep_kpc) & (z_diff < 0.01)

# Create a list of Galaxy IDs for each cluster
reduced_clusters_locals = [gz_id[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]

# Print the results
print(len(reduced_clusters_locals))
print(len(reduced_clusters_id))

# Convert reduced_clusters_locals into a DataFrame
df = pd.DataFrame({'Cluster_ID': reduced_clusters_id, 'Galaxy_IDs': reduced_clusters_locals})

# Save to CSV
df.to_csv('reduced_clusters_locals_main2.csv', index=False)
