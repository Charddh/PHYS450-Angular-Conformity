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

max_z = 0.1 #maximum redshift of galaxies being looked for
min_n = 30 #minimum companions according to cluster catalogue

# Load cluster data from a FITS file, storing in a Table and NumPy array
with fits.open('catCluster-SPIDERS_RASS_CLUS-v3.0.fits') as cluster:
    cluster_table = Table(cluster[1].data)
    cluster_data = np.array(list(cluster[1].data))

# Load BCG data from a FITS file, storing in a Table and NumPy array
with fits.open('SpidersXclusterBCGs-v2.0.fits') as bcg:
    bcg_table = Table(bcg[1].data)
    bcg_data = np.array(list(bcg[1].data))

# Load Galaxy Zoo data from a FITS file, storing in a Table and NumPy array
with fits.open('GZDR1SFRMASS.fits') as galzoo:
    galzoo_table = Table(galzoo[1].data)
    galzoo_data = np.array(Table(galzoo[1].data))
# Convert the Table to a pandas DataFrame
galzoo_df = galzoo_table.to_pandas()
# Convert to a 2D NumPy array
galzoo_data = galzoo_df.to_numpy()

# Apply filters to select clusters based on redshift and minimum companion count
mask = (cluster_data[:, 12].astype(float) < max_z) & (cluster_data[:, 18].astype(float) > min_n)
cluster_id = cluster_data[mask, 0] # Extract filtered cluster IDs
cluster_ra = cluster_data[mask, 3] # Extract Right Ascension of filtered clusters
cluster_dec = cluster_data[mask, 4] # Extract Declination of filtered clusters
cluster_z = cluster_data[mask, 12] # Extract redshift of filtered clusters
cluster_n = cluster_data[mask, 18] # Extract companion count of filtered clusters

# Filter BCG data to match selected clusters
mask = np.isin(bcg_data[:, 0], cluster_id)
reduced_clusters_id = bcg_data[mask, 0].astype(float) # Filtered BCG cluster IDs
reduced_clusters_z = bcg_data[mask, 1].astype(float) # Filtered BCG redshifts
reduced_clusters_ra = bcg_data[mask, 2].astype(float) # Filtered BCG Right Ascensions
reduced_clusters_dec = bcg_data[mask, 3].astype(float) # Filtered BCG Declinations

# Extract data from Galaxy Zoo catalog
gz_id = galzoo_data[:, 0] # Galaxy IDs
gz_ra = galzoo_data[:, 13] # Right Ascension of galaxies
gz_dec = galzoo_data[:, 14] # Declination of galaxies
gz_z = galzoo_data[:, 18] # Redshift of galaxies



"""for i in range(len(reduced_clusters_id)):
    dec_diff = reduced_clusters_dec[i] - gz_dec
    ra_diff = reduced_clusters_ra[i] - gz_ra
    z_diff = reduced_clusters_z[i] - gz_z
    if (np.sqrt((ra_diff ** 2) * (np.cos(reduced_clusters_dec[j]) ** 2) + ((reduced_clusters_dec[j] - gz_dec[i]) ** 2)) < ((1000 / 3600) * cosmo.arcsec_per_kpc_proper(float(reduced_clusters_z[j])).value)):# and ((reduced_clusters_z[j] - 0.01) < gz_z[i] < (reduced_clusters_z[j] + 0.01)):
            templist.append(gz_id[i])
    reduced_clusters_locals.append(templist)


for i in range(len())

print(dec_diff)
print(ra_diff)
print(z_diff)
print(min(abs(dec_diff)))
print(min(abs(ra_diff)))
print(min(abs(z_diff)))"""



reduced_clusters_locals = []



ang_diff = []
z_diff = []

for j in range(len(reduced_clusters_id)):
    gz_id_templist = []
    for i in range(len(gz_id)):
        z_diff = abs(reduced_clusters_z[j] - gz_z[i])
        if (np.sqrt(((reduced_clusters_ra[j] - gz_ra[i]) ** 2) * (np.cos(reduced_clusters_dec[j]) ** 2) + ((reduced_clusters_dec[j] - gz_dec[i]) ** 2)) < ((1000 / 3600) * cosmo.arcsec_per_kpc_proper(float(reduced_clusters_z[j])).value)) and z_diff < 0.01:
            gz_id_templist.append(gz_id[i])
    reduced_clusters_locals.append(gz_id_templist)

print(len(reduced_clusters_id))
print(reduced_clusters_locals)
print(len(reduced_clusters_locals))