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

with fits.open('catCluster-SPIDERS_RASS_CLUS-v3.0.fits') as cluster:
    cluster_table = Table(cluster[1].data)
    cluster_data = np.array(list(cluster[1].data))

with fits.open('SpidersXclusterBCGs-v2.0.fits') as bcg:
    bcg_table = Table(bcg[1].data)
    bcg_data = np.array(list(bcg[1].data))

with fits.open('GZDR1SFRMASS.fits') as galzoo:
    galzoo_table = Table(galzoo[1].data)
    galzoo_data = np.array(Table(galzoo[1].data))
# Convert the Table to a pandas DataFrame
galzoo_df = galzoo_table.to_pandas()
# Convert to a 2D NumPy array
galzoo_data = galzoo_df.to_numpy()

mask = (cluster_data[:, 12].astype(float) < max_z) & (cluster_data[:, 18].astype(float) > min_n)
cluster_id = cluster_data[mask, 0]
cluster_ra = cluster_data[mask, 3]
cluster_dec = cluster_data[mask, 4]
cluster_z = cluster_data[mask, 12]
cluster_n = cluster_data[mask, 18]

mask = np.isin(bcg_data[:, 0], cluster_id)
reduced_clusters_id = bcg_data[mask, 0].astype(float)
reduced_clusters_z = bcg_data[mask, 1].astype(float)
reduced_clusters_ra = bcg_data[mask, 2].astype(float)
reduced_clusters_dec = bcg_data[mask, 3].astype(float)

gz_id = galzoo_data[:, 0]
gz_ra = galzoo_data[:, 13]
gz_dec = galzoo_data[:, 14]
gz_z = galzoo_data[:, 18]

reduced_clusters_locals = []

"""print(reduced_clusters_ra)
print(np.sqrt(((reduced_clusters_ra[0] - gz_ra[0]) ** 2) * (np.cos(reduced_clusters_dec[0]) ** 2) + ((reduced_clusters_dec[0] - gz_dec[0]) ** 2)))
print(((1000 / 3600) * cosmo.arcsec_per_kpc_proper(float(0.0453)).value))
print((reduced_clusters_z[0] - 0.01))
print(gz_z[0])"""

ang_diff = []
z_diff = []

for j in range(len(reduced_clusters_id)):
    templist = []
    for i in range(len(gz_id)):
        if (np.sqrt(((reduced_clusters_ra[j] - gz_ra[i]) ** 2) * (np.cos(reduced_clusters_dec[j]) ** 2) + ((reduced_clusters_dec[j] - gz_dec[i]) ** 2)) < ((1000 / 3600) * cosmo.arcsec_per_kpc_proper(float(reduced_clusters_z[j])).value)) and ((reduced_clusters_z[j] - 0.01) < gz_z[i] < (reduced_clusters_z[j] + 0.01)):
            templist.append(gz_id[i])
    reduced_clusters_locals.append(templist)

print(len(reduced_clusters_id))
print(reduced_clusters_locals)
print(reduced_clusters_locals.shape)