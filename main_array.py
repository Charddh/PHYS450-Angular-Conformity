import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.io import ascii #astropy is an astronomy module but here we are just importing its data read in functionality. If astropy not included in your installation you can see how to get it here https://www.astropy.org/
from astropy.table import Table, Column, MaskedColumn# this is astropy's way to make a data table, for reading out data
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import glob #allows you to read the names of files in folders 
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

max_z = 0.1 #maximum redshift of galaxies being looked for
min_n = 30 #minimum companions according to cluster catalogue

with fits.open('catCluster-SPIDERS_RASS_CLUS-v3.0.fits') as cluster:
    cluster_table = Table(cluster[1].data)
    cluster_data = np.array(cluster[1].data)

with fits.open('SpidersXclusterBCGs-v2.0.fits') as bcg:
    bcg_table = Table(bcg[1].data)
    bcg_data = np.array(bcg[1].data)

with fits.open('GZDR1SFRMASS.fits') as galzoo:
    galzoo_table = Table(galzoo[1].data)
    galzoo_data = np.array(galzoo[1].data)

mask = (cluster_data[:, 12] < max_z) & (cluster_data[:, 18] > min_n)
cluster_id = cluster_data[mask, 0]
cluster_ra = cluster_data[mask, 3]
cluster_dec = cluster_data[mask, 4]
cluster_z = cluster_data[mask, 12]
cluster_n = cluster_data[mask, 18]


cluster_id1 = []
cluster_ra1 = []
cluster_dec1 = []
cluster_z1 = []
cluster_n1 = []

for i in range(len(cluster_data)):
    if cluster_data[i][12] < max_z and cluster_data[i][18] > min_n:
        cluster_id1.append(cluster_data[i][0])
        cluster_ra1.append(cluster_data[i][3])
        cluster_dec1.append(cluster_data[i][4])
        cluster_z1.append(cluster_data[i][12])
        cluster_n1.append(cluster_data[i][18])

print(cluster_id1)
print(cluster_id)