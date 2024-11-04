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
    cluster_data = cluster[1].data
#print(cluster_table[1])

with fits.open('SpidersXclusterBCGs-v2.0.fits') as bcg:
    bcg_table = Table(bcg[1].data)
    bcg_data = bcg[1].data
#print(bcg_table)

with fits.open('GZDR1SFRMASS.fits') as galzoo:
    galzoo_table = Table(galzoo[1].data)
    galzoo_data = galzoo[1].data
#print(asn_table[1])

cluster_id = []
cluster_ra = []
cluster_dec = []
cluster_z = []
cluster_n = []

for i in range(len(cluster_data)):
    if cluster_data[i][12] < max_z and cluster_data[i][18] > min_n:
        cluster_id.append(cluster_data[i][0])
        cluster_ra.append(cluster_data[i][3])
        cluster_dec.append(cluster_data[i][4])
        cluster_z.append(cluster_data[i][12])
        cluster_n.append(cluster_data[i][18])

"""for i in range(len(bcg_data)):
    print(bcg_data[i][0])"""
#print(cluster_id)

reduced_clusters_id = []
reduced_clusters_z = []
reduced_clusters_ra = []
reduced_clusters_dec = []

for i in range(len(bcg_data)):
    for id1 in cluster_id:
        if id1 == bcg_data[i][0]:
            reduced_clusters_id.append(bcg_data[i][0])
            reduced_clusters_z.append(bcg_data[i][1])
            reduced_clusters_ra.append(bcg_data[i][2])
            reduced_clusters_dec.append(bcg_data[i][3])

reduced_clusters_locals = []

ID = []
RA = []
DEC = []
z = []

for i in range(len(galzoo_data)):
    ID.append(galzoo_data[i][0])
    RA.append(galzoo_data[i][13])
    DEC.append(galzoo_data[i][14])
    z.append(galzoo_data[i][18])

for i in range(len(ID)):
    print(z[i])
    for j in range(len(reduced_clusters_id)):
        templist = []
        if np.sqrt(((reduced_clusters_ra[j] - RA[i]) ** 2) * (np.cos(reduced_clusters_dec[j]) ** 2) + ((reduced_clusters_dec[j] - DEC[i]) ** 2)) < ((1000 / 3600) * cosmo.arcsec_per_kpc_proper(z[i])) and (z[i] - 0.01) < z[i] < (z[i] + 0.01):
            templist.append(ID[i])
    reduced_clusters_locals.append(templist)

print(len(reduced_clusters_id))
print(reduced_clusters_locals)
print(len(reduced_clusters_locals))



