import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.io import ascii #astropy is an astronomy module but here we are just importing its data read in functionality. If astropy not included in your installation you can see how to get it here https://www.astropy.org/
from astropy.table import Table, Column, MaskedColumn# this is astropy's way to make a data table, for reading out data
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import glob #allows you to read the names of files in folders 
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table
 
with fits.open('GZDR1SFRMASS.fits') as hdu:
    asn_table = Table(hdu[1].data)
print(asn_table[1])

with fits.open('GZDR1SFRMASS.fits') as hdu:
    science_data1 = hdu[1].data
    science_header1 = hdu[1].header
#print(science_data1[0])
#print(science_data1[0][13])
#print(science_data1[0][14])

ID = []
RA = []
DEC = []
z = []
RAc = 18.73997
DECc = 0.43081

for i in range(len(science_data1)):
    ID.append(science_data1[i][0])
    RA.append(science_data1[i][13])
    DEC.append(science_data1[i][14])
    z.append(science_data1[i][18])

local_galaxies = []

for i in range(len(ID)):
    if np.sqrt(((RAc - RA[i]) ** 2) * (np.cos(DECc) ** 2) + ((DECc - DEC[i]) ** 2)) < 0.31 and 0.0403 < z[i] < 0.0503:
            local_galaxies.append(ID[i])

print(len(local_galaxies))


