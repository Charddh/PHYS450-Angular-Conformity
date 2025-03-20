import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, sine_function_2, sine_function_3, cosine_function, horizontal_line, chi_squared, chi2_red, assign_morph, calculate_theta

max_z = 0.125 #Maximum redshift in the sample.
min_lx = 1e43 #Minimum x-ray luminosity for clusters.
bin_size = 30 #Size in degrees of the bins.
axis_bin = 60 #Size in degrees of the axis bins.
sfr_bin_size = 30 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
classification_threshold = 0.5 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.5 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 1 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 2250 #Maximum physical separation in kpc between BCG and satellite galaxies.
min_phys_sep = 0 #Minimum physical separation in kpc between BCG and satellite galaxies.

show_mass_z = 0
show_sSFR_mass = 1

signal_to_noise = 1 #Minimum signal-to-noise ratio for galaxy spectra.
axis_bin = 60
mergers = ['1_9772', '1_1626', '2_3729', '1_5811', '1_1645', '1_9618', '1_4456', '2_19468']
binaries = ['2_13471', '1_12336', '1_9849', '1_21627', '2_1139', '1_23993', '2_3273', '2_11151', '1_14486', '1_4409', '1_14573', '1_14884', '1_5823', '1_14426']
#Open the cluster FITS file and retrieve the data from the second HDU (Header Data Unit).

cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data
#Convert the structured array to a dictionary with byte-swapping and endian conversion for each column.
cluster_df = pd.DataFrame({
    name: cluster_data[name].byteswap().newbyteorder()  #Apply byte-swapping and endian conversion to each field.
    for name in cluster_data.dtype.names  #Iterate over each field name in the structured array.
})

#Import the galaxy zoo and bcg datasets.
gz_df = pd.DataFrame(fits.open("GZDR1SFRMASS.fits")[1].data)

gz_id = gz_df['SPECOBJID_1'].values
gz_ra = gz_df['RA_1'].values
gz_dec = gz_df['DEC_1'].values
gz_z = gz_df['Z'].values
gz_mass = gz_df['LGM_TOT_P50'].values
gz_mass16 = gz_df['LGM_TOT_P16'].values
gz_mass84 = gz_df['LGM_TOT_P84'].values
gz_sfr = gz_df['SPECSFR_TOT_P50'].values
gz_sfr16 = gz_df['SPECSFR_TOT_P16'].values
gz_sfr84 = gz_df['SPECSFR_TOT_P84'].values
gz_s_n = gz_df['SN_MEDIAN'].values

if show_mass_z == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
    ax.scatter(gz_z, gz_mass, label="Galaxy", color="black", marker = 'o', s = 5, linewidth = 0.5, linestyle = 'None', alpha = 0.25)
    ax.set_ylabel(r'$\log ( M_*/M_{\odot})$', fontsize=16)
    ax.set_xlabel("Redshift (z)", fontsize=16)
    ax.axvline(x=0.125, color='grey', linestyle='--', linewidth=1.5, label="z = 0.125")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    #ax.set_ylim(-13, -10.5)
    ax.set_xlim(0, 0.32)
    ax.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()

if show_sSFR_mass == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
    ax.scatter(gz_mass, gz_sfr, label="Galaxy", color="black", marker = 'o', s = 5, linewidth = 0.5, linestyle = 'None', alpha = 0.15)
    ax.set_xlabel(r'$\log ( M_*/M_{\odot})$', fontsize=16)
    ax.set_ylabel(r"sSFR [yr$^{-1}$]", fontsize=16)
    ax.axvline(x=0.125, color='grey', linestyle='--', linewidth=1.5, label="z = 0.125")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(-14, -7)
    ax.set_xlim(7, 13)
    ax.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()    