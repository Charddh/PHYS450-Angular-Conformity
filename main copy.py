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

max_z = 0.12
min_n = 25
bin_size = 20 #Size in degrees of the bins
min_satellite_mass = 11

# Open the FITS file and retrieve the data from the second HDU (Header Data Unit)
cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data
# Convert the structured array to a dictionary with byte-swapping and endian conversion for each column
cluster_df = pd.DataFrame({
    name: cluster_data[name].byteswap().newbyteorder()  # Apply byte-swapping and endian conversion to each field
    for name in cluster_data.dtype.names  # Iterate over each field name in the structured array
})

gz_df = pd.DataFrame(fits.open("GZDR1SFRMASS.fits")[1].data)
bcg_df = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

cluster_df2 = cluster_df[(cluster_df['SCREEN_CLUZSPEC'] < max_z) & (cluster_df['SCREEN_NMEMBERS_W'] > min_n)]

# Extract relevant columns from the filtered clusters DataFrame
cluster_id = cluster_df2['CLUS_ID'].values
cluster_ra = cluster_df2['RA'].values
cluster_dec = cluster_df2['DEC'].values
cluster_z = cluster_df2['SCREEN_CLUZSPEC'].values

# Extract relevant columns from the BCG DataFrame (match by CLUS_ID)
bcg_df2 = bcg_df[bcg_df['CLUS_ID'].isin(cluster_id)]

#print(list(bcg_df2.columns))
#print(list(bcg_df2.columns[bcg_df.columns.str.contains('pa', case=False)]))
reduced_clusters_id = bcg_df2['CLUS_ID'].values
reduced_clusters_ra = bcg_df2['RA_BCG'].values
reduced_clusters_dec = bcg_df2['DEC_BCG'].values
reduced_clusters_z = bcg_df2['CLUZSPEC'].values
reduced_clusters_pa = bcg_df2['GAL_sdss_i_modSX_C2_PA'].values

# Extract relevant columns from the Galaxy Zoo DataFrame
gz_id = gz_df['SPECOBJID_1'].values
gz_ra = gz_df['RA_1'].values
gz_dec = gz_df['DEC_1'].values
gz_z = gz_df['Z'].values
gz_elliptical = gz_df['P_EL'].values
gz_spiral = gz_df['P_CS'].values
gz_mass = gz_df['LGM_TOT_P50'].values

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
reduced_clusters_locals_id = [gz_id[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_ra = [gz_ra[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_dec = [gz_dec[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_z = [gz_z[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_elliptical = [gz_elliptical[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_spiral = [gz_spiral[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_mass = [gz_mass[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]

gz_id_list = [item for sublist in reduced_clusters_locals_id for item in sublist]
gz_ra_list = [item for sublist in reduced_clusters_locals_ra for item in sublist]
gz_dec_list = [item for sublist in reduced_clusters_locals_dec for item in sublist]
gz_z_list = [item for sublist in reduced_clusters_locals_z for item in sublist]

df_gz = pd.DataFrame({'gz_id': gz_id_list,'gz_ra': gz_ra_list,'gz_dec': gz_dec_list,'gz_z': gz_z_list})
df_bcg = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec,'bcg_z': reduced_clusters_z, 'bcg_sdss_pa': reduced_clusters_pa, 'sat_id': reduced_clusters_locals_id, 'sat_ra': reduced_clusters_locals_ra, 'sat_dec': reduced_clusters_locals_dec, 'sat_elliptical': reduced_clusters_locals_elliptical, 'sat_spiral': reduced_clusters_locals_spiral, 'sat_mass': reduced_clusters_locals_mass})

#df_gz.to_csv('gz.csv', index=False)
#df_bcg.to_csv('bcg.csv', index=False)

angle_df = pd.read_csv('BCGAngleOffset.csv')
angle_df["clus_id"] = angle_df["clus_id"].str.strip()
df_bcg["bcg_id"] = df_bcg["bcg_id"].str.strip()

merged_df = pd.merge(angle_df, df_bcg, left_on='clus_id', right_on='bcg_id', how = 'inner').drop(columns=['bcg_id'])
merged_df['corrected_pa'] = ((((90 - merged_df['spa']) % 360) - merged_df['bcg_sdss_pa']) % 360)

print(f"Merged df columns: {list(merged_df.columns)}")

def calculate_theta(bcg_ra, bcg_dec, gal_ra, gal_dec):
    """
    Compute the angle (theta) in radians between a BCG and satellite galaxy.
    """
    if (isinstance(gal_ra, str) and gal_ra.strip() == "[]") or (isinstance(gal_dec, str) and gal_dec.strip() == "[]"):
        return []
    gal_ra = np.array(gal_ra, dtype=float)
    gal_dec = np.array(gal_dec, dtype=float)
    avg_dec = np.radians((bcg_dec + gal_dec)/2)
    delta_ra = np.radians(bcg_ra - gal_ra)*np.cos(avg_dec)
    delta_dec = np.radians(bcg_dec - gal_dec)
    theta_raw = np.arctan2(delta_ra, delta_dec)
    theta_clockwise = (-theta_raw) % (2 * np.pi)
    return np.degrees(theta_clockwise)

merged_df['theta'] = merged_df.apply(lambda row: calculate_theta(row['bcg_ra'], row['bcg_dec'], row['sat_ra'], row['sat_dec']), axis=1)

merged_df['sat_majoraxis_angle'] = merged_df.apply(lambda row: [(theta - row['corrected_pa']) % 360 for theta in row['theta']], axis=1)

satellite_df = pd.DataFrame({
    "clus_id": merged_df["clus_id"],
    "bcg_ra": merged_df["bcg_ra"],
    "bcg_dec": merged_df["bcg_dec"],
    "sat_ra": merged_df["sat_ra"],
    "sat_dec": merged_df["sat_dec"],
    "theta": merged_df["theta"]})
satellite_df.to_csv('satellite.csv', index=False)

merged_df['sat_elliptical'] = merged_df['sat_elliptical'].apply(lambda x: [1 if value >= 0.5 else 0 for value in x])
merged_df['sat_spiral'] = merged_df['sat_spiral'].apply(lambda x: [1 if value >= 0.5 else 0 for value in x])

sat_majoraxis_list = np.concatenate(merged_df['sat_majoraxis_angle'].values)
sat_elliptical_list = np.concatenate(merged_df['sat_elliptical'].values)
sat_spiral_list = np.concatenate(merged_df['sat_spiral'].values)
sat_mass_list = np.concatenate(merged_df['sat_mass'].values)

print(f"Sat major axis list: {len(sat_majoraxis_list)}")
print(f"Sat elliptical list: {len(sat_elliptical_list)}")
print(f"Sat spiral list: {len(sat_spiral_list)}")
print(f"Sat mass list: {len(sat_mass_list)}")

mask = sat_mass_list >= min_satellite_mass

filtered_majoraxis_list = sat_majoraxis_list[mask]
filtered_elliptical_list = sat_elliptical_list[mask]
filtered_spiral_list = sat_spiral_list[mask]
filtered_mass_list = sat_mass_list[mask]

print(f"Sat major axis list: {len(filtered_majoraxis_list)}")
print(f"Sat elliptical list: {len(filtered_elliptical_list)}")
print(f"Sat spiral list: {len(filtered_spiral_list)}")
print(f"Sat mass list: {len(filtered_mass_list)}")
print(f"min_satellite_mass: {min(filtered_mass_list)}")

sat_type_list = [
    "e" if elliptical == 1 and spiral == 0 else
    "s" if elliptical == 0 and spiral == 1 else
    "u"
    for elliptical, spiral in zip(sat_elliptical_list, sat_spiral_list)]

data = {"angles": sat_majoraxis_list, "types": sat_type_list}

df=pd.DataFrame(data)

df["spiral_angles"] = df["angles"].where(df["types"] == "s")
df["elliptical_angles"] = df["angles"].where(df["types"] == "e")
df["unknown_angles"] = df["angles"].where(df["types"] == "u")

spirals = df["spiral_angles"].dropna().reset_index(drop=True)
ellipticals = df["elliptical_angles"].dropna().reset_index(drop=True)
unknowns = df["unknown_angles"].dropna().reset_index(drop=True)

df_angles = pd.concat([spirals, ellipticals, unknowns], axis=1, keys=["spirals", "ellipticals", "unknowns"])
df.to_csv('angles.csv', index=False)

bins = np.arange(0, 360, bin_size)

spiral_hist, _ = np.histogram(spirals, bins=bins)
elliptical_hist, _ = np.histogram(ellipticals, bins=bins)
unknown_hist, _ = np.histogram(unknowns, bins=bins)

bin_centres = (bins[:-1] + bins[1:]) / 2 #Bin midpoints

plt.figure(figsize=(10, 6))

plt.plot(bin_centres, spiral_hist, label="Spirals", color="blue")
plt.plot(bin_centres, elliptical_hist, label="Ellipticals", color="red")
plt.plot(bin_centres, unknown_hist, label="Unknowns", color="green")
plt.xlabel("Angle (degrees)")
plt.ylabel("Number of Galaxies")
plt.title(f"Galaxy Angle Distribution")
plt.xlim([0, 360])
plt.text(0.7, 0.9, f"{bin_size}° Bins\n{min_n} Minimum members\nz < {max_z}", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 12, fontweight = "normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
plt.xticks(bins)  #Show bin edges on x-axis
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()