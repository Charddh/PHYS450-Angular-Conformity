import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, calculate_r200, sine_function_2, sine_function_3, calculate_radial_distance, calculate_velocity_distance, cosine_function, horizontal_line, chi_squared, chi2_red, assign_morph, calculate_theta

classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
debiased = 0 #If 1, will use debiased classifications. Else, will use raw classifications.
bin_size = 60 #Size in degrees of the bins.
phys_sep = 2750 #Maximum physical separation in kpc between BCG and satellite galaxies.
min_phys_sep = 0 #Minimum physical separation in kpc between BCG and satellite galaxies.
min_lx = 3.8e43 #Minimum x-ray luminosity for clusters.
sfr_threshold = -11 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
max_z = 0.115 #Maximum redshift in the sample.
min_vel_diff = 0

use_r200 = 1
max_r200 = 3
min_r200 = 1.6

axis_bin = 60 #Size in degrees of the axis bins.
sfr_bin_size = 60 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
#max_satellite_mass = 11.5 #Maximum satellite galaxy mass.
max_satellite_mass = 11.5 #Maximum satellite galaxy mass.
mergers = ['1_9618', '1_1626', '1_5811', '1_1645']

show_qs = 0
show_fs = 0
show_q_binom = 0
show_sq_binom = 0 #If 1, will show the quiescent spiral fraction plot.
show_eq_binom = 0 #If 1, will show the star forming elliptical fraction plot.
show_phase_heat = 1
show_phase_combined = 0

phase_bin_size = 0.6

restricted = 1

show_elliptical = 0 #If 1, will show the elliptical fraction plot.
show_quiescent = 0 #If 1, will show the quiescent fraction plot.

signal_to_noise = 1 #Minimum signal-to-noise ratio for galaxy spectra.
axis_bin = 60

show_eq_phase = 0
show_sq_phase = 0
show_q_phase = 0
show_qs_phase = 0
show_fs_phase = 0
show_phase_space = 0
show_phase_space_r200 = 0
show_ef = 0 #If 1, will show the star forming elliptical fraction plot.
show_ef_binom = 0 #If 1, will show the star forming elliptical fraction plot.
show_eq = 0 #If 1, will show the star forming elliptical fraction plot.
show_sq = 0 #If 1, will show the quiescent spiral fraction plot.
show_sf = 0 #If 1, will show the quiescent spiral fraction plot.
show_sf_binom = 0 #If 1, will show the quiescent spiral fraction plot.
show_q = 0 #If 1, will show the quiescent fraction plot.
show_f = 0 #If 1, will show the star forming fraction plot.
show_f_binom = 0
show_e = 0 #If 1, will show the elliptical fraction plot.
show_e_binom = 0 #If 1, will show the elliptical fraction plot.
show_s = 0 #If 1, will show the spiral fraction plot.
show_s_binom = 0 #If 1, will show the spiral fraction plot.
show_ssfr = 0

#Open the cluster FITS file and retrieve the data from the second HDU (Header Data Unit).

cluster_data1 = pd.read_csv("VII110Atable3.csv")
df_clus1 = pd.DataFrame(cluster_data1)
cluster_data2 = pd.read_csv("VII110Atable4.csv")
df_clus2 = pd.DataFrame(cluster_data2)
cluster_data3 = pd.read_csv("VII110Atable5.csv")
df_clus3 = pd.DataFrame(cluster_data3)
cluster_data4 = pd.read_csv("VII110Atable6.csv")
df_clus4 = pd.DataFrame(cluster_data4)
cluster_df = pd.concat(
    [cluster_data1, cluster_data2, cluster_data3, cluster_data4],
    ignore_index=True
)

cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data
#Convert the structured array to a dictionary with byte-swapping and endian conversion for each column.
cluster_df = pd.DataFrame({
    name: cluster_data[name].byteswap().view(cluster_data[name].dtype.newbyteorder())
    for name in cluster_data.dtype.names
})

#print("cluster_df:", cluster_df.columns)

gal_data = pd.read_csv("lsst_table.csv")
galaxy_df = pd.DataFrame(gal_data)

galaxy_df = pd.DataFrame(fits.open("GZDR1SFRMASS.fits")[1].data)

print(galaxy_df.columns)

bcg_df = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

"""cluster_id = cluster_df['ACO'].values
cluster_ra = cluster_df['_RA_icrs'].values
cluster_dec = cluster_df['_DE_icrs'].values
cluster_z = cluster_df['z'].values"""

cluster_id = cluster_df['CLUS_ID'].values
cluster_ra = cluster_df['RA'].values
cluster_dec = cluster_df['DEC'].values
cluster_z = cluster_df['SCREEN_CLUZSPEC'].values

#Extract relevant columns from the galaxy zoo dataframe.
"""gal_id = galaxy_df['objectId'].values
gal_ra = galaxy_df['coord_ra'].values
gal_dec= galaxy_df['coord_dec'].values
gal_z = galaxy_df['knn_z_median'].values"""

gal_id = galaxy_df['SPECOBJID_1'].values
gal_ra = galaxy_df['RA_1'].values
gal_dec= galaxy_df['DEC_1'].values
gal_z = galaxy_df['Z'].values

"""#Calculate the angular separation and redshift difference.
ra_diff = cluster_ra[:, None] - gal_ra  #Broadcast the RA difference calculation.
dec_diff = cluster_dec[:, None] - gal_dec  #Broadcast the Dec difference calculation.
z_diff = np.abs(cluster_z[:, None] - gal_z)  #Compute absolute redshift difference.

#Compute the angular separation using the Haversine formula and the proper scaling.
angular_separation = np.sqrt((ra_diff ** 2) * (np.cos(np.radians(cluster_dec[:, None])) ** 2) + dec_diff ** 2)

#Number of degrees corresponding to 1 kpc at the redshift of each cluster.
degrees_per_kpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(cluster_z[:, None]).value
phys_sep_galaxy = angular_separation / degrees_per_kpc
#r200_sep_galaxy = phys_sep_galaxy / reduced_clusters_r200[:, None]


max_vel = np.where(
    (phys_sep_galaxy >= 0) & (phys_sep_galaxy <= 3000),
    -0.43 * phys_sep_galaxy + 2000,
    np.where(
        (phys_sep_galaxy > 3000),
        np.inf,
        np.nan))


selected_galaxies_mask = (
    (phys_sep_galaxy <= phys_sep) & 
    (phys_sep_galaxy > min_phys_sep) &
    (z_diff < max_vel / 3e5))

selected_counts = [np.sum(selected_galaxies_mask[i]) for i in range(len(cluster_ra))]
print("Sel", sum(selected_counts))

cluster_locals_id = [gal_id[selected_galaxies_mask[i]] for i in range(len(cluster_ra))]
print("1")
cluster_locals_ra = [gal_ra[selected_galaxies_mask[i]] for i in range(len(cluster_ra))]
print("2")
cluster_locals_dec = [gal_dec[selected_galaxies_mask[i]] for i in range(len(cluster_ra))]
print("3")
cluster_locals_z = [gal_z[selected_galaxies_mask[i]] for i in range(len(cluster_ra))]
"""

chunk_size = 100  # Adjust as needed
n_clusters = len(cluster_ra)
n_chunks = int(np.ceil(n_clusters / chunk_size))

selected_counts = []
cluster_locals_id = []
cluster_locals_ra = []
cluster_locals_dec = []
cluster_locals_z = []

for i, start in enumerate(range(0, n_clusters, chunk_size)):
    end = min(start + chunk_size, n_clusters)
    
    print(f"Processing chunk {i+1}/{n_chunks} ({end}/{n_clusters} clusters)...", flush=True)
    
    # Slice chunk of clusters
    ra_chunk = cluster_ra[start:end]
    dec_chunk = cluster_dec[start:end]
    z_chunk = cluster_z[start:end]
    
    # Compute angular separation for this chunk
    ra_diff = gal_ra[None, :] - ra_chunk[:, None]
    dec_diff = gal_dec[None, :] - dec_chunk[:, None]
    z_diff = np.abs(gal_z[None, :] - cluster_z[:, None])
    ang_sep = np.sqrt((ra_diff**2) * np.cos(np.radians(dec_chunk[:, None]))**2 + dec_diff**2)
    
    degrees_per_kpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(z_chunk[:, None]).value
    phys_sep_galaxy = ang_sep / degrees_per_kpc
    
    max_vel = np.where(
        (phys_sep_galaxy >= 0) & (phys_sep_galaxy <= 3000),
        -0.43 * phys_sep_galaxy + 2000,
        np.where(phys_sep_galaxy > 3000, np.inf, np.nan)
    )
    
    # z_diff might need slicing too if it's 2D
    selected_mask = (
        (phys_sep_galaxy <= phys_sep) & 
        (phys_sep_galaxy > min_phys_sep) &
        (z_diff[start:end, :] < max_vel / 3e5)
    )

    selected_counts.extend(np.sum(selected_mask, axis=1))
    cluster_locals_id.extend([gal_id[m] for m in selected_mask])
    cluster_locals_ra.extend([gal_ra[m] for m in selected_mask])
    cluster_locals_dec.extend([gal_dec[m] for m in selected_mask])
    cluster_locals_z.extend([gal_z[m] for m in selected_mask])
    
    print(f"✅ Finished chunk {i+1}/{n_chunks}")

print("✅ All chunks complete!")
print("Total selected galaxies:", sum(selected_counts))

df_clusters = pd.DataFrame({'id': cluster_locals_id, 'ra': cluster_locals_ra, 'dec': cluster_locals_dec, 'z': cluster_locals_z})

print(df_clusters.head())

#Create a list of Galaxy data for each cluster.
reduced_clusters_locals_id = [gz_id[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_ra = [gz_ra[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_dec = [gz_dec[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_z = [gz_z[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_elliptical = [gz_elliptical[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_spiral = [gz_spiral[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_mass = [gz_mass[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_sfr = [gz_sfr[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_sfr16 = [gz_sfr16[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
reduced_clusters_locals_sfr84 = [gz_sfr84[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]

#Create a dataframe containing the bcg data with corresponding satellite data.
df_bcg = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec,'bcg_z': reduced_clusters_z, 'bcg_sdss_pa': reduced_clusters_pa, 'bcg_r200': reduced_clusters_r200, 'sat_id': reduced_clusters_locals_id, 'sat_ra': reduced_clusters_locals_ra, 'sat_dec': reduced_clusters_locals_dec, 'sat_z': reduced_clusters_locals_z, 'sat_elliptical': reduced_clusters_locals_elliptical, 'sat_spiral': reduced_clusters_locals_spiral, 'sat_mass': reduced_clusters_locals_mass, 'sat_sfr': reduced_clusters_locals_sfr, 'sat_sfr16': reduced_clusters_locals_sfr16, 'sat_sfr84': reduced_clusters_locals_sfr84})
clusters_df = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec, 'bcg_sdss_pa': reduced_clusters_pa})
clusters_df.to_csv('bcg_data.csv', index=False)
df_bcg["bcg_id"] = df_bcg["bcg_id"].str.strip()
angle_df = pd.read_csv('BCGAngleOffset.csv')
angle_df["clus_id"] = angle_df["clus_id"].str.strip()

missing_ids = df_bcg["bcg_id"][~df_bcg["bcg_id"].isin(angle_df["clus_id"])]

merged_df = pd.merge(angle_df, df_bcg, left_on='clus_id', right_on='bcg_id', how = 'inner').drop(columns=['bcg_id'])
merged_df['corrected_pa'] = ((((90 - merged_df['spa']) % 360) - merged_df['bcg_sdss_pa']) % 360)

merged_df['theta'] = merged_df.apply(lambda row: calculate_theta(row['bcg_ra'], row['bcg_dec'], row['sat_ra'], row['sat_dec']), axis=1)
merged_df['sat_majoraxis_angle'] = merged_df.apply(lambda row: [((row['corrected_pa'] - theta) % 360) % 180 for theta in row['theta']], axis=1)
merged_df['radial_sep'] = merged_df.apply(lambda row: calculate_radial_distance(row['bcg_ra'], row['bcg_dec'], row['bcg_z'], row['sat_ra'], row['sat_dec']), axis=1)
merged_df['vel_diff'] = merged_df.apply(lambda row: calculate_velocity_distance(row['bcg_z'], row['sat_z']), axis=1)
merged_df['r/r200'] = merged_df['radial_sep'] / merged_df['bcg_r200']

if classification_threshold == 1:
    merged_df[['sat_elliptical', 'sat_spiral']] = merged_df.apply(
    lambda row: assign_morph(row['sat_elliptical'], row['sat_spiral']),
    axis=1,
    result_type='expand')
else:
    merged_df['sat_elliptical'] = merged_df['sat_elliptical'].apply(lambda x: [1 if value >= classification_threshold else 0 for value in x])
    merged_df['sat_spiral'] = merged_df['sat_spiral'].apply(lambda x: [1 if value >= classification_threshold else 0 for value in x])

merged_df['star_forming'] = merged_df['sat_sfr'].apply(lambda x: [1 if value >= sfr_threshold else 0 for value in x])
merged_df['quiescent'] = merged_df['sat_sfr'].apply(lambda x: [1 if value < sfr_threshold else 0 for value in x])

sat_majoraxis_list = np.concatenate(merged_df['sat_majoraxis_angle'].values)
sat_elliptical_list = np.concatenate(merged_df['sat_elliptical'].values)
sat_spiral_list = np.concatenate(merged_df['sat_spiral'].values)
sat_mass_list = np.concatenate(merged_df['sat_mass'].values)
star_forming_list = np.concatenate(merged_df['star_forming'].values)
quiescent_list = np.concatenate(merged_df['quiescent'].values)
vel_diff_list = np.concatenate(merged_df['vel_diff'].values)
radial_sep_list = np.concatenate(merged_df['radial_sep'].values)
sfr_list = np.concatenate(merged_df['sat_sfr'].values)
sfr16_list = np.concatenate(merged_df['sat_sfr16'].values)
sfr84_list = np.concatenate(merged_df['sat_sfr84'].values)
sfr_error = (sfr84_list - sfr16_list) / 2
r200_list = np.concatenate(merged_df['r/r200'].values)


print("max r200:", max(r200_list))


print("number of satellites in sample", len(sat_majoraxis_list))

axis_sat_majoraxis_list = sat_majoraxis_list.copy()
axis_sat_majoraxis_list = np.where(axis_sat_majoraxis_list > 135, axis_sat_majoraxis_list - 180, axis_sat_majoraxis_list)

sat_type_list = [
    "e" if elliptical == 1 and spiral == 0 else
    "s" if elliptical == 0 and spiral == 1 else
    "u"
    for elliptical, spiral in zip(sat_elliptical_list, sat_spiral_list)]

sfr_type_list = [
    "q" if quiescent == 1 and forming == 0 else
    "f" if quiescent == 0 and forming == 1 else
    "u"
    for forming, quiescent in zip(star_forming_list, quiescent_list)]

morph_forming_list = [
    "ef" if morph == "e" and sfr == "f" else
    "sf" if morph == "s" and sfr == "f" else
    "eq" if morph == "e" and sfr == "q" else
    "sq" if morph == "s" and sfr == "q" else
    "uq" if morph == "u" and sfr == "q" else
    "uf" if morph == "u" and sfr == "f" else
    "uu"
    for morph, sfr in zip(sat_type_list, sfr_type_list)]

vel_diff   = np.array(vel_diff_list)
r200_sep = np.array(r200_list)
sat_type   = np.array(sat_type_list)
sfr_type = np.array(sfr_type_list)
morph_forming = np.array(morph_forming_list)

data = {"angles": sat_majoraxis_list, "types": sat_type_list, "morph_sfr": morph_forming_list, "sep": r200_list}
df=pd.DataFrame(data)
df["spiral_angles"] = df["angles"].where(df["types"] == "s")
df["elliptical_angles"] = df["angles"].where(df["types"] == "e")
df["unknown_angles"] = df["angles"].where(df["types"] == "u")
spirals = df["spiral_angles"].dropna().reset_index(drop=True)
ellipticals = df["elliptical_angles"].dropna().reset_index(drop=True)
unknowns = df["unknown_angles"].dropna().reset_index(drop=True)

df["ef_angles"] = df["angles"].where(df["morph_sfr"] == "ef")
df["sf_angles"] = df["angles"].where(df["morph_sfr"] == "sf")
df["eq_angles"] = df["angles"].where(df["morph_sfr"] == "eq")
df["sq_angles"] = df["angles"].where(df["morph_sfr"] == "sq")
df["uq_angles"] = df["angles"].where(df["morph_sfr"] == "uq")
df["uf_angles"] = df["angles"].where(df["morph_sfr"] == "uf")
df["uu_angles"] = df["angles"].where(df["morph_sfr"] == "uu")
ef = df["ef_angles"].dropna().reset_index(drop=True)
sf = df["sf_angles"].dropna().reset_index(drop=True)
eq = df["eq_angles"].dropna().reset_index(drop=True)
sq = df["sq_angles"].dropna().reset_index(drop=True)
uq = df["uq_angles"].dropna().reset_index(drop=True)
uf = df["uf_angles"].dropna().reset_index(drop=True)
uu = df["uu_angles"].dropna().reset_index(drop=True)
df["ef_sep"] = df["sep"].where(df["morph_sfr"] == "ef")
df["sf_sep"] = df["sep"].where(df["morph_sfr"] == "sf")
df["eq_sep"] = df["sep"].where(df["morph_sfr"] == "eq")
df["sq_sep"] = df["sep"].where(df["morph_sfr"] == "sq")
df["uu_sep"] = df["sep"].where(df["morph_sfr"] == "uu")
df["q_sep"] = df["sep"].where(df["morph_sfr"].isin(["eq", "sq", "uq"]))
df["f_sep"] = df["sep"].where(df["morph_sfr"].isin(["ef", "sf", "uf"]))
phase_ef = df["ef_sep"].dropna().reset_index(drop=True)
phase_sf = df["sf_sep"].dropna().reset_index(drop=True)
phase_eq = df["eq_sep"].dropna().reset_index(drop=True)
phase_sq = df["sq_sep"].dropna().reset_index(drop=True)
phase_uu = df["uu_sep"].dropna().reset_index(drop=True)
phase_q = df["q_sep"].dropna().reset_index(drop=True)
phase_f = df["f_sep"].dropna().reset_index(drop=True)

axis_data = {"angles": axis_sat_majoraxis_list, "types": sat_type_list, "morph_sfr": morph_forming_list}
axis_df=pd.DataFrame(axis_data)
axis_df["spiral_angles"] = axis_df["angles"].where(axis_df["types"] == "s")
axis_df["elliptical_angles"] = axis_df["angles"].where(axis_df["types"] == "e")
axis_df["unknown_angles"] = axis_df["angles"].where(axis_df["types"] == "u")
axis_spirals = axis_df["spiral_angles"].dropna().reset_index(drop=True)
axis_ellipticals = axis_df["elliptical_angles"].dropna().reset_index(drop=True)
axis_unknowns = axis_df["unknown_angles"].dropna().reset_index(drop=True)

axis_df["ef_angles"] = axis_df["angles"].where(axis_df["morph_sfr"] == "ef")
axis_df["sf_angles"] = axis_df["angles"].where(axis_df["morph_sfr"] == "sf")
axis_df["eq_angles"] = axis_df["angles"].where(axis_df["morph_sfr"] == "eq")
axis_df["sq_angles"] = axis_df["angles"].where(axis_df["morph_sfr"] == "sq")
axis_df["uu_angles"] = axis_df["angles"].where(axis_df["morph_sfr"] == "uu")
axis_ef = axis_df["ef_angles"].dropna().reset_index(drop=True)
axis_sf = axis_df["sf_angles"].dropna().reset_index(drop=True)
axis_eq = axis_df["eq_angles"].dropna().reset_index(drop=True)
axis_sq = axis_df["sq_angles"].dropna().reset_index(drop=True)
axis_uu = axis_df["uu_angles"].dropna().reset_index(drop=True)

sfr_data = {"angles": sat_majoraxis_list, "types": sfr_type_list}
sfr_df=pd.DataFrame(sfr_data)
sfr_df["forming_angles"] = sfr_df["angles"].where(sfr_df["types"] == "f")
sfr_df["quiescent_angles"] = sfr_df["angles"].where(sfr_df["types"] == "q")
sfr_df["unknown_angles"] = sfr_df["angles"].where(sfr_df["types"] == "u")
sfr_forming = sfr_df["forming_angles"].dropna().reset_index(drop=True)
sfr_quiescent = sfr_df["quiescent_angles"].dropna().reset_index(drop=True)
sfr_unknowns = sfr_df["unknown_angles"].dropna().reset_index(drop=True)

axis_sfr_data = {"angles": axis_sat_majoraxis_list, "types": sfr_type_list}
axis_sfr_df=pd.DataFrame(sfr_data)
axis_sfr_df["forming_angles"] = axis_sfr_df["angles"].where(axis_sfr_df["types"] == "f")
axis_sfr_df["quiescent_angles"] = axis_sfr_df["angles"].where(axis_sfr_df["types"] == "q")
axis_sfr_df["unknown_angles"] = axis_sfr_df["angles"].where(axis_sfr_df["types"] == "u")
axis_sfr_forming = axis_sfr_df["forming_angles"].dropna().reset_index(drop=True)
axis_sfr_quiescent = axis_sfr_df["quiescent_angles"].dropna().reset_index(drop=True)
axis_sfr_unknowns = axis_sfr_df["unknown_angles"].dropna().reset_index(drop=True)

df_angles = pd.concat([spirals, ellipticals, unknowns], axis=1, keys=["spirals", "ellipticals", "unknowns"])
df.to_csv('angles.csv', index=False)

bins = np.arange(0, 181, bin_size)
bin_centres = (bins[:-1] + bins[1:]) / 2
axis_trialX = np.linspace(-45, 135, 1000)
trialX = np.linspace(0, 180, 1000)
if axis_bin == 60:
    axis_bins = np.array([-30, 30, 60, 120])
    axis_bin_centres = (axis_bins[:-1] + axis_bins[1:]) / 2
if axis_bin == 90:
    axis_bins = np.array([-45, 45, 135])
    axis_bin_centres = np.array([0, 90])
else:
    print("axis_bin must be either 60 or 90.")

sfr_bins = np.arange(0, 181, sfr_bin_size)

ef_hist, _ = np.histogram(ef, bins=bins)
sf_hist, _ = np.histogram(sf, bins=bins)
eq_hist, _ = np.histogram(eq, bins=bins)
sq_hist, _ = np.histogram(sq, bins=bins)
uq_hist, _ = np.histogram(uq, bins=bins)
uf_hist, _ = np.histogram(uf, bins=bins)
uu_hist, _ = np.histogram(uu, bins=bins)
ef_err = np.sqrt(ef_hist)
sf_err = np.sqrt(sf_hist)
eq_err = np.sqrt(eq_hist)
sq_err = np.sqrt(sq_hist)
uq_err = np.sqrt(uq_hist)
uf_err = np.sqrt(uf_hist)
uu_err = np.sqrt(uu_hist)
ef_fraction = np.where(ef_hist + eq_hist > 0, (ef_hist / (ef_hist + eq_hist)), 0)
eq_fraction = np.where(ef_hist + eq_hist > 0, (eq_hist / (ef_hist + eq_hist)), 0)
sq_fraction = np.where(sq_hist + sf_hist > 0, (sq_hist / (sq_hist + sf_hist)), 0)
sf_fraction = np.where(sq_hist + sf_hist > 0, (sf_hist / (sq_hist + sf_hist)), 0)
total_gals = ef_hist + eq_hist + sq_hist + sf_hist + uf_hist + uq_hist + uu_hist
total_ellipticals = ef_hist + eq_hist
total_spirals = sf_hist + sq_hist
total_morph = ef_hist + eq_hist + sq_hist + sf_hist
total_forming = ef_hist + sf_hist
total_quiescent = eq_hist + sq_hist
q_fraction = np.where(total_gals > 0, ((eq_hist + sq_hist + uq_hist) / (total_gals)), 0)
f_fraction = np.where(total_gals > 0, ((ef_hist + sf_hist + uf_hist) / (total_gals)), 0)
e_fraction = np.where(total_morph > 0, ((eq_hist + ef_hist) / (total_morph)), 0)
s_fraction = np.where(total_morph > 0, ((sq_hist + sf_hist) / (total_morph)), 0)
fs_fraction = np.where(total_forming > 0, (sf_hist / (total_forming)), 0)
qs_fraction = np.where(total_quiescent > 0, (sq_hist / (total_quiescent)), 0)
ef_fraction_err = np.where(ef_hist + eq_hist > 0, np.sqrt(ef_hist) / (ef_hist + eq_hist), np.nan)
ef_fraction_err_binom = np.where(total_ellipticals > 0, np.sqrt((ef_hist / total_ellipticals) * (1 - (ef_hist / total_ellipticals)) / total_ellipticals), np.nan)
eq_fraction_err = np.where(ef_hist + eq_hist > 0, np.sqrt(eq_hist) / (ef_hist + eq_hist), np.nan)
eq_fraction_err_binom = np.where(total_ellipticals > 0, np.sqrt((eq_hist / total_ellipticals) * (1 - (eq_hist / total_ellipticals)) / total_ellipticals), np.nan)
sq_fraction_err = np.where(sq_hist + sf_hist > 0, np.sqrt(sq_hist) / (sq_hist + sf_hist), np.nan)
sq_fraction_err_binom = np.where(total_spirals > 0, np.sqrt((sq_hist / total_spirals) * (1 - (sq_hist / total_spirals)) / total_spirals), np.nan)
sf_fraction_err = np.where(sq_hist + sf_hist > 0, np.sqrt(sf_hist) / (sq_hist + sf_hist), np.nan)
sf_fraction_err_binom = np.where(total_spirals > 0, np.sqrt((sf_hist / total_spirals) * (1 - (sf_hist / total_spirals)) / total_spirals), np.nan)
q_fraction_err = np.where(total_gals > 0, np.sqrt(eq_hist + sq_hist + uq_hist) / (total_gals), np.nan)
q_fraction_err_binom = np.where(total_gals > 0, np.sqrt(q_fraction * (1 - q_fraction) / total_gals), np.nan) 
f_fraction_err = np.where(total_gals > 0, np.sqrt(ef_hist + sf_hist + uf_hist) / (total_gals), np.nan)
f_fraction_err_binom = np.where(total_gals > 0, np.sqrt(f_fraction * (1 - f_fraction) / total_gals), np.nan) 
e_fraction_err = np.where(total_morph > 0, np.sqrt(eq_hist + ef_hist) / (total_morph), np.nan)
e_fraction_err_binom = np.where(total_morph > 0, np.sqrt(e_fraction * (1 - e_fraction) / total_morph), np.nan)
s_fraction_err = np.where(total_morph > 0, np.sqrt(sq_hist + sf_hist) / (total_morph), np.nan)
s_fraction_err_binom = np.where(total_morph > 0, np.sqrt(s_fraction * (1 - s_fraction) / total_morph), np.nan)
fs_fraction_err = np.where(total_forming > 0, np.sqrt(fs_fraction * (1 - fs_fraction) / total_forming), np.nan)
qs_fraction_err = np.where(total_quiescent > 0, np.sqrt(qs_fraction * (1 - qs_fraction) / total_quiescent), np.nan)
popt_ef_frac, pcov_ef_frac = opt.curve_fit(sine_function_2, bin_centres, ef_fraction, sigma = ef_fraction_err, p0 = [0.03, 0.03], absolute_sigma = True)
popt_ef_frac_line, pcov_ef_frac_line = opt.curve_fit(horizontal_line, bin_centres, ef_fraction, sigma = ef_fraction_err, p0 = [0.05], absolute_sigma = True)
popt_ef_frac_binom, pcov_ef_frac_binom = opt.curve_fit(sine_function_2, bin_centres, ef_fraction, sigma = ef_fraction_err_binom, p0 = [0.03, 0.03], absolute_sigma = True)
popt_ef_frac_line_binom, pcov_ef_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, ef_fraction, sigma = ef_fraction_err_binom, p0 = [0.05], absolute_sigma = True)
popt_eq_frac, pcov_eq_frac = opt.curve_fit(sine_function_2, bin_centres, eq_fraction, sigma = eq_fraction_err, p0 = [0.03, 0.03], absolute_sigma = True)
popt_eq_frac_line, pcov_eq_frac_line = opt.curve_fit(horizontal_line, bin_centres, eq_fraction, sigma = eq_fraction_err, p0 = [0.05], absolute_sigma = True)
popt_eq_frac_binom, pcov_eq_frac_binom = opt.curve_fit(sine_function_2, bin_centres, eq_fraction, sigma = eq_fraction_err_binom, p0 = [0.03, 0.03], absolute_sigma = True)
popt_eq_frac_line_binom, pcov_eq_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, eq_fraction, sigma = eq_fraction_err_binom, p0 = [0.05], absolute_sigma = True)
popt_sq_frac, pcov_sq_frac = opt.curve_fit(sine_function_2, bin_centres, sq_fraction, sigma = sq_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_sq_frac_line, pcov_sq_frac_line = opt.curve_fit(horizontal_line, bin_centres, sq_fraction, sigma = sq_fraction_err, absolute_sigma = True)
popt_sq_frac_binom, pcov_sq_frac_binom = opt.curve_fit(sine_function_2, bin_centres, sq_fraction, sigma = sq_fraction_err_binom, p0 = [0.1, 0.75], absolute_sigma = True)
popt_sq_frac_line_binom, pcov_sq_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, sq_fraction, sigma = sq_fraction_err_binom, absolute_sigma = True)
popt_sf_frac, pcov_sf_frac = opt.curve_fit(sine_function_2, bin_centres, sf_fraction, sigma = sf_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_sf_frac_line, pcov_sf_frac_line = opt.curve_fit(horizontal_line, bin_centres, sf_fraction, sigma = sf_fraction_err, absolute_sigma = True)
popt_sf_frac_binom, pcov_sf_frac_binom = opt.curve_fit(sine_function_2, bin_centres, sf_fraction, sigma = sf_fraction_err_binom, p0 = [0.1, 0.75], absolute_sigma = True)
popt_sf_frac_line_binom, pcov_sf_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, sf_fraction, sigma = sf_fraction_err_binom, absolute_sigma = True)
popt_q_frac, pcov_q_frac = opt.curve_fit(sine_function_2, bin_centres, q_fraction, sigma = q_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_q_frac_line, pcov_q_frac_line = opt.curve_fit(horizontal_line, bin_centres, q_fraction, sigma = q_fraction_err, absolute_sigma = True)
popt_q_frac_binom, pcov_q_frac_binom = opt.curve_fit(sine_function_2, bin_centres, q_fraction, sigma = q_fraction_err_binom, p0 = [0.1, 0.75], absolute_sigma = True)
popt_q_frac_line_binom, pcov_q_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, q_fraction, sigma = q_fraction_err_binom, absolute_sigma = True)
popt_f_frac, pcov_f_frac = opt.curve_fit(sine_function_2, bin_centres, f_fraction, sigma = f_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_f_frac_line, pcov_f_frac_line = opt.curve_fit(horizontal_line, bin_centres, f_fraction, sigma = f_fraction_err, absolute_sigma = True)
popt_f_frac_binom, pcov_f_frac_binom = opt.curve_fit(sine_function_2, bin_centres, f_fraction, sigma = f_fraction_err_binom, p0 = [0.1, 0.75], absolute_sigma = True)
popt_f_frac_line_binom, pcov_f_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, f_fraction, sigma = f_fraction_err_binom, absolute_sigma = True)
popt_e_frac, pcov_e_frac = opt.curve_fit(sine_function_2, bin_centres, e_fraction, sigma = e_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_e_frac_line, pcov_e_frac_line = opt.curve_fit(horizontal_line, bin_centres, e_fraction, sigma = e_fraction_err, absolute_sigma = True)
popt_e_frac_binom, pcov_e_frac_binom = opt.curve_fit(sine_function_2, bin_centres, e_fraction, sigma = e_fraction_err_binom, p0 = [0.1, 0.75], absolute_sigma = True)
popt_e_frac_line_binom, pcov_e_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, e_fraction, sigma = e_fraction_err_binom, absolute_sigma = True)
popt_s_frac, pcov_s_frac = opt.curve_fit(sine_function_2, bin_centres, s_fraction, sigma = s_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_s_frac_line, pcov_s_frac_line = opt.curve_fit(horizontal_line, bin_centres, s_fraction, sigma = s_fraction_err, absolute_sigma = True)
popt_s_frac_binom, pcov_s_frac_binom = opt.curve_fit(sine_function_2, bin_centres, s_fraction, sigma = s_fraction_err_binom, p0 = [0.1, 0.75], absolute_sigma = True)
popt_s_frac_line_binom, pcov_s_frac_line_binom = opt.curve_fit(horizontal_line, bin_centres, s_fraction, sigma = s_fraction_err_binom, absolute_sigma = True)
popt_fs_frac, pcov_fs_frac = opt.curve_fit(sine_function_2, bin_centres, fs_fraction, sigma = fs_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_fs_frac_line, pcov_fs_frac_line = opt.curve_fit(horizontal_line, bin_centres, fs_fraction, sigma = fs_fraction_err, absolute_sigma = True)
popt_qs_frac, pcov_qs_frac = opt.curve_fit(sine_function_2, bin_centres, qs_fraction, sigma = qs_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_qs_frac_line, pcov_qs_frac_line = opt.curve_fit(horizontal_line, bin_centres, qs_fraction, sigma = qs_fraction_err, absolute_sigma = True)
trialY_ef_frac = sine_function_2(trialX, *popt_ef_frac)
trialY_ef_frac_line = horizontal_line(trialX, *popt_ef_frac_line)
trialY_ef_frac_binom = sine_function_2(trialX, *popt_ef_frac_binom)
trialY_ef_frac_line_binom = horizontal_line(trialX, *popt_ef_frac_line_binom)
trialY_eq_frac = sine_function_2(trialX, *popt_eq_frac)
trialY_eq_frac_line = horizontal_line(trialX, *popt_eq_frac_line)
trialY_eq_frac_binom = sine_function_2(trialX, *popt_eq_frac_binom)
trialY_eq_frac_line_binom = horizontal_line(trialX, *popt_eq_frac_line_binom)
trialY_sq_frac = sine_function_2(trialX, *popt_sq_frac)
trialY_sq_frac_line = horizontal_line(trialX, *popt_sq_frac_line)
trialY_sq_frac_binom = sine_function_2(trialX, *popt_sq_frac_binom)
trialY_sq_frac_line_binom = horizontal_line(trialX, *popt_sq_frac_line_binom)
trialY_sf_frac = sine_function_2(trialX, *popt_sf_frac)
trialY_sf_frac_line = horizontal_line(trialX, *popt_sf_frac_line)
trialY_sf_frac_binom = sine_function_2(trialX, *popt_sf_frac_binom)
trialY_sf_frac_line_binom = horizontal_line(trialX, *popt_sf_frac_line_binom)
trialY_q_frac = sine_function_2(trialX, *popt_q_frac)
trialY_q_frac_line = horizontal_line(trialX, *popt_q_frac_line)
trialY_q_frac_binom = sine_function_2(trialX, *popt_q_frac_binom)
trialY_q_frac_line_binom = horizontal_line(trialX, *popt_q_frac_line_binom)
trialY_f_frac = sine_function_2(trialX, *popt_f_frac)
trialY_f_frac_line = horizontal_line(trialX, *popt_f_frac_line)
trialY_f_frac_binom = sine_function_2(trialX, *popt_f_frac_binom)
trialY_f_frac_line_binom = horizontal_line(trialX, *popt_f_frac_line_binom)
trialY_e_frac = sine_function_2(trialX, *popt_e_frac)
trialY_e_frac_line = horizontal_line(trialX, *popt_e_frac_line)
trialY_e_frac_binom = sine_function_2(trialX, *popt_e_frac_binom)
trialY_e_frac_line_binom = horizontal_line(trialX, *popt_e_frac_line_binom)
trialY_s_frac = sine_function_2(trialX, *popt_s_frac)
trialY_s_frac_line = horizontal_line(trialX, *popt_s_frac_line)
trialY_s_frac_binom = sine_function_2(trialX, *popt_s_frac_binom)
trialY_s_frac_line_binom = horizontal_line(trialX, *popt_s_frac_line_binom)
trialY_fs_frac = sine_function_2(trialX, *popt_fs_frac)
trialY_fs_frac_line = horizontal_line(trialX, *popt_fs_frac_line)
trialY_qs_frac = sine_function_2(trialX, *popt_qs_frac)
trialY_qs_frac_line = horizontal_line(trialX, *popt_qs_frac_line)
chi2_red_ef_frac = chi2_red(bin_centres, ef_fraction, ef_fraction_err, popt_ef_frac, sine_function_2)
chi2_red_ef_frac_line = chi2_red(bin_centres, ef_fraction, ef_fraction_err, popt_ef_frac_line, horizontal_line)
chi2_red_eq_frac = chi2_red(bin_centres, eq_fraction, eq_fraction_err, popt_eq_frac, sine_function_2)
chi2_red_eq_frac_line = chi2_red(bin_centres, eq_fraction, eq_fraction_err, popt_eq_frac_line, horizontal_line)
chi2_red_sq_frac = chi2_red(bin_centres, sq_fraction, sq_fraction_err, popt_sq_frac, sine_function_2)
chi2_red_sq_frac_line = chi2_red(bin_centres, sq_fraction, sq_fraction_err, popt_sq_frac_line, horizontal_line)
chi2_red_sf_frac = chi2_red(bin_centres, sf_fraction, sf_fraction_err, popt_sf_frac, sine_function_2)
chi2_red_sf_frac_line = chi2_red(bin_centres, sf_fraction, sf_fraction_err, popt_sf_frac_line, horizontal_line)
chi2_red_q_frac = chi2_red(bin_centres, q_fraction, q_fraction_err, popt_q_frac, sine_function_2)
chi2_red_q_frac_line = chi2_red(bin_centres, q_fraction, q_fraction_err, popt_q_frac_line, horizontal_line)
chi2_red_q_frac_binom = chi2_red(bin_centres, q_fraction, q_fraction_err_binom, popt_q_frac_binom, sine_function_2)
chi2_red_q_frac_line_binom = chi2_red(bin_centres, q_fraction, q_fraction_err_binom, popt_q_frac_line_binom, horizontal_line)
chi2_red_f_frac = chi2_red(bin_centres, f_fraction, f_fraction_err, popt_f_frac, sine_function_2)
chi2_red_f_frac_line = chi2_red(bin_centres, f_fraction, f_fraction_err, popt_f_frac_line, horizontal_line)
chi2_red_f_frac_binom = chi2_red(bin_centres, f_fraction, f_fraction_err_binom, popt_f_frac_binom, sine_function_2)
chi2_red_f_frac_line_binom = chi2_red(bin_centres, f_fraction, f_fraction_err_binom, popt_f_frac_line_binom, horizontal_line)
chi2_red_e_frac = chi2_red(bin_centres, e_fraction, e_fraction_err, popt_e_frac, sine_function_2)
chi2_red_e_frac_line = chi2_red(bin_centres, e_fraction, e_fraction_err, popt_e_frac_line, horizontal_line)
chi2_red_e_frac_binom = chi2_red(bin_centres, e_fraction, e_fraction_err_binom, popt_e_frac_binom, sine_function_2)
chi2_red_e_frac_line_binom = chi2_red(bin_centres, e_fraction, e_fraction_err_binom, popt_e_frac_line_binom, horizontal_line)
chi2_red_s_frac = chi2_red(bin_centres, s_fraction, s_fraction_err, popt_s_frac, sine_function_2)
chi2_red_s_frac_line = chi2_red(bin_centres, s_fraction, s_fraction_err, popt_s_frac_line, horizontal_line)
chi2_red_s_frac_binom = chi2_red(bin_centres, s_fraction, s_fraction_err_binom, popt_s_frac_binom, sine_function_2)
chi2_red_s_frac_line_binom = chi2_red(bin_centres, s_fraction, s_fraction_err_binom, popt_s_frac_line_binom, horizontal_line)

axis_ef_hist, _ = np.histogram(axis_ef, bins=axis_bins)
axis_sf_hist, _ = np.histogram(axis_sf, bins=axis_bins)
axis_eq_hist, _ = np.histogram(axis_eq, bins=axis_bins)
axis_sq_hist, _ = np.histogram(axis_sq, bins=axis_bins)
axis_uu_hist, _ = np.histogram(axis_uu, bins=axis_bins)
axis_ef_err = np.sqrt(axis_ef_hist)
axis_sf_err = np.sqrt(axis_sf_hist)
axis_eq_err = np.sqrt(axis_eq_hist)
axis_sq_err = np.sqrt(axis_sq_hist)
axis_uu_err = np.sqrt(axis_uu_hist)
axis_ef_fraction = np.where(axis_ef_hist + axis_eq_hist > 0, (axis_ef_hist / (axis_ef_hist + axis_eq_hist)), 0)
axis_sq_fraction = np.where(axis_sq_hist + axis_sf_hist > 0, (axis_sq_hist / (axis_sq_hist + axis_sf_hist)), 0)
axis_ef_fraction_err = np.where(axis_ef_hist + axis_eq_hist > 0, np.sqrt(axis_ef_hist) / (axis_ef_hist + axis_eq_hist), np.nan)
axis_sq_fraction_err = np.where(axis_sq_hist + axis_sf_hist > 0, np.sqrt(axis_sq_hist) / (axis_sq_hist + axis_sf_hist), np.nan)
axis_popt_ef_frac, axis_pcov_ef_frac = opt.curve_fit(sine_function_3, axis_bin_centres, axis_ef_fraction, sigma = axis_ef_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
axis_popt_ef_frac_line, axis_pcov_ef_frac_line = opt.curve_fit(horizontal_line, axis_bin_centres, axis_ef_fraction, sigma = axis_ef_fraction_err, absolute_sigma = True)
axis_popt_sq_frac, axis_pcov_sq_frac = opt.curve_fit(sine_function_3, axis_bin_centres, axis_sq_fraction, sigma = axis_sq_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
axis_popt_sq_frac_line, axis_pcov_sq_frac_line = opt.curve_fit(horizontal_line, axis_bin_centres, axis_sq_fraction, sigma = axis_sq_fraction_err, absolute_sigma = True)
axis_trialY_ef_frac = sine_function_3(axis_trialX, *axis_popt_ef_frac)
axis_trialY_ef_frac_line = horizontal_line(axis_trialX, *axis_popt_ef_frac_line)
axis_trialY_sq_frac = sine_function_3(axis_trialX, *axis_popt_sq_frac)
axis_trialY_sq_frac_line = horizontal_line(axis_trialX, *axis_popt_sq_frac_line)
axis_chi2_red_ef_frac = chi2_red(axis_bin_centres, axis_ef_fraction, axis_ef_fraction_err, axis_popt_ef_frac, sine_function_2)
axis_chi2_red_ef_frac_line = chi2_red(axis_bin_centres, axis_ef_fraction, axis_ef_fraction_err, axis_popt_ef_frac_line, horizontal_line)
axis_chi2_red_sq_frac = chi2_red(axis_bin_centres, axis_sq_fraction, axis_sq_fraction_err, axis_popt_sq_frac, sine_function_2)
axis_chi2_red_sq_frac_line = chi2_red(axis_bin_centres, axis_sq_fraction, axis_sq_fraction_err, axis_popt_sq_frac_line, horizontal_line)

spiral_hist, _ = np.histogram(spirals, bins=bins)
elliptical_hist, _ = np.histogram(ellipticals, bins=bins)
unknown_hist, _ = np.histogram(unknowns, bins=bins, density=False)
sfr_forming_hist, _ = np.histogram(sfr_forming, bins=bins)
sfr_quiescent_hist, _ = np.histogram(sfr_quiescent, bins=bins)
sfr_unknown_hist, _ = np.histogram(sfr_unknowns, bins=bins)
spiral_errors = np.sqrt(spiral_hist)
elliptical_errors = np.sqrt(elliptical_hist)
unknown_errors = np.sqrt(unknown_hist)
sfr_forming_errors = np.sqrt(sfr_forming_hist)
sfr_quiescent_errors = np.sqrt(sfr_quiescent_hist)
sfr_unknown_errors = np.sqrt(sfr_unknown_hist)
fraction = np.where(spiral_hist + elliptical_hist + unknown_hist > 0, (elliptical_hist / (spiral_hist + elliptical_hist + unknown_hist)), 0)
fraction_errors = np.where(elliptical_hist + spiral_hist + unknown_hist > 0, np.sqrt(elliptical_hist) / (elliptical_hist + spiral_hist + unknown_hist), np.nan)
sfr_fraction = np.where(sfr_forming_hist + sfr_quiescent_hist + sfr_unknown_hist> 0, (sfr_quiescent_hist / (sfr_forming_hist + sfr_quiescent_hist + sfr_unknown_hist)), 0)
sfr_fraction_errors = np.where(sfr_quiescent_hist + sfr_forming_hist + sfr_unknown_hist> 0, np.sqrt(sfr_quiescent_hist) / (sfr_quiescent_hist +  sfr_forming_hist + sfr_unknown_hist), np.nan)
popt_sfr, pcov_sfr = opt.curve_fit(sine_function, sat_majoraxis_list, sfr_list, sigma = sfr_error, p0 = [1, -11, 0], absolute_sigma = True)
popt_frac, pcov_frac = opt.curve_fit(sine_function_2, bin_centres, fraction, sigma = fraction_errors, p0 = [0.1, 0.75], absolute_sigma = True)
popt_frac_line, pcov_frac_line = opt.curve_fit(horizontal_line, bin_centres, fraction, sigma = fraction_errors, absolute_sigma = True)
popt_sfr_frac, pcov_sfr_frac = opt.curve_fit(sine_function_2, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, p0 = [0.1, 0], absolute_sigma = True)
popt_sfr_frac_line, pcov_sfr_frac_line = opt.curve_fit(horizontal_line, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, absolute_sigma = True)
trialY_frac = sine_function_2(trialX, *popt_frac)
trialY_frac_line = horizontal_line(trialX, *popt_frac_line)
trialY_sfr = sine_function(trialX, *popt_sfr)
trialY_sfr_frac = sine_function_2(trialX, *popt_sfr_frac)
trialY_sfr_frac_line = horizontal_line(trialX, *popt_sfr_frac_line)
chi2_red_frac = chi2_red(bin_centres, fraction, fraction_errors, popt_frac, sine_function_2)
chi2_red_frac_line = chi2_red(bin_centres, fraction, fraction_errors, popt_frac_line, horizontal_line)
chi2_red_sfr_frac = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac, sine_function_2)
chi2_red_sfr_frac_line = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac_line, horizontal_line)

sfr_err_binned, _ = np.histogram(sat_majoraxis_list, sfr_bins, weights=sfr_error**2)
sfr_bin_counts, sfr_bin_edges = np.histogram(sat_majoraxis_list, sfr_bins)
sfr_binned, _ = np.histogram(sat_majoraxis_list, sfr_bins, weights = sfr_list)
sfr_mean = np.where(sfr_bin_counts > 0, sfr_binned / sfr_bin_counts, np.nan)
sfr_error_mean = np.where(sfr_bin_counts > 0, np.sqrt(sfr_binned) / sfr_bin_counts, np.nan)
sfr_bin_centres = (sfr_bin_edges[:-1] + sfr_bin_edges[1:]) /2
popt_avgsfr, pcov_avgsfr = opt.curve_fit(sine_function_2, sfr_bin_centres, sfr_mean, sigma = sfr_error_mean, p0 = [0.5, -12], absolute_sigma = True)
popt_avgsfr_line, pcov_avgsfr_line = opt.curve_fit(horizontal_line, sfr_bin_centres, sfr_mean, sigma = sfr_error_mean, absolute_sigma = True)
trialY_avgsfr = sine_function_2(trialX, *popt_avgsfr)
trialY_avgsfr_line = horizontal_line(trialX, *popt_avgsfr_line)
sfr_chi2_red = chi2_red(sfr_bin_centres, sfr_mean, sfr_error_mean, popt_avgsfr, sine_function_2)
sfr_chi2_red_line = chi2_red(sfr_bin_centres, sfr_mean, sfr_error_mean, popt_avgsfr_line, horizontal_line)

axis_spiral_hist, _ = np.histogram(axis_spirals, bins=axis_bins)
axis_elliptical_hist, _ = np.histogram(axis_ellipticals, bins=axis_bins)
axis_unknown_hist, _ = np.histogram(axis_unknowns, bins=axis_bins)
axis_sfr_forming_hist, _ = np.histogram(axis_sfr_forming, bins=axis_bins)
axis_sfr_quiescent_hist, _ = np.histogram(axis_sfr_quiescent, bins=axis_bins)
axis_spiral_errors = np.sqrt(axis_spiral_hist)
axis_elliptical_errors = np.sqrt(axis_elliptical_hist)
axis_unknown_errors = np.sqrt(axis_unknown_hist)
axis_sfr_forming_errors = np.sqrt(axis_sfr_forming_hist)
axis_sfr_quiescent_errors = np.sqrt(axis_sfr_quiescent_hist)
axis_fraction = np.where(axis_spiral_hist + axis_elliptical_hist > 0, (axis_elliptical_hist / (axis_spiral_hist + axis_elliptical_hist)), 0)
axis_fraction_errors = np.where(axis_elliptical_hist + axis_spiral_hist > 0, np.sqrt(axis_elliptical_hist) / (axis_elliptical_hist + axis_spiral_hist), np.nan)
axis_sfr_fraction = np.where(axis_sfr_forming_hist + axis_sfr_quiescent_hist > 0, (axis_sfr_quiescent_hist / (axis_sfr_forming_hist + axis_sfr_quiescent_hist)), 0)
axis_sfr_fraction_errors = np.where(axis_sfr_quiescent_hist + axis_sfr_forming_hist > 0, np.sqrt(axis_sfr_quiescent_hist) / (axis_sfr_quiescent_hist +  axis_sfr_forming_hist), np.nan)
axis_popt_frac, axis_pcov_frac = opt.curve_fit(sine_function_2, axis_bin_centres, axis_fraction, sigma = axis_fraction_errors, p0 = [0.1, 0.75], absolute_sigma = True)
axis_popt_frac_line, axis_pcov_frac_line = opt.curve_fit(horizontal_line, axis_bin_centres, axis_fraction, sigma = axis_fraction_errors, absolute_sigma = True)
axis_popt_sfr_frac, axis_pcov_sfr_frac = opt.curve_fit(sine_function_2, axis_bin_centres, axis_sfr_fraction, sigma = axis_sfr_fraction_errors, p0 = [0.1, 0.75], absolute_sigma = True)
axis_popt_sfr_frac_line, axis_pcov_sfr_frac_line = opt.curve_fit(horizontal_line, axis_bin_centres, axis_sfr_fraction, sigma = axis_sfr_fraction_errors, absolute_sigma = True)
axis_trialY_frac = sine_function_2(axis_trialX, *axis_popt_frac)
axis_trialY_frac_line = horizontal_line(axis_trialX, *axis_popt_frac_line)
axis_trialY_sfr_frac = sine_function_2(axis_trialX, *axis_popt_sfr_frac)
axis_trialY_sfr_frac_line = horizontal_line(axis_trialX, *axis_popt_sfr_frac_line)
axis_chi2_red_frac = chi2_red(axis_bin_centres, axis_fraction, axis_fraction_errors, axis_popt_frac, sine_function_2)
axis_chi2_red_frac_line = chi2_red(axis_bin_centres, axis_fraction, axis_fraction_errors, axis_popt_frac_line, horizontal_line)
axis_chi2_red_sfr_frac = chi2_red(axis_bin_centres, axis_sfr_fraction, axis_sfr_fraction_errors, axis_popt_sfr_frac, sine_function_2)
axis_chi2_red_sfr_frac_line = chi2_red(axis_bin_centres, axis_sfr_fraction, axis_sfr_fraction_errors, axis_popt_sfr_frac_line, horizontal_line)

axis_sfr_bin_counts, axis_sfr_bin_edges = np.histogram(axis_sat_majoraxis_list, axis_bins)
axis_sfr_binned, _ = np.histogram(axis_sat_majoraxis_list, axis_bins, weights = sfr_list)
axis_sfr_mean = axis_sfr_binned / axis_sfr_bin_counts
axis_sfr_err_binned, _ = np.histogram(axis_sat_majoraxis_list, axis_bins, weights = sfr_error)
axis_sfr_error_mean = axis_sfr_err_binned / axis_sfr_bin_counts

phase_bins = np.arange(0, max_r200, phase_bin_size)
phase_bin_centres = (phase_bins[:-1] + phase_bins[1:]) / 2
phase_trialX = np.linspace(0, max_r200, 1000)
phase_ef_hist, _ = np.histogram(phase_ef, bins=phase_bins)
phase_sf_hist, _ = np.histogram(phase_sf, bins=phase_bins)
phase_eq_hist, _ = np.histogram(phase_eq, bins=phase_bins)
phase_sq_hist, _ = np.histogram(phase_sq, bins=phase_bins)
phase_uu_hist, _ = np.histogram(phase_uu, bins=phase_bins)
phase_q_hist, _ = np.histogram(phase_q, bins=phase_bins)
phase_f_hist, _ = np.histogram(phase_f, bins=phase_bins)
phase_ef_err = np.sqrt(phase_ef_hist)
phase_sf_err = np.sqrt(phase_sf_hist)
phase_eq_err = np.sqrt(phase_eq_hist)
phase_sq_err = np.sqrt(phase_sq_hist)
phase_uu_err = np.sqrt(phase_uu_hist)
phase_q_err = np.sqrt(phase_q_hist)
phase_f_err = np.sqrt(phase_f_hist)
tot_ellipticals = phase_ef_hist + phase_eq_hist
tot_spirals = phase_sq_hist + phase_sf_hist
tot_morph = tot_spirals + tot_ellipticals 
tot_gals = phase_q_hist + phase_f_hist
tot_quiescent = phase_eq_hist + phase_sq_hist
tot_forming = phase_ef_hist + phase_sf_hist
phase_eq_fraction = np.where(phase_ef_hist + phase_eq_hist > 0, (phase_eq_hist / (phase_ef_hist + phase_eq_hist)), 0)
phase_sq_fraction = np.where(phase_sq_hist + phase_sf_hist> 0, (phase_sq_hist / (phase_sq_hist + phase_sf_hist)), 0)
phase_q_fraction = np.where(tot_gals > 0, (phase_q_hist / (tot_gals)), 0)
phase_qs_fraction = np.where(tot_quiescent > 0, (phase_sq_hist / (tot_quiescent)), 0)
phase_fs_fraction = np.where(tot_forming > 0, (phase_sf_hist / (tot_forming)), 0)
phase_s_fraction = np.where(tot_morph > 0, (tot_spirals / (tot_morph )), 0)

phase_eq_fraction_err = np.where(tot_ellipticals > 0, np.sqrt((phase_eq_hist / tot_ellipticals) * (1 - (phase_eq_hist / tot_ellipticals)) / tot_ellipticals), np.nan)
phase_sq_fraction_err = np.where(tot_spirals > 0, np.sqrt((phase_sq_hist / tot_spirals) * (1 - (phase_sq_hist / tot_spirals)) / tot_spirals), np.nan)
phase_q_fraction_err = np.where(tot_gals > 0, np.sqrt((phase_q_hist / tot_gals) * (1 - (phase_q_hist / tot_gals)) / tot_gals), np.nan)
phase_qs_fraction_err = np.where(tot_quiescent > 0, np.sqrt((phase_sq_hist / tot_quiescent) * (1 - (phase_sq_hist / tot_quiescent)) / tot_quiescent), np.nan)
phase_fs_fraction_err = np.where(tot_forming > 0, np.sqrt((phase_sf_hist / tot_forming) * (1 - (phase_sf_hist / tot_forming)) / tot_forming), np.nan)
phase_s_fraction_err = np.where(tot_morph > 0, np.sqrt((tot_spirals / tot_morph) * (1 - (tot_spirals / tot_morph)) / tot_morph), np.nan)

if show_eq_phase == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(phase_bin_centres, phase_eq_fraction, yerr=phase_eq_fraction_err, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Ellipticals", fontsize=16)
    ax.set_ylim(np.nanmin(phase_eq_fraction) * 0.8, np.nanmax(phase_eq_fraction) * 1.2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)    
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if show_sq_phase == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(phase_bin_centres, phase_sq_fraction, yerr=phase_sq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Spirals", fontsize=16)
    ax.set_ylim(np.nanmin(phase_sq_fraction) * 0.8, np.nanmax(phase_sq_fraction) * 1.2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)    
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if show_q_phase == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(phase_bin_centres, phase_q_fraction, yerr=phase_q_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Galaxies", fontsize=16)
    ax.set_ylim(np.nanmin(phase_q_fraction) * 0.8, np.nanmax(phase_q_fraction) * 1.2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)    
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if show_qs_phase == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(phase_bin_centres, phase_qs_fraction, yerr=phase_qs_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Galaxies Which Are Spirals", fontsize=16)
    ax.set_ylim(np.nanmin(phase_qs_fraction) * 0.8, np.nanmax(phase_qs_fraction) * 1.2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)    
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if show_fs_phase == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(phase_bin_centres, phase_fs_fraction, yerr=phase_fs_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Fraction of Star-Forming Galaxies Which Are Spirals", fontsize=16)
    ax.set_ylim(np.nanmin(phase_fs_fraction) * 0.8, np.nanmax(phase_fs_fraction) * 1.2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)    
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if show_phase_combined == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(phase_bin_centres, phase_sq_fraction, yerr=phase_sq_fraction_err, marker='o', linestyle='-', color="purple", label="Spiral Quiescent Fraction", capsize=2)
    ax.errorbar(phase_bin_centres, phase_eq_fraction, yerr=phase_eq_fraction_err, marker='x', linestyle='-', color="blue", label="Elliptical Quiescent Fraction", capsize=2)
    ax.errorbar(phase_bin_centres, phase_q_fraction, yerr=phase_q_fraction_err, marker='D', linestyle='-', color="orange", label="Quiescent Fraction", capsize=2)
    ax.errorbar(phase_bin_centres, phase_qs_fraction, yerr=phase_qs_fraction_err, marker='s', linestyle='-', color="grey", label="Quiescent Spiral Fraction", capsize=2)
    ax.errorbar(phase_bin_centres, phase_fs_fraction, yerr=phase_fs_fraction_err, marker='p', linestyle='-', color="red", label="Star-Forming Spiral Fraction Amplitude", capsize=2)
    ax.errorbar(phase_bin_centres, phase_s_fraction, yerr=phase_s_fraction_err, marker='h', linestyle='-', color="green", label="Spiral Fraction Amplitude", capsize=2)

    ax.set_xlabel(r"R/R$_{200}$", fontsize=18)
    ax.set_ylabel("Galaxy Fraction", fontsize=18)
    ax.set_ylim(0, 1.4)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)    
    ax.legend(fontsize=18, frameon=True, facecolor="white", edgecolor="black", framealpha=1)
    ax.grid(axis="y", linestyle="--", alpha=1, color = 'black')
    plt.show()    

## 0-180 plots for elliptical and quiescent fractions.
"""#3 d.o.f.
print("Elliptical fraction")
print(f"Sinusoid reduced chi squared: {chi2_red_frac:.3f}")
print(f"y = ({popt_frac[0]:.2f} ± {np.sqrt(pcov_frac[0,0]):.2f})sin(x + ({popt_frac[2]:.2f} ± {np.sqrt(pcov_frac[2,2]):.2f})) + ({popt_frac[1]:.2f} ± {np.sqrt(pcov_frac[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {chi2_red_frac_line:.3f}")
print(f"y = {popt_frac_line[0]:.2f} ± {np.sqrt(pcov_frac_line[0,0]):.2f}")
print("Quiescent fraction")
print(f"Sinusoid reduced chi squared: {chi2_red_sfr_frac:.3f}")
print(f"y = ({popt_sfr_frac[0]:.2f} ± {np.sqrt(pcov_sfr_frac[0,0]):.2f})sin(x + ({popt_sfr_frac[2]:.2f} ± {np.sqrt(pcov_sfr_frac[2,2]):.2f})) + ({popt_sfr_frac[1]:.2f} ± {np.sqrt(pcov_sfr_frac[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {chi2_red_sfr_frac_line:.3f}")
print(f"y = {popt_sfr_frac_line[0]:.2f} ± {np.sqrt(pcov_sfr_frac_line[0,0]):.2f}")"""

"""
#2 d.o.f.
print("Elliptical fraction")
print(f"Sinusoid reduced chi squared: {chi2_red_frac:.3f}")
print(f"y = ({popt_frac[0]:.2f} ± {np.sqrt(pcov_frac[0,0]):.2f})sin(x) + ({popt_frac[1]:.2f} ± {np.sqrt(pcov_frac[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {chi2_red_frac_line:.3f}")
print(f"y = {popt_frac_line[0]:.2f} ± {np.sqrt(pcov_frac_line[0,0]):.2f}")
print("Quiescent fraction")
print(f"Sinusoid reduced chi squared: {chi2_red_sfr_frac:.3f}")
print(f"y = ({popt_sfr_frac[0]:.2f} ± {np.sqrt(pcov_sfr_frac[0,0]):.2f})sin(x) + ({popt_sfr_frac[1]:.2f} ± {np.sqrt(pcov_sfr_frac[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {chi2_red_sfr_frac_line:.3f}")
print(f"y = {popt_sfr_frac_line[0]:.2f} ± {np.sqrt(pcov_sfr_frac_line[0,0]):.2f}")
"""

##Separate Plots
#Elliptical fraction
if show_elliptical == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
    ax.errorbar(bin_centres, fraction, yerr=fraction_errors, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Fraction of Ellipticals")
    #ax.set_title("Elliptical Fraction as a Function of Angle")
    ax.set_ylim(np.nanmin(fraction) * 0.8, np.nanmax(fraction) * 1.2)
    ax.plot(trialX, trialY_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_frac[0]:.3f} ± {np.sqrt(pcov_frac[0,0]):.3f})') 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    #ax.text(0.7, 0.7, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax.transAxes, =8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    # Add a text box near the vertical line
    ax.text(90, np.nanmax(fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()


#Quiescent fraction
if show_quiescent == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
    ax.errorbar(bin_centres, sfr_fraction, yerr=sfr_fraction_errors, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Fraction of Quiescent Galaxies")
    #ax.set_title("Quiescent Fraction as a Function of Angle")
    ax.set_ylim(np.nanmin(sfr_fraction) * 0.8, np.nanmax(sfr_fraction) * 1.2)
    ax.plot(trialX, trialY_sfr_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_sfr_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sfr_frac[0]:.3f} ± {np.sqrt(pcov_sfr_frac[0,0]):.3f})') 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    #ax.text(0.7, 0.7, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax.transAxes, fontsize=8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    # Add a text box near the vertical line
    ax.text(90, np.nanmax(sfr_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

"""##Together
fig, ax = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True, dpi = 150)
#Elliptical fraction
ax[0].errorbar(bin_centres, fraction, yerr=fraction_errors, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
ax[0].set_ylabel("Fraction of Ellipticals")
ax[0].set_title("Elliptical Fraction as a Function of Angle")
ax[0].set_ylim(np.nanmin(fraction) * 0.8, np.nanmax(fraction) * 1.2)
ax[0].plot(trialX, trialY_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[0].plot(trialX, trialY_frac, 'r-', label = 'Sinusoidal Fit') 
ax[0].legend()
ax[0].grid(axis="y", linestyle="--", alpha=0.7)
ax[0].text(0.7, 0.7, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[0].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

#Quiescent fraction
ax[1].errorbar(bin_centres, sfr_fraction, yerr=sfr_fraction_errors, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
ax[1].set_xlabel("Angle (degrees)")
ax[1].set_ylabel("Fraction of Quiescent Galaxies")
ax[1].set_title("Quiescent Fraction as a Function of Angle")
ax[1].set_ylim(np.nanmin(sfr_fraction) * 0.8, np.nanmax(sfr_fraction) * 1.2)
ax[1].plot(trialX, trialY_sfr_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[1].plot(trialX, trialY_sfr_frac, 'r-', label = 'Sinusoidal Fit') 
ax[1].legend()
ax[1].grid(axis="y", linestyle="--", alpha=0.7)
ax[1].text(0.7, 0.7, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()"""

"""
## 0-180 plots for morphology and sSFR galaxy counts, elliptical and quiescent fractions.
fig, ax = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

#Galaxy counts
ax[0,0].errorbar(bin_centres, spiral_hist, yerr=spiral_errors, label="Spirals", color="blue", capsize=2)
ax[0,0].errorbar(bin_centres, elliptical_hist, yerr=elliptical_errors ,label="Ellipticals", color="red", capsize=2)
ax[0,0].errorbar(bin_centres, unknown_hist, yerr=unknown_errors, label="Unknowns", color="green", capsize=2)
ax[0,0].set_ylabel("Number of Galaxies")
ax[0,0].set_title("Galaxy Angle Distribution: Classification")
ax[0,0].legend()
ax[0,0].grid(axis="y", linestyle="--", alpha=0.7)

#Elliptical fraction
ax[0,1].errorbar(bin_centres, fraction, yerr=fraction_errors, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
ax[0,1].set_ylabel("Fraction of Ellipticals")
ax[0,1].set_title("Elliptical Fraction as a Function of Angle")
ax[0,1].set_ylim(np.nanmax(fraction) * 0.8, np.nanmax(fraction) * 1.2)
ax[0,1].plot(trialX, trialY_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[0,1].plot(trialX, trialY_frac, 'r-', label = 'Sinusoidal Fit') 
ax[0,1].legend()
ax[0,1].grid(axis="y", linestyle="--", alpha=0.7)
ax[0,1].text(0.7, 0.7, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[0,1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

ax[1,0].errorbar(bin_centres, sfr_forming_hist, yerr=sfr_forming_errors, label="Star-Forming", color="blue", capsize=2)
ax[1,0].errorbar(bin_centres, sfr_quiescent_hist, yerr=sfr_quiescent_errors ,label="Quiescent", color="red", capsize=2)
ax[1,0].errorbar(bin_centres, sfr_unknown_hist, yerr=sfr_unknown_errors, label="Unknowns", color="green", capsize=2)
ax[1,0].set_xlabel("Angle (degrees)")
ax[1,0].set_ylabel("Number of Galaxies")
ax[1,0].set_title("Galaxy Angle Distribution: SSFR")
ax[1,0].legend()
ax[1,0].grid(axis="y", linestyle="--", alpha=0.7)

#Quiescent fraction
ax[1,1].errorbar(bin_centres, sfr_fraction, yerr=sfr_fraction_errors, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
ax[1,1].set_xlabel("Angle (degrees)")
ax[1,1].set_ylabel("Fraction of Quiescent Galaxies")
ax[1,1].set_title("Quiescent Fraction as a Function of Angle")
ax[1,1].set_ylim(np.nanmax(sfr_fraction) * 0.8, np.nanmax(sfr_fraction) * 1.2)
ax[1,1].plot(trialX, trialY_sfr_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[1,1].plot(trialX, trialY_sfr_frac, 'r-', label = 'Sinusoidal Fit') 
ax[1,1].plot(trialX, trialY_sfr_frac_cos, 'b-', label = 'Cosine Fit') 
ax[1,1].legend()
ax[1,1].grid(axis="y", linestyle="--", alpha=0.7)
ax[1,1].text(0.7, 0.7, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[1, 1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()
"""

## 0-180 sSFR plots
if show_ssfr == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
    print("Elliptical fraction")
    print(f"Sinusoid reduced chi squared: {sfr_chi2_red:.3f}")
    print(f"y = ({popt_avgsfr[0]:.2f} ± {np.sqrt(pcov_avgsfr[0,0]):.2f})sin(x) + ({popt_avgsfr[1]:.2f} ± {np.sqrt(pcov_avgsfr[1,1]):.2f})")
    print(f"Horizontal line reduced chi squared: {sfr_chi2_red_line:.3f}")
    print(f"y = {popt_avgsfr_line[0]:.2f} ± {np.sqrt(pcov_avgsfr_line[0,0]):.2f}")

    ax.scatter(sat_majoraxis_list, sfr_list, label="sSFR", color="orange", marker = 'o', s = 5, linewidth = 0.5, linestyle = 'None')
    ax.set_ylabel("sSFR", fontsize=16)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_title("Specific Star Formation Rate as a Function of Angle", fontsize=18)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.errorbar(sfr_bin_centres, sfr_mean, yerr=sfr_error_mean, label="Binned sSFR", markerfacecolor='blue', markeredgecolor='black', ecolor='blue', capsize=2, marker = 's', markersize = 10, linewidth = 0.5, linestyle = 'None')
    ax.plot(trialX, trialY_avgsfr, 'g-', label = 'Sinusoidal Fit') 
    ax.plot(trialX, trialY_avgsfr_line, 'r-', label = 'Horizontal Line Fit')
    ax.set_ylim(-13, -10.5)
    ax.set_xlim(0, 180)
    ax.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()

"""
## -45 - 135 sSFR plots
fig, ax = plt.subplots(1, 1, figsize=(16, 12), constrained_layout=True, dpi = 125)

print("Elliptical fraction")
print(f"Sinusoid reduced chi squared: {axis_sfr_chi2_red:.3f}")
print(f"y = ({axis_popt_avgsfr[0]:.2f} ± {np.sqrt(axis_pcov_avgsfr[0,0]):.2f})sin(x + ({axis_popt_avgsfr[2]:.2f} ± {np.sqrt(axis_pcov_avgsfr[2,2]):.2f})) + ({axis_popt_avgsfr[1]:.2f} ± {np.sqrt(axis_pcov_avgsfr[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {axis_sfr_chi2_red_line:.3f}")
print(f"y = {axis_popt_avgsfr_line[0]:.2f} ± {np.sqrt(axis_pcov_avgsfr_line[0,0]):.2f}")

ax.scatter(axis_sat_majoraxis_list, sfr_list, label="sSFR", color="orange", marker = 'o', s = 5, linewidth = 0.5, linestyle = 'None')
ax.set_ylabel("sSFR", fontsize=16)
ax.set_xlabel("Angle (degrees)", fontsize=16)
ax.set_title("Specific Star Formation Rate as a Function of Angle", fontsize=18)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.errorbar(axis_bin_centres, axis_sfr_mean, yerr=axis_sfr_error_mean, label="Binned sSFR", markerfacecolor='blue', markeredgecolor='black', ecolor='blue', capsize=2, marker = 's', markersize = 10, linewidth = 0.5, linestyle = 'None')
ax.plot(axis_trialX, axis_trialY_avgsfr, 'g-', label = 'Sinusoidal Fit') 
ax.plot(axis_trialX, axis_trialY_avgsfr_line, 'r-', label = 'Horizontal Line Fit')
ax.set_ylim(-13, -10.5)  # Modify the Y-axis limits as needed
ax.set_xlim(-45, 135)  # Modify the Y-axis limits as needed
ax.legend(fontsize=14)
plt.tick_params(axis='both', labelsize=12)  # Adjust the tick labels size for both axes

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()"""

"""
## 0-180 morphology + sSFR plots, elliptical star forming fraction, sprial quiescent fraction
fig, ax = plt.subplots(1, 1, figsize=(16, 12), constrained_layout=True, dpi=200)
#Galaxy counts
ax.errorbar(bin_centres, ef_hist, yerr=ef_err, label="Elliptical Star Forming", color="blue", capsize=2)
ax.errorbar(bin_centres, eq_hist, yerr=eq_err, label="Elliptical Quiescent", color="green", capsize=2)
ax.errorbar(bin_centres, sf_hist, yerr=sf_err, label="Spiral Star Forming", color="red", capsize=2)
ax.errorbar(bin_centres, sq_hist, yerr=sq_err, label="Spiral Quiescent", color="grey", capsize=2)
ax.set_ylabel("Number of Galaxies")
ax.set_title("Galaxy Angle Distribution: Classification + sSFR")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()"""

#EF fraction
if show_ef == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, ef_fraction, yerr=ef_fraction_err, marker='o', linestyle='-', color="purple", label="Star-Forming Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star Forming Ellipticals", fontsize=16)
    #ax.set_title("Fraction of ellipticals which are star forming as a function of angle")
    ax.set_ylim(np.nanmin(ef_fraction) * 0.8, np.nanmax(ef_fraction) * 1.2)
    ax.plot(trialX, trialY_ef_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_ef_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_ef_frac[0]:.3f} ± {np.sqrt(pcov_ef_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(ef_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(ef_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(ef_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_ef_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, ef_fraction, yerr=ef_fraction_err_binom, marker='o', linestyle='-', color="purple", label="Star-Forming Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star Forming Ellipticals (Binomial)", fontsize=16)
    #ax.set_title("Fraction of ellipticals which are star forming as a function of angle")
    ax.set_ylim(np.nanmin(ef_fraction) * 0.8, np.nanmax(ef_fraction) * 1.2)
    ax.plot(trialX, trialY_ef_frac_line_binom, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_ef_frac_binom, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_ef_frac_binom[0]:.3f} ± {np.sqrt(pcov_ef_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(ef_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(ef_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(ef_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_eq == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, eq_fraction, yerr=eq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Ellipticals", fontsize=16)
    #ax.set_title("Fraction of ellipticals which are star forming as a function of angle")
    ax.set_ylim(np.nanmin(eq_fraction) * 0.8, np.nanmax(eq_fraction) * 1.2)
    ax.plot(trialX, trialY_eq_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_eq_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_eq_frac[0]:.3f} ± {np.sqrt(pcov_eq_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(eq_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(eq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(eq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_eq_binom == 1:
    """fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, eq_fraction, yerr=eq_fraction_err_binom, marker='o', linestyle='-', color="purple", label="Quiescent Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Ellipticals (Binomial)", fontsize=16)
    #ax.set_title("Fraction of ellipticals which are star forming as a function of angle")
    ax.set_ylim(np.nanmin(eq_fraction) * 0.8, np.nanmax(eq_fraction) * 1.2)
    ax.plot(trialX, trialY_eq_frac_line_binom, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_eq_frac_binom, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_eq_frac_binom[0]:.3f} ± {np.sqrt(pcov_eq_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(eq_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(eq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(eq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()"""

    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, eq_fraction, yerr=eq_fraction_err_binom, linestyle='none', linewidth = 3, marker='o', markersize = 10, markeredgecolor='black', markeredgewidth=1.5, color="red", label="Medians", capsize=4)
    ax.set_xlabel("Angle from Major Axis (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Ellipticals Which Are Quiescent", fontsize=16)
    ax.set_ylim(np.nanmin(eq_fraction) * 0.9, np.nanmax(eq_fraction) * 1.1)
    ax.plot(trialX, trialY_eq_frac_line_binom, linestyle='-', color = 'blue', linewidth = 2, label = 'Linear fit') 
    ax.plot(trialX, trialY_eq_frac_binom, linestyle='-', color = 'red', linewidth = 2, label = f'Sinusoid fit (amplitude = {popt_eq_frac_binom[0]:.3f} ± {np.sqrt(pcov_eq_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16, loc = 'lower right', bbox_to_anchor=(1, 0))
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=2)
    ax.text(90, np.nanmax(eq_fraction) * 1.05, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=2)
    ax.text(0, np.nanmax(eq_fraction) * 1.05, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=2)
    ax.text(180, np.nanmax(eq_fraction) * 1.05, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.show()

if show_sq == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, sq_fraction, yerr=sq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Spiral Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Spirals", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(sq_fraction) * 0.8, np.nanmax(sq_fraction) * 1.2)
    ax.plot(trialX, trialY_sq_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_sq_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sq_frac[0]:.3f} ± {np.sqrt(pcov_sq_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(sq_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(sq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(sq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_sq_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, sq_fraction, yerr=sq_fraction_err_binom, linestyle='none', linewidth = 3, marker='o', markersize = 10, markeredgecolor='black', markeredgewidth=1.5, color="red", label="Medians", capsize=4)
    ax.set_xlabel("Angle from Major Axis (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Spirals Which Are Quiescent", fontsize=16)
    ax.set_ylim(np.nanmin(sq_fraction) * 0.6, np.nanmax(sq_fraction) * 1.2)
    ax.plot(trialX, trialY_sq_frac_line_binom, linestyle='-', color = 'blue', linewidth = 2, label = 'Linear fit') 
    ax.plot(trialX, trialY_sq_frac_binom, linestyle='-', color = 'red', linewidth = 2, label = f'Sinusoid fit (amplitude = {popt_sq_frac_binom[0]:.3f} ± {np.sqrt(pcov_sq_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16, loc = 'lower right', bbox_to_anchor=(1, 0))
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=2)
    ax.text(90, np.nanmax(sq_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=2)
    ax.text(0, np.nanmax(sq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=2)
    ax.text(180, np.nanmax(sq_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.show()

if show_sf == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, sf_fraction, yerr=sf_fraction_err, marker='o', linestyle='-', color="purple", label="Star-Forming Spiral Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star-Forming Spirals", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(sf_fraction) * 0.8, np.nanmax(sf_fraction) * 1.2)
    ax.plot(trialX, trialY_sf_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_sf_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sf_frac[0]:.3f} ± {np.sqrt(pcov_sf_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(sf_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(sf_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(sf_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_sf_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, sf_fraction, yerr=sf_fraction_err_binom, marker='o', linestyle='-', color="purple", label="Star-Forming Spiral Fraction", capsize=2)
    ax.set_xlabel("Angle from Major Axis (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star-Forming Spirals", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(sf_fraction) * 0.8, np.nanmax(sf_fraction) * 1.2)
    ax.plot(trialX, trialY_sf_frac_line_binom, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_sf_frac_binom, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sf_frac_binom[0]:.3f} ± {np.sqrt(pcov_sf_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(sf_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(sf_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(sf_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_q == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, q_fraction, yerr=q_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Galaxies", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(q_fraction) * 0.8, np.nanmax(q_fraction) * 1.2)
    ax.plot(trialX, trialY_q_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_q_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_q_frac[0]:.3f} ± {np.sqrt(pcov_q_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(q_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(q_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(q_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_q_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, q_fraction, yerr=q_fraction_err_binom, linestyle='none', linewidth = 3, marker='o', markersize = 10, markeredgecolor='black', markeredgewidth=1.5, color="red", label="Medians", capsize=4)
    ax.set_xlabel("Angle from Major Axis (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Galaxies", fontsize=16)
    ax.set_ylim(np.nanmin(q_fraction) * 0.9, np.nanmax(q_fraction) * 1.1)
    ax.plot(trialX, trialY_q_frac_line_binom, linestyle='-', color = 'blue', linewidth = 2, label = 'Linear fit') 
    ax.plot(trialX, trialY_q_frac_binom, linestyle='-', color = 'red', linewidth = 2, label = f'Sinusoid fit (amplitude = {popt_q_frac_binom[0]:.3f} ± {np.sqrt(pcov_q_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16, loc = 'lower right', bbox_to_anchor=(1, 0))
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=2)
    ax.text(90, np.nanmax(q_fraction) * 1.05, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=2)
    ax.text(0, np.nanmax(q_fraction) * 1.05, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=2)
    ax.text(180, np.nanmax(q_fraction) * 1.05, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.show()
    print("q red.chi2:")
    print("sine", chi2_red_q_frac_binom)
    print("line", chi2_red_q_frac_line_binom)

if show_f == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, f_fraction, yerr=f_fraction_err, marker='o', linestyle='-', color="purple", label="Star-Forming Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star-Forming Galaxies", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(f_fraction) * 0.8, np.nanmax(f_fraction) * 1.2)
    ax.plot(trialX, trialY_f_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_f_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_f_frac[0]:.3f} ± {np.sqrt(pcov_f_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(f_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(f_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(f_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    #ax[2].text(0.1, 0.2, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax[2].transAxes, fontsize=8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.show()

if show_f_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, f_fraction, yerr=f_fraction_err_binom, marker='o', linestyle='-', color="purple", label="Star-Forming Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star-Forming Galaxies (Binomial)", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(f_fraction) * 0.8, np.nanmax(f_fraction) * 1.2)
    ax.plot(trialX, trialY_f_frac_line_binom, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_f_frac_binom, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_f_frac_binom[0]:.3f} ± {np.sqrt(pcov_f_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(f_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(f_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(f_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_e == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, e_fraction, yerr=e_fraction_err, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Elliptical Galaxies", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(e_fraction) * 0.8, np.nanmax(e_fraction) * 1.2)
    ax.plot(trialX, trialY_e_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_e_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_e_frac[0]:.3f} ± {np.sqrt(pcov_e_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(e_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(e_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(e_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_e_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, e_fraction, yerr=e_fraction_err_binom, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Elliptical Galaxies (Binomial)", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(e_fraction) * 0.8, np.nanmax(e_fraction) * 1.2)
    ax.plot(trialX, trialY_e_frac_line_binom, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_e_frac_binom, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_e_frac_binom[0]:.3f} ± {np.sqrt(pcov_e_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(e_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(e_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(e_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_s == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, s_fraction, yerr=s_fraction_err, marker='o', linestyle='-', color="purple", label="Spiral Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Spiral Galaxies", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(s_fraction) * 0.8, np.nanmax(s_fraction) * 1.2)
    ax.plot(trialX, trialY_s_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_s_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_s_frac[0]:.3f} ± {np.sqrt(pcov_s_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(s_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(s_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(s_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_s_binom == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, s_fraction, yerr=s_fraction_err_binom, marker='o', linestyle='-', color="purple", label="Spiral Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Spiral Galaxies", fontsize=16)
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(s_fraction) * 0.8, np.nanmax(s_fraction) * 1.2)
    ax.plot(trialX, trialY_s_frac_line_binom, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_s_frac_binom, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_s_frac_binom[0]:.3f} ± {np.sqrt(pcov_s_frac_binom[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(s_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(s_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(s_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_qs == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, qs_fraction, yerr=qs_fraction_err, marker='o', linestyle='-', color="purple", label="Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Quiescent Galaxies Which Are Spirals", fontsize=16)
    ax.set_ylim(np.nanmin(qs_fraction) * 0.8, np.nanmax(qs_fraction) * 1.2)
    ax.plot(trialX, trialY_qs_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_qs_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_qs_frac[0]:.3f} ± {np.sqrt(pcov_qs_frac[0,0]):.3f})') 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    ax.text(90, np.nanmax(qs_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=1)
    ax.text(0, np.nanmax(qs_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=1)
    ax.text(180, np.nanmax(qs_fraction) * 1.15, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

if show_fs == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, fs_fraction, yerr=fs_fraction_err, linestyle='none', linewidth = 3, marker='o', markersize = 10, markeredgecolor='black', markeredgewidth=1.5, color="red", label="Medians", capsize=4)
    ax.set_xlabel("Angle from Major Axis (degrees)", fontsize=16)
    ax.set_ylabel("Fraction of Star-Forming Galaxies Which are Spirals", fontsize=16)
    ax.set_ylim(0, np.nanmax(fs_fraction) * 1.3)
    ax.plot(trialX, trialY_fs_frac_line, linestyle='-', color = 'blue', linewidth = 2, label = 'Linear fit') 
    ax.plot(trialX, trialY_fs_frac, linestyle='-', color = 'red', linewidth = 2, label = f'Sinusoid fit (amplitude = {popt_fs_frac[0]:.3f} ± {np.sqrt(pcov_fs_frac[0,0]):.3f})') 
    ax.legend(fontsize=16, loc = 'lower right', bbox_to_anchor=(1, 0))
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=2)
    ax.text(90, np.nanmax(fs_fraction) * 1.20, 'Minor Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=0, color='black', linestyle='dotted', linewidth=2)
    ax.text(0, np.nanmax(fs_fraction) * 1.20, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.axvline(x=180, color='black', linestyle='dotted', linewidth=2)
    ax.text(180, np.nanmax(fs_fraction) * 1.20, 'Major Axis', ha='center', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.show()

if show_phase_space == 1:
    colour_map = {
    'ef': '#FF0000',  # Red
    'sf': '#0000FF',  # Blue
    'eq': '#00FF00',  # Green
    'sq': '#800080',  # Purple
    'uu': '#808080'   # Gray
    }
    plt.figure(figsize=(10, 6), dpi = 150)
    for category in colour_map:
        mask = [morph == category for morph in morph_forming_list]
        plt.scatter(
        [radial_sep_list[i] for i in range(len(radial_sep_list)) if mask[i]],
        [vel_diff_list[i] for i in range(len(vel_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel('Radial Separation [kpc]', fontsize=12)
    plt.xlim(0, max(radial_sep_list))
    plt.ylabel(r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    plt.title('Radial Separation vs. Velocity Difference\nColoured by Morphological Class', fontsize=14)
    plt.legend(
    title='Class',
    bbox_to_anchor=(0.75, 0.95),
    loc='upper left',
    facecolor='white',   # Opaque background
    edgecolor='black',   # Border color
    framealpha=1         # Full opacity (no transparency)
)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if show_phase_space_r200 == 1:
    colour_map = {
    'ef': '#FF0000',  # Red
    'sf': '#0000FF',  # Blue
    'eq': '#00FF00',  # Green
    'sq': '#800080',  # Purple
    'uu': '#808080'   # Gray
    }
    plt.figure(figsize=(10, 6), dpi = 150)
    for category in colour_map:
        mask = [morph == category for morph in morph_forming_list]
        plt.scatter(
        [r200_list[i] for i in range(len(r200_list)) if mask[i]],
        [vel_diff_list[i] for i in range(len(vel_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel(r'Radial Separation [R/$R_{200}$]', fontsize=12)
    plt.xlim(0, max(r200_list))
    plt.ylabel(r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    plt.title('Radial Separation vs. Velocity Difference\nColoured by Morphological Class', fontsize=14)
    plt.legend(
    title='Class',
    bbox_to_anchor=(0.75, 0.95),
    loc='upper left',
    facecolor='white',   # Opaque background
    edgecolor='black',   # Border color
    framealpha=1         # Full opacity (no transparency)
)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

"""
## -45-135 plots for morphology and sSFR galaxy counts, elliptical and quiescent fractions.
fig, ax = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

#Galaxy counts
ax[0,0].errorbar(axis_bin_centres, axis_spiral_hist, yerr=axis_spiral_errors, label="Spirals", color="blue", capsize=2)
ax[0,0].errorbar(axis_bin_centres, axis_elliptical_hist, yerr=axis_elliptical_errors ,label="Ellipticals", color="red", capsize=2)
ax[0,0].errorbar(axis_bin_centres, axis_unknown_hist, yerr=axis_unknown_errors, label="Unknowns", color="green", capsize=2)
ax[0,0].set_ylabel("Number of Galaxies")
ax[0,0].set_title("Galaxy Angle Distribution: Classification")
ax[0,0].legend()
ax[0,0].grid(axis="y", linestyle="--", alpha=0.7)

#Elliptical fraction
ax[0,1].errorbar(axis_bin_centres, axis_fraction, yerr=axis_fraction_errors, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
ax[0,1].set_ylabel("Fraction of Ellipticals")
ax[0,1].set_title("Elliptical Fraction as a Function of Angle")
ax[0,1].set_ylim(np.nanmax(axis_fraction) * 0.8, np.nanmax(axis_fraction) * 1.2)
ax[0,1].plot(axis_trialX, axis_trialY_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[0,1].plot(axis_trialX, axis_trialY_frac, 'r-', label = 'Sinusoidal Fit') 
ax[0,1].legend()
ax[0,1].grid(axis="y", linestyle="--", alpha=0.7)
ax[0,1].text(0.7, 0.7, f"Minimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[0,1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

ax[1,0].errorbar(axis_bin_centres, axis_sfr_forming_hist, yerr=axis_sfr_forming_errors, label="Star-Forming", color="blue", capsize=2)
ax[1,0].errorbar(axis_bin_centres, axis_sfr_quiescent_hist, yerr=axis_sfr_quiescent_errors ,label="Quiescent", color="red", capsize=2)
ax[1,0].set_xlabel("Angle (degrees)")
ax[1,0].set_ylabel("Number of Galaxies")
ax[1,0].set_title("Galaxy Angle Distribution: SSFR")
ax[1,0].legend()
ax[1,0].grid(axis="y", linestyle="--", alpha=0.7)

#Quiescent fraction
ax[1,1].errorbar(axis_bin_centres, axis_sfr_fraction, yerr=axis_sfr_fraction_errors, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
ax[1,1].set_xlabel("Angle (degrees)")
ax[1,1].set_ylabel("Fraction of Quiescent Galaxies")
ax[1,1].set_title("Quiescent Fraction as a Function of Angle")
ax[1,1].set_ylim(np.nanmax(axis_sfr_fraction) * 0.8, np.nanmax(axis_sfr_fraction) * 1.2)
ax[1,1].plot(axis_trialX, axis_trialY_sfr_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[1,1].plot(axis_trialX, axis_trialY_sfr_frac, 'r-', label = 'Sinusoidal Fit') 
ax[1,1].legend()
ax[1,1].grid(axis="y", linestyle="--", alpha=0.7)
ax[1,1].text(0.7, 0.7, f"Minimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[1, 1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()
"""
"""
## -45-135 morphology + sSFR plots, elliptical star forming fraction, sprial quiescent fraction
fig, ax = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)

#Galaxy counts
ax[0].errorbar(axis_bin_centres, axis_ef_hist, yerr=axis_ef_err, label="Elliptical Star Forming", color="blue", capsize=2)
ax[0].errorbar(axis_bin_centres, axis_eq_hist, yerr=axis_eq_err, label="Elliptical Quiescent", color="green", capsize=2)
ax[0].errorbar(axis_bin_centres, axis_sf_hist, yerr=axis_sf_err, label="Spiral Star Forming", color="red", capsize=2)
ax[0].errorbar(axis_bin_centres, axis_sq_hist, yerr=axis_sq_err, label="Spiral Quiescent", color="grey", capsize=2)
ax[0].set_ylabel("Number of Galaxies")
ax[0].set_xlim(-45, 135)
ax[0].set_title("Galaxy Angle Distribution: Classification + sSFR")
ax[0].legend()
ax[0].grid(axis="y", linestyle="--", alpha=0.7)

#EF fraction
ax[1].errorbar(axis_bin_centres, axis_ef_fraction, yerr=axis_ef_fraction_err, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
ax[1].set_ylabel("Fraction of Star Forming Ellipticals")
ax[1].set_title("Fraction of ellipticals which are star forming as a function of angle")
ax[1].set_ylim(np.nanmin(axis_ef_fraction) * 0.8, np.nanmax(axis_ef_fraction) * 1.2)
ax[1].set_xlim(-45, 135)
ax[1].plot(axis_trialX, axis_trialY_ef_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[1].plot(axis_trialX, axis_trialY_ef_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {axis_popt_ef_frac[0]:.2f} ± {np.sqrt(axis_pcov_ef_frac[0,0]):.2f})') 
ax[1].legend()
ax[1].grid(axis="y", linestyle="--", alpha=0.7)
#ax[1].text(0.1, 0.5, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax[1].transAxes, fontsize=8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

#SQ fraction
ax[2].errorbar(axis_bin_centres, axis_sq_fraction, yerr=axis_sq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
ax[2].set_xlabel("Angle (degrees)")
ax[2].set_ylabel("Fraction of Quiescent Spirals")
ax[2].set_title("Fraction of spirals which are quiescent as a function of angle")
ax[2].set_ylim(np.nanmin(axis_sq_fraction) * 0.8, np.nanmax(axis_sq_fraction) * 1.2)
ax[2].set_xlim(-45, 135)
ax[2].plot(axis_trialX, axis_trialY_sq_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[2].plot(axis_trialX, axis_trialY_sq_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {axis_popt_sq_frac[0]:.2f} ± {np.sqrt(axis_pcov_sq_frac[0,0]):.2f})') 
ax[2].legend()
ax[2].grid(axis="y", linestyle="--", alpha=0.7)
#ax[2].text(0.1, 0.5, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax[2].transAxes, fontsize=8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()"""

if show_phase_heat == 1:
    x_bins = np.linspace(min(r200_list), max(r200_list), 30)
    y_bins = np.linspace(min(vel_diff_list), max(vel_diff_list), 30)

    heatmap, xedges, yedges = np.histogram2d(
        r200_list, 
        vel_diff_list, 
        bins=(x_bins, y_bins))

    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis',
        aspect='auto')
    cbar = plt.colorbar(label='Density')
    cbar.ax.tick_params(labelsize=14)  # Adjust colorbar tick labels
    cbar.set_label('Density', fontsize=14)  # Adjust colorbar label font size
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xlabel(r"R/R$_{200}$", fontsize=14)
    plt.ylabel(r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=16)
    plt.show()
