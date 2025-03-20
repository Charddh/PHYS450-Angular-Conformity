import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from astropy.io import fits
import seaborn as sns
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, sine_function_2, sine_function_3, cosine_function, calculate_radial_distance, calculate_velocity_distance, horizontal_line, chi_squared, chi2_red, assign_morph, calculate_theta

max_z = 0.125 #Maximum redshift in the sample.
min_lx = 1e43 #Minimum x-ray luminosity for clusters.
bin_size = 45 #Size in degrees of the bins.
sfr_bin_size = 45 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.25 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 1 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 3000 #Maximum physical separation in kpc between BCG and satellite galaxies.
min_phys_sep = 0 #Minimum physical separation in kpc between BCG and satellite galaxies.
max_vel = 1500 #Maximum velocity difference in km/s between BCG and satellite galaxies.
min_vel = 0 #Minimum velocity difference in km/s between BCG and satellite galaxies.

core_radius = 1750 #Radius of the 'core' satellite galaxies in kpc
outer_radius = 3000 #Maximum radius of satellites considered in kpc

phase_bin_size = 1000 #Size in kpc of the bins

continuous = 1
show_morph = 0
show_forming = 0
show_ef = 0
show_sq = 0

show_physical_combo = 0
show_physical_morph = 0
show_physical_forming = 0
show_physical_heat = 0
show_physical_heat_morph = 0
show_physical_heat_forming = 0

show_phase_space = 0
show_phase_space_morph = 0
show_phase_space_forming = 0
show_phase_heat = 0
show_phase_heat_morph = 0
show_phase_heat_morph_form = 1
show_phase_heat_forming = 0

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
bcg_df = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

bcg_df["CLUS_ID"] = bcg_df["CLUS_ID"].astype(str).str.strip().str.replace(" ", "")
bcg_df = bcg_df[(bcg_df['GAL_sdss_i_modSX_C2_PA'] > 0) & (~bcg_df['CLUS_ID'].isin(mergers))]

cluster_df2 = cluster_df[(cluster_df['SCREEN_CLUZSPEC'] < max_z) & (cluster_df['LX0124'] > min_lx)]
cluster_df2.loc[:, "CLUS_ID"] = cluster_df2["CLUS_ID"].astype(str).str.strip().str.replace(" ", "")
#Extract relevant columns from the filtered clusters DataFrame.
cluster_id = cluster_df2['CLUS_ID'].values
cluster_ra = cluster_df2['RA'].values
cluster_dec = cluster_df2['DEC'].values
cluster_z = cluster_df2['SCREEN_CLUZSPEC'].values
#Extract relevant columns from the bcg dataframe.
bcg_df2 = bcg_df[bcg_df['CLUS_ID'].isin(cluster_id)]

reduced_clusters_id = bcg_df2['CLUS_ID'].values
reduced_clusters_ra = bcg_df2['RA_BCG'].values
reduced_clusters_dec = bcg_df2['DEC_BCG'].values
reduced_clusters_z = bcg_df2['CLUZSPEC'].values
reduced_clusters_pa = bcg_df2['GAL_sdss_i_modSX_C2_PA'].values

#Extract relevant columns from the galaxy zoo dataframe.
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

#Calculate the median error in mass for the highest redshift range being considered.
mask = (gz_z > max_z - 0.05) & (gz_z < max_z + 0.05)
filtered_gz_mass = gz_mass[mask]
filtered_gz_mass16 = gz_mass16[mask]
median_mass16 = np.median(filtered_gz_mass16)
median_mass = np.median(filtered_gz_mass)
median_sigma = (median_mass - median_mass16)

#Switch to choose either the debiased or undebiased values for elliptical / spiral probability.
if debiased == 1:
    gz_elliptical = gz_df['P_EL_DEBIASED'].values
    gz_spiral = gz_df['P_CS_DEBIASED'].values
else:
    gz_elliptical = gz_df['P_EL'].values
    gz_spiral = gz_df['P_CS'].values

#Calculate the angular separation and redshift difference.
ra_diff = reduced_clusters_ra[:, None] - gz_ra  #Broadcast the RA difference calculation.
dec_diff = reduced_clusters_dec[:, None] - gz_dec  #Broadcast the Dec difference calculation.
z_diff = np.abs(reduced_clusters_z[:, None] - gz_z)  #Compute absolute redshift difference.

#Compute the angular separation using the Haversine formula and the proper scaling.
angular_separation = np.sqrt((ra_diff ** 2) * (np.cos(np.radians(reduced_clusters_dec[:, None])) ** 2) + dec_diff ** 2)

#Number of degrees corresponding to 1 kpc at the redshift of each cluster.
degrees_per_kpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(reduced_clusters_z[:, None]).value

#Apply the selection criteria (angular separation and redshift difference).
selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (angular_separation > min_phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise) & (z_diff > min_vel / 3e5)

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
df_bcg = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec,'bcg_z': reduced_clusters_z, 'bcg_sdss_pa': reduced_clusters_pa, 'sat_id': reduced_clusters_locals_id, 'sat_ra': reduced_clusters_locals_ra, 'sat_dec': reduced_clusters_locals_dec, 'sat_z': reduced_clusters_locals_z, 'sat_elliptical': reduced_clusters_locals_elliptical, 'sat_spiral': reduced_clusters_locals_spiral, 'sat_mass': reduced_clusters_locals_mass, 'sat_sfr': reduced_clusters_locals_sfr, 'sat_sfr16': reduced_clusters_locals_sfr16, 'sat_sfr84': reduced_clusters_locals_sfr84})
clusters_df = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec, 'bcg_sdss_pa': reduced_clusters_pa})
clusters_df.to_csv('bcg_data.csv', index=False)
df_bcg["bcg_id"] = df_bcg["bcg_id"].str.strip()
angle_df = pd.read_csv('BCGAngleOffset.csv')
angle_df["clus_id"] = angle_df["clus_id"].str.strip()
#print("df_bcg", df_bcg["bcg_id"][~df_bcg["bcg_id"].isin(angle_df["clus_id"])])
#print("Length of df_bcg['bcg_id']:", len(df_bcg["bcg_id"]))

missing_ids = df_bcg["bcg_id"][~df_bcg["bcg_id"].isin(angle_df["clus_id"])]

merged_df = pd.merge(angle_df, df_bcg, left_on='clus_id', right_on='bcg_id', how = 'inner').drop(columns=['bcg_id'])
merged_df['corrected_pa'] = ((((90 - merged_df['spa']) % 360) - merged_df['bcg_sdss_pa']) % 360)

merged_df['theta'] = merged_df.apply(lambda row: calculate_theta(row['bcg_ra'], row['bcg_dec'], row['sat_ra'], row['sat_dec']), axis=1)
merged_df['sat_majoraxis_angle'] = merged_df.apply(lambda row: [((row['corrected_pa'] - theta) % 360) % 180 for theta in row['theta']], axis=1)
merged_df['radial_sep'] = merged_df.apply(lambda row: calculate_radial_distance(row['bcg_ra'], row['bcg_dec'], row['bcg_z'], row['sat_ra'], row['sat_dec']), axis=1)
merged_df['vel_diff'] = merged_df.apply(lambda row: calculate_velocity_distance(row['bcg_z'], row['sat_z']), axis=1)
merged_df['ra_diff'] = merged_df.apply(lambda row: [row['bcg_ra'] - x for x in row['sat_ra']], axis=1)
merged_df['dec_diff'] = merged_df.apply(lambda row: [row['bcg_dec'] - x for x in row['sat_dec']], axis=1)

satellite_df = pd.DataFrame({
    "clus_id": merged_df["clus_id"],
    "bcg_ra": merged_df["bcg_ra"],
    "bcg_dec": merged_df["bcg_dec"],
    "corrected_pa": merged_df["corrected_pa"],
    "sat_ra": merged_df["sat_ra"],
    "sat_dec": merged_df["sat_dec"],
    "theta": merged_df["theta"],
    "sat_majoraxis_angle": merged_df["sat_majoraxis_angle"]})
satellite_df.to_csv('satellite.csv', index=False)

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
ra_diff_list = np.concatenate(merged_df['ra_diff'].values)
dec_diff_list = np.concatenate(merged_df['dec_diff'].values)
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

print("number of satellites in sample", len(sat_majoraxis_list))

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
    "uu"
    for morph, sfr in zip(sat_type_list, sfr_type_list)]

vel_diff   = np.array(vel_diff_list)
radial_sep = np.array(radial_sep_list)
sat_type   = np.array(sat_type_list)
sfr_type = np.array(sfr_type_list)
morph_forming = np.array(morph_forming_list)

core_vel = 0.5 * core_radius
outer_vel = 0.5 * outer_radius

mask_bin1 = ((vel_diff < core_vel) & (radial_sep < core_radius) & (vel_diff < 0.5 * radial_sep))
mask_bin2 = ((vel_diff >= core_vel) & (vel_diff < outer_vel) & (radial_sep >= core_radius) & (radial_sep < outer_radius) & (vel_diff < 0.5 * radial_sep))
mask_bin3 = ((vel_diff > outer_vel) & (radial_sep > outer_radius))

centres = [np.median(radial_sep[mask_bin1]), np.median(radial_sep[mask_bin2]), np.median(radial_sep[mask_bin3])]

bin1_morph = sat_type[mask_bin1]
bin2_morph = sat_type[mask_bin2]
bin3_morph = sat_type[mask_bin3]
morph_fraction_bin1 = np.sum(bin1_morph == 'e') / len(bin1_morph)
morph_x_bin1_err = np.sqrt(morph_fraction_bin1) / len(bin1_morph)
morph_fraction_bin2 = np.sum(bin2_morph == 'e') / len(bin2_morph)
morph_x_bin2_err = np.sqrt(morph_fraction_bin2) / len(bin2_morph)
morph_fraction_bin3 = np.sum(bin3_morph == 'e') / len(bin3_morph)
morph_x_bin3_err = np.sqrt(morph_fraction_bin3) / len(bin3_morph)

bin1_forming = sfr_type[mask_bin1]
bin2_forming = sfr_type[mask_bin2]
bin3_forming = sfr_type[mask_bin3]
forming_fraction_bin1 = np.sum(bin1_forming == 'q') / len(bin1_forming)
forming_x_bin1_err = np.sqrt(forming_fraction_bin1) / len(bin1_forming)
forming_fraction_bin2 = np.sum(bin2_forming == 'q') / len(bin2_forming)
forming_x_bin2_err = np.sqrt(forming_fraction_bin2) / len(bin2_forming)
forming_fraction_bin3 = np.sum(bin3_forming == 'q') / len(bin3_forming)
forming_x_bin3_err = np.sqrt(forming_fraction_bin3) / len(bin3_forming)

bin1_ef = morph_forming[mask_bin1]
bin2_ef = morph_forming[mask_bin2]
bin3_ef = morph_forming[mask_bin3]
ef_fraction_bin1 = np.sum(bin1_ef == 'ef') / len(bin1_ef)
ef_x_bin1_err = np.sqrt(ef_fraction_bin1) / len(bin1_ef)
ef_fraction_bin2 = np.sum(bin2_ef == 'ef') / len(bin2_ef)
ef_x_bin2_err = np.sqrt(ef_fraction_bin2) / len(bin2_ef)
ef_fraction_bin3 = np.sum(bin3_ef == 'ef') / len(bin3_ef)
ef_x_bin3_err = np.sqrt(ef_fraction_bin3) / len(bin3_ef)

bin1_sq = morph_forming[mask_bin1]
bin2_sq = morph_forming[mask_bin2]
bin3_sq = morph_forming[mask_bin3]
sq_fraction_bin1 = np.sum(bin1_sq == 'sq') / len(bin1_sq)
sq_x_bin1_err = np.sqrt(sq_fraction_bin1) / len(bin1_sq)
sq_fraction_bin2 = np.sum(bin2_sq == 'sq') / len(bin2_sq)
sq_x_bin2_err = np.sqrt(sq_fraction_bin2) / len(bin2_sq)
sq_fraction_bin3 = np.sum(bin3_sq == 'sq') / len(bin3_sq)
sq_x_bin3_err = np.sqrt(sq_fraction_bin3) / len(bin3_sq)

data = {"angles": sat_majoraxis_list, "types": sat_type_list, "morph_sfr": morph_forming_list, "sep": radial_sep_list}
df=pd.DataFrame(data)
df["spiral_angles"] = df["angles"].where(df["types"] == "s")
df["elliptical_angles"] = df["angles"].where(df["types"] == "e")
df["unknown_angles"] = df["angles"].where(df["types"] == "u")
spirals = df["spiral_angles"].dropna().reset_index(drop=True)
ellipticals = df["elliptical_angles"].dropna().reset_index(drop=True)
unknowns = df["unknown_angles"].dropna().reset_index(drop=True)
df["ef_sep"] = df["sep"].where(df["morph_sfr"] == "ef")
df["sf_sep"] = df["sep"].where(df["morph_sfr"] == "sf")
df["eq_sep"] = df["sep"].where(df["morph_sfr"] == "eq")
df["sq_sep"] = df["sep"].where(df["morph_sfr"] == "sq")
df["uu_sep"] = df["sep"].where(df["morph_sfr"] == "uu")
phase_ef = df["ef_sep"].dropna().reset_index(drop=True)
phase_sf = df["sf_sep"].dropna().reset_index(drop=True)
phase_eq = df["eq_sep"].dropna().reset_index(drop=True)
phase_sq = df["sq_sep"].dropna().reset_index(drop=True)
phase_uu = df["uu_sep"].dropna().reset_index(drop=True)

df["ef_angles"] = df["angles"].where(df["morph_sfr"] == "ef")
df["sf_angles"] = df["angles"].where(df["morph_sfr"] == "sf")
df["eq_angles"] = df["angles"].where(df["morph_sfr"] == "eq")
df["sq_angles"] = df["angles"].where(df["morph_sfr"] == "sq")
df["uu_angles"] = df["angles"].where(df["morph_sfr"] == "uu")
ef = df["ef_angles"].dropna().reset_index(drop=True)
sf = df["sf_angles"].dropna().reset_index(drop=True)
eq = df["eq_angles"].dropna().reset_index(drop=True)
sq = df["sq_angles"].dropna().reset_index(drop=True)
uu = df["uu_angles"].dropna().reset_index(drop=True)

sfr_data = {"angles": sat_majoraxis_list, "types": sfr_type_list}
sfr_df=pd.DataFrame(sfr_data)
sfr_df["forming_angles"] = sfr_df["angles"].where(sfr_df["types"] == "f")
sfr_df["quiescent_angles"] = sfr_df["angles"].where(sfr_df["types"] == "q")
sfr_df["unknown_angles"] = sfr_df["angles"].where(sfr_df["types"] == "u")
sfr_forming = sfr_df["forming_angles"].dropna().reset_index(drop=True)
sfr_quiescent = sfr_df["quiescent_angles"].dropna().reset_index(drop=True)
sfr_unknowns = sfr_df["unknown_angles"].dropna().reset_index(drop=True)

df_angles = pd.concat([spirals, ellipticals, unknowns], axis=1, keys=["spirals", "ellipticals", "unknowns"])
df.to_csv('angles.csv', index=False)

bins = np.arange(0, 181, bin_size)
bin_centres = (bins[:-1] + bins[1:]) / 2
trialX = np.linspace(0, 180, 1000)
sfr_bins = np.arange(0, 181, sfr_bin_size)

ef_hist, _ = np.histogram(ef, bins=bins)
sf_hist, _ = np.histogram(sf, bins=bins)
eq_hist, _ = np.histogram(eq, bins=bins)
sq_hist, _ = np.histogram(sq, bins=bins)
uu_hist, _ = np.histogram(uu, bins=bins)
ef_err = np.sqrt(ef_hist)
sf_err = np.sqrt(sf_hist)
eq_err = np.sqrt(eq_hist)
sq_err = np.sqrt(sq_hist)
uu_err = np.sqrt(uu_hist)
ef_fraction = np.where(ef_hist + eq_hist + uu_hist > 0, (ef_hist / (ef_hist + eq_hist + uu_hist)), 0)
sq_fraction = np.where(sq_hist + sf_hist + uu_hist> 0, (sq_hist / (sq_hist + sf_hist + uu_hist)), 0)
ef_fraction_err = np.where(ef_hist + eq_hist + uu_hist> 0, np.sqrt(ef_hist) / (ef_hist + eq_hist + uu_hist), np.nan)
sq_fraction_err = np.where(sq_hist + sf_hist + uu_hist> 0, np.sqrt(sq_hist) / (sq_hist + sf_hist + uu_hist), np.nan)
popt_ef_frac, pcov_ef_frac = opt.curve_fit(sine_function_2, bin_centres, ef_fraction, sigma = ef_fraction_err, p0 = [0.03, 0.03], absolute_sigma = True)
popt_ef_frac_line, pcov_ef_frac_line = opt.curve_fit(horizontal_line, bin_centres, ef_fraction, sigma = ef_fraction_err, p0 = [0.05], absolute_sigma = True)
popt_sq_frac, pcov_sq_frac = opt.curve_fit(sine_function_2, bin_centres, sq_fraction, sigma = sq_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
popt_sq_frac_line, pcov_sq_frac_line = opt.curve_fit(horizontal_line, bin_centres, sq_fraction, sigma = sq_fraction_err, absolute_sigma = True)
trialY_ef_frac = sine_function_2(trialX, *popt_ef_frac)
trialY_ef_frac_line = horizontal_line(trialX, *popt_ef_frac_line)
trialY_sq_frac = sine_function_2(trialX, *popt_sq_frac)
trialY_sq_frac_line = horizontal_line(trialX, *popt_sq_frac_line)
chi2_red_ef_frac = chi2_red(bin_centres, ef_fraction, ef_fraction_err, popt_ef_frac, sine_function_2)
chi2_red_ef_frac_line = chi2_red(bin_centres, ef_fraction, ef_fraction_err, popt_ef_frac_line, horizontal_line)
chi2_red_sq_frac = chi2_red(bin_centres, sq_fraction, sq_fraction_err, popt_sq_frac, sine_function_2)
chi2_red_sq_frac_line = chi2_red(bin_centres, sq_fraction, sq_fraction_err, popt_sq_frac_line, horizontal_line)

spiral_hist, _ = np.histogram(spirals, bins=bins)
elliptical_hist, _ = np.histogram(ellipticals, bins=bins)
unknown_hist, _ = np.histogram(unknowns, bins=bins, density=False)
sfr_forming_hist, _ = np.histogram(sfr_forming, bins=bins)
sfr_quiescent_hist, _ = np.histogram(sfr_quiescent, bins=bins)
sfr_unknown_hist, _ = np.histogram(sfr_unknowns, bins=bins)
sfr_bin_counts, sfr_bin_edges = np.histogram(sat_majoraxis_list, sfr_bins)
sfr_binned, _ = np.histogram(sat_majoraxis_list, sfr_bins, weights = sfr_list)
sfr_mean = sfr_binned / sfr_bin_counts
sfr_err_binned, _ = np.histogram(sat_majoraxis_list, sfr_bins, weights = sfr_error)
sfr_error_mean = sfr_err_binned / sfr_bin_counts
sfr_bin_centres = (sfr_bin_edges[:-1] + sfr_bin_edges[1:]) /2
spiral_errors = np.sqrt(spiral_hist)
elliptical_errors = np.sqrt(elliptical_hist)
unknown_errors = np.sqrt(unknown_hist)
sfr_forming_errors = np.sqrt(sfr_forming_hist)
sfr_quiescent_errors = np.sqrt(sfr_quiescent_hist)
sfr_unknown_errors = np.sqrt(sfr_unknown_hist)
fraction = np.where(spiral_hist + elliptical_hist + unknown_hist > 0, (elliptical_hist / (spiral_hist + elliptical_hist + unknown_hist)), 0)
fraction_errors = np.where(elliptical_hist + spiral_hist + unknown_hist > 0, np.sqrt(elliptical_hist) / (elliptical_hist + spiral_hist + unknown_hist), np.nan)
sfr_fraction = np.where(sfr_forming_hist + sfr_quiescent_hist > 0, (sfr_quiescent_hist / (sfr_forming_hist + sfr_quiescent_hist)), 0)
sfr_fraction_errors = np.where(sfr_quiescent_hist + sfr_forming_hist > 0, np.sqrt(sfr_quiescent_hist) / (sfr_quiescent_hist +  sfr_forming_hist), np.nan)
popt_avgsfr, pcov_avgsfr = opt.curve_fit(sine_function, sfr_bin_centres, sfr_mean, sigma = sfr_error_mean, p0 = [1, -11, 0], absolute_sigma = True)
popt_avgsfr_line, pcov_avgsfr_line = opt.curve_fit(horizontal_line, sfr_bin_centres, sfr_mean, sigma = sfr_error_mean, absolute_sigma = True)
popt_sfr, pcov_sfr = opt.curve_fit(sine_function, sat_majoraxis_list, sfr_list, sigma = sfr_error, p0 = [1, -11, 0], absolute_sigma = True)
popt_frac, pcov_frac = opt.curve_fit(sine_function_2, bin_centres, fraction, sigma = fraction_errors, p0 = [0.1, 0.75], absolute_sigma = True)
popt_frac_line, pcov_frac_line = opt.curve_fit(horizontal_line, bin_centres, fraction, sigma = fraction_errors, absolute_sigma = True)
popt_sfr_frac, pcov_sfr_frac = opt.curve_fit(sine_function_2, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, p0 = [0.1, 0], absolute_sigma = True)
popt_sfr_frac_line, pcov_sfr_frac_line = opt.curve_fit(horizontal_line, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, absolute_sigma = True)
trialY_avgsfr = sine_function(trialX, *popt_avgsfr)
trialY_avgsfr_line = horizontal_line(trialX, *popt_avgsfr_line)
trialY_frac = sine_function_2(trialX, *popt_frac)
trialY_frac_line = horizontal_line(trialX, *popt_frac_line)
trialY_sfr = sine_function(trialX, *popt_sfr)
trialY_sfr_frac = sine_function_2(trialX, *popt_sfr_frac)
trialY_sfr_frac_line = horizontal_line(trialX, *popt_sfr_frac_line)
chi2_red_frac = chi2_red(bin_centres, fraction, fraction_errors, popt_frac, sine_function_2)
chi2_red_frac_line = chi2_red(bin_centres, fraction, fraction_errors, popt_frac_line, horizontal_line)
chi2_red_sfr_frac = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac, sine_function_2)
chi2_red_sfr_frac_line = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac_line, horizontal_line)
sfr_chi2_red = chi2_red(sfr_bin_centres, sfr_mean, sfr_error_mean, popt_avgsfr, sine_function)
sfr_chi2_red_line = chi2_red(sfr_bin_centres, sfr_mean, sfr_error_mean, popt_avgsfr_line, horizontal_line)

phase_bins = np.arange(0, phys_sep, phase_bin_size)
phase_bin_centres = (phase_bins[:-1] + phase_bins[1:]) / 2
phase_trialX = np.linspace(0, phys_sep, 1000)

phase_ef_hist, _ = np.histogram(phase_ef, bins=phase_bins)
phase_sf_hist, _ = np.histogram(phase_sf, bins=phase_bins)
phase_eq_hist, _ = np.histogram(phase_eq, bins=phase_bins)
phase_sq_hist, _ = np.histogram(phase_sq, bins=phase_bins)
phase_uu_hist, _ = np.histogram(phase_uu, bins=phase_bins)
phase_ef_err = np.sqrt(phase_ef_hist)
phase_sf_err = np.sqrt(phase_sf_hist)
phase_eq_err = np.sqrt(phase_eq_hist)
phase_sq_err = np.sqrt(phase_sq_hist)
phase_uu_err = np.sqrt(phase_uu_hist)
phase_ef_fraction = np.where(phase_ef_hist + phase_eq_hist + phase_uu_hist > 0, (phase_ef_hist / (phase_ef_hist + phase_eq_hist + phase_uu_hist)), 0)
phase_sq_fraction = np.where(phase_sq_hist + phase_sf_hist + phase_uu_hist> 0, (phase_sq_hist / (phase_sq_hist + phase_sf_hist + phase_uu_hist)), 0)
phase_ef_fraction_err = np.where(phase_ef_hist + phase_eq_hist + phase_uu_hist> 0, np.sqrt(phase_ef_hist) / (phase_ef_hist + phase_eq_hist + phase_uu_hist), np.nan)
phase_sq_fraction_err = np.where(phase_sq_hist + phase_sf_hist + phase_uu_hist> 0, np.sqrt(phase_sq_hist) / (phase_sq_hist + phase_sf_hist + phase_uu_hist), np.nan)

############ GRAPHS ############
if show_physical_combo == 1:
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
        [ra_diff_list[i] for i in range(len(ra_diff_list)) if mask[i]],
        [dec_diff_list[i] for i in range(len(dec_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel('RA', fontsize=12)
    plt.ylabel('Dec', fontsize=12)
    plt.title('RA against Dec\nColoured by Morphological Class', fontsize=14)
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

if show_physical_morph == 1:
    colour_map = {
    'e': '#FF0000',  # Red
    's': '#0000FF',  # Blue
    'u': '#00FF00',  # Green
    }
    plt.figure(figsize=(10, 6), dpi = 150)
    for category in colour_map:
        mask = [morph == category for morph in sat_type_list]
        plt.scatter(
        [ra_diff_list[i] for i in range(len(ra_diff_list)) if mask[i]],
        [dec_diff_list[i] for i in range(len(dec_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel('RA', fontsize=12)
    plt.ylabel('Dec', fontsize=12)
    plt.title('RA against Dec\nColoured by Morphological Class', fontsize=14)
    plt.legend(title='Morphological Class', bbox_to_anchor=(0.85, 0.95), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if show_physical_forming == 1:
    colour_map = {
    'f': '#FF0000',  # Red
    'q': '#0000FF',  # Blue
    'u': '#00FF00',  # Green
    }
    plt.figure(figsize=(10, 6), dpi = 150)
    for category in colour_map:
        mask = [morph == category for morph in sfr_type_list]
        plt.scatter(
        [ra_diff_list[i] for i in range(len(ra_diff_list)) if mask[i]],
        [dec_diff_list[i] for i in range(len(dec_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel('RA', fontsize=12)
    plt.ylabel('Dec', fontsize=12)
    plt.title('RA against Dec\nColoured by Star Forming Status', fontsize=14)
    plt.legend(title='Star Forming Status Class', bbox_to_anchor=(0.85, 0.95), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show() 

if show_physical_heat == 1:
    x_bins = np.linspace(min(ra_diff_list), max(ra_diff_list), 75)
    y_bins = np.linspace(min(dec_diff_list), max(dec_diff_list), 75)

    heatmap, xedges, yedges = np.histogram2d(
        ra_diff_list, 
        dec_diff_list, 
        bins=(x_bins, y_bins))

    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis',  # Choose colormap: 'plasma', 'inferno', etc.
        aspect='auto')
    plt.colorbar(label='Density')
    plt.xlabel('RA', fontsize=12)
    plt.ylabel(r'Dec', fontsize=12)
    plt.title('Heatmap of RA vs. Dec', fontsize=14)
    plt.show()

if show_physical_heat_morph == 1:
    # Create DataFrame
    df = pd.DataFrame({
        'ra_diff' : ra_diff_list,
        'dec_diff': dec_diff_list,
        'morph': morph_forming_list
    })

    # FacetGrid by morphological class
    g = sns.FacetGrid(
        df, 
        col='morph',  # Split by category
        col_order=['ef', 'sf', 'eq', 'sq', 'uu'],  # Ensure consistent order
        col_wrap=3,  # Adjust layout
        height=4, 
        aspect=1.2
    )
    g.map_dataframe(
        sns.histplot, 
        x='ra_diff', 
        y='dec_diff', 
        bins=(25, 25), 
        cmap='viridis', 
        cbar=True
    )
    g.set_axis_labels('RA', r'Dec', fontsize=12)
    g.fig.suptitle('Heatmaps by Morphological Class', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show() 

if show_physical_heat_forming == 1:
    # Create DataFrame
    df = pd.DataFrame({
        'ra_diff': ra_diff_list,
        'dec_diff': dec_diff_list,
        'star forming status': sfr_type_list
    })

    # FacetGrid by star forming status
    g = sns.FacetGrid(
        df, 
        col='star forming status',  # Split by category
        col_order=['q', 'f'],  # Ensure consistent order
        col_wrap=2,  # Adjust layout
        height=4, 
        aspect=1.2
    )
    g.map_dataframe(
        sns.histplot, 
        x='ra_diff', 
        y='dec_diff', 
        bins=(25, 25), 
        cmap='viridis', 
        cbar=True
    )
    g.set_axis_labels(
        'RA', 
        r'Dec', 
        fontsize=16
    )
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f'Star Forming Status = {title}', fontsize=18)
    for ax in g.axes.flat:
        ax.tick_params(labelsize=16)
    g.fig.suptitle('Heatmaps by Star Forming Status', y=0.95, fontsize=20)
    g.fig.subplots_adjust(top=0.9)
    
    plt.show() 

if continuous == 0:
    if show_morph == 1:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
        ax.errorbar(centres, [morph_fraction_bin1, morph_fraction_bin2, morph_fraction_bin3], yerr=[morph_x_bin1_err, morph_x_bin2_err, morph_x_bin3_err], marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
        ax.set_xlabel("Distance [kpc]")
        ax.set_ylabel("Fraction of Ellipticals")
        ax.set_title("Elliptical Fraction as a Function of Radial Distance")
        ax.set_ylim(np.nanmin([morph_fraction_bin1, morph_fraction_bin2, morph_fraction_bin3]) * 0.8, np.nanmax([morph_fraction_bin1, morph_fraction_bin2, morph_fraction_bin3]) * 1.2)
        #ax.plot(trialX, trialY_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_frac[0]:.3f} ± {np.sqrt(pcov_frac[0,0]):.3f})') 
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    if show_forming == 1:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
        ax.errorbar(centres, [forming_fraction_bin1, forming_fraction_bin2, forming_fraction_bin3], yerr=[forming_x_bin1_err, forming_x_bin2_err, forming_x_bin3_err], marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
        ax.set_xlabel("Distance [kpc]")
        ax.set_ylabel("Fraction of Quiescent Galaxies")
        ax.set_title("Quiescent Fraction as a Function of Radial Distance")
        ax.set_ylim(np.nanmin([forming_fraction_bin1, forming_fraction_bin2, forming_fraction_bin3]) * 0.8, np.nanmax([forming_fraction_bin1, forming_fraction_bin2, forming_fraction_bin3]) * 1.2)
        #ax.plot(trialX, trialY_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_frac[0]:.3f} ± {np.sqrt(pcov_frac[0,0]):.3f})') 
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    if show_ef == 1:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
        ax.errorbar(centres, [ef_fraction_bin1, ef_fraction_bin2, ef_fraction_bin3], yerr=[ef_x_bin1_err, ef_x_bin2_err, ef_x_bin3_err], marker='o', linestyle='-', color="purple", label=" Star-Forming Elliptical Fraction", capsize=2)
        ax.set_xlabel("Distance [kpc]")
        ax.set_ylabel("Fraction of Elliptical Galaxies Which Are Star-Forming")
        ax.set_title("Star-Forming Elliptical Fraction as a Function of Radial Distance")
        ax.set_ylim(np.nanmin([ef_fraction_bin1, ef_fraction_bin2, ef_fraction_bin3]) * 0.8, np.nanmax([ef_fraction_bin1, ef_fraction_bin2, ef_fraction_bin3]) * 1.2)
        #ax.plot(trialX, trialY_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_frac[0]:.3f} ± {np.sqrt(pcov_frac[0,0]):.3f})') 
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    if show_sq == 1:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi = 200)
        ax.errorbar(centres, [sq_fraction_bin1, sq_fraction_bin2, sq_fraction_bin3], yerr=[sq_x_bin1_err, sq_x_bin2_err, sq_x_bin3_err], marker='o', linestyle='-', color="purple", label="Quiescent Spiral Fraction", capsize=2)
        ax.set_xlabel("Distance [kpc]")
        ax.set_ylabel("Fraction of Spiral Galaxies Which Are Quiescent")
        ax.set_title("Quiescent Spiral Fraction as a Function of Radial Distance")
        ax.set_ylim(np.nanmin([sq_fraction_bin1, sq_fraction_bin2, sq_fraction_bin3]) * 0.8, np.nanmax([sq_fraction_bin1, sq_fraction_bin2, sq_fraction_bin3]) * 1.2)
        #ax.plot(trialX, trialY_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_frac[0]:.3f} ± {np.sqrt(pcov_frac[0,0]):.3f})') 
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

if continuous == 1:
    if show_ef == 1:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
        ax.errorbar(phase_bin_centres, phase_ef_fraction, yerr=phase_ef_fraction_err, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Fraction of Star Forming Ellipticals")
        ax.set_title("Fraction of ellipticals which are star forming as a function of radial separation")
        ax.set_ylim(np.nanmin(phase_ef_fraction) * 0.8, np.nanmax(phase_ef_fraction) * 1.2)
        #ax.plot(trialX, trialY_ef_frac_line, 'g-', label = 'Horiztontal Line Fit') 
        #ax.plot(trialX, trialY_ef_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_ef_frac[0]:.3f} ± {np.sqrt(pcov_ef_frac[0,0]):.3f})') 
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    if show_sq == 1:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
        ax.errorbar(phase_bin_centres, phase_sq_fraction, yerr=phase_sq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Fraction of Quiescent Spirals")
        ax.set_title("Fraction of spirals which are quiescent as a function of radial separation")
        ax.set_ylim(np.nanmin(phase_sq_fraction) * 0.8, np.nanmax(phase_sq_fraction) * 1.2)
        #ax.plot(trialX, trialY_sq_frac_line, 'g-', label = 'Horiztontal Line Fit') 
        #ax.plot(trialX, trialY_sq_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sq_frac[0]:.3f} ± {np.sqrt(pcov_sq_frac[0,0]):.3f})') 
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
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

if show_phase_space_morph == 1:
    colour_map = {
    'e': '#FF0000',  # Red
    's': '#0000FF',  # Blue
    'u': '#00FF00',  # Green
    }
    plt.figure(figsize=(10, 6), dpi = 150)
    for category in colour_map:
        mask = [morph == category for morph in sat_type_list]
        plt.scatter(
        [radial_sep_list[i] for i in range(len(radial_sep_list)) if mask[i]],
        [vel_diff_list[i] for i in range(len(vel_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel('Radial Separation [kpc]', fontsize=12)
    plt.ylabel(r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    plt.title('Radial Separation vs. Velocity Difference\nColoured by Morphological Class', fontsize=14)
    plt.legend(title='Morphological Class', bbox_to_anchor=(0.85, 0.95), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if show_phase_space_forming == 1:
    colour_map = {
    'f': '#FF0000',  # Red
    'q': '#0000FF',  # Blue
    'u': '#00FF00',  # Green
    }
    plt.figure(figsize=(10, 6), dpi = 150)
    for category in colour_map:
        mask = [morph == category for morph in sfr_type_list]
        plt.scatter(
        [radial_sep_list[i] for i in range(len(radial_sep_list)) if mask[i]],
        [vel_diff_list[i] for i in range(len(vel_diff_list)) if mask[i]],
        c=colour_map[category],
        label=category,
        alpha=0.7,
        edgecolors='w')
    plt.xlabel('Radial Separation [kpc]', fontsize=12)
    plt.ylabel(r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    plt.title('Radial Separation vs. Velocity Difference\nColoured by Star Forming Status', fontsize=14)
    plt.legend(title='Star Forming Status Class', bbox_to_anchor=(0.85, 0.95), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()    

if show_phase_heat == 1:
    x_bins = np.linspace(min(radial_sep_list), max(radial_sep_list), 30)
    y_bins = np.linspace(min(vel_diff_list), max(vel_diff_list), 30)

    heatmap, xedges, yedges = np.histogram2d(
        radial_sep_list, 
        vel_diff_list, 
        bins=(x_bins, y_bins))

    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis',  # Choose colormap: 'plasma', 'inferno', etc.
        aspect='auto')
    plt.colorbar(label='Density')
    plt.xlabel('Radial Separation [kpc]', fontsize=12)
    plt.ylabel(r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    plt.title('Heatmap of Radial Separation vs. Velocity Difference', fontsize=14)
    plt.show()

if show_phase_heat_morph == 1:
    title_mapping = {'e': 'Elliptical', 's': 'Spiral'}
    # Create DataFrame
    df = pd.DataFrame({
        'radial_sep': radial_sep_list,
        'vel_diff': vel_diff_list,
        'morph': sat_type_list
    })
    df = df[df['morph'].isin(['e', 's'])]

    # FacetGrid by morphological class
    g = sns.FacetGrid(
        df, 
        col='morph',  # Split by category
        col_order=['e', 's'],  # Ensure consistent order
        col_wrap=2,  # Adjust layout
        height=4, 
        aspect=1.2
    )
    g.map_dataframe(
        sns.histplot, 
        x='radial_sep', 
        y='vel_diff', 
        bins=(12, 12), 
        cmap='viridis', 
        cbar=True
    )
    g.set_axis_labels('Radial Separation [kpc]', r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    #g.fig.suptitle('Heatmaps by Morphological Class', y=1.02, fontsize=12)

    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f'Morphology = {title_mapping[title]}', fontsize=16)
    for ax in g.axes.flat:
        ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()

if show_phase_heat_morph_form == 1:
    title_mapping = {'ef': 'Star-Forming Elliptical', 'sf': 'Star-Forming Spiral', 'eq': 'Quiescent Elliptical', 'sq': 'Quiescent Spiral'}
    df = pd.DataFrame({
        'radial_sep': radial_sep_list,
        'vel_diff': vel_diff_list,
        'morph': morph_forming_list
    })

    # FacetGrid by morphological class
    g = sns.FacetGrid(
        df, 
        col='morph',  # Split by category
        col_order=['ef', 'sf', 'eq', 'sq'],  # Ensure consistent order
        col_wrap=2,  # Adjust layout
        height=4, 
        aspect=1.2
    )
    g.map_dataframe(
        sns.histplot, 
        x='radial_sep', 
        y='vel_diff', 
        bins=(12, 12), 
        cmap='viridis', 
        cbar=True
    )
    g.set_axis_labels('Radial Separation [kpc]', r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', fontsize=12)
    #g.fig.suptitle('Heatmaps by Morphological Class', y=1.02, fontsize=12)
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f'Morphology = {title_mapping[title]}', fontsize=16)
    for ax in g.axes.flat:
        ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show() 

if show_phase_heat_forming == 1:
    title_mapping = {'q': 'Quiescent', 'f': 'Star-Forming'}
    # Create DataFrame
    df = pd.DataFrame({
        'radial_sep': radial_sep_list,
        'vel_diff': vel_diff_list,
        'star forming status': sfr_type_list
    })

    # FacetGrid by star forming status
    g = sns.FacetGrid(
        df, 
        col='star forming status',  # Split by category
        col_order=['q', 'f'],  # Ensure consistent order
        col_wrap=2,  # Adjust layout
        height=4, 
        aspect=1.2
    )
    g.map_dataframe(
        sns.histplot, 
        x='radial_sep', 
        y='vel_diff', 
        bins=(12, 12), 
        cmap='viridis', 
        cbar=True
    )
    g.set_axis_labels(
        'Radial Separation [kpc]', 
        r'Velocity Difference [$\mathrm{km\,s^{-1}}$]', 
        fontsize=12
    )
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f'Star Forming Status = {title_mapping[title]}', fontsize=16)
    for ax in g.axes.flat:
        ax.tick_params(labelsize=12)
    #g.fig.suptitle('Heatmaps by Star Forming Status', y=0.95, fontsize=20)
    g.fig.subplots_adjust(top=0.9)
    
    plt.show() 