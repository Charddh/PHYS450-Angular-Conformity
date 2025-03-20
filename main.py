import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, sine_function_2, sine_function_3, cosine_function, horizontal_line, chi_squared, chi2_red, assign_morph, calculate_theta

definate_mergers = ['1_9618']
mergers = ['1_9772', '1_1626', '2_3729', '1_5811', '1_1645', '1_9618', '1_4456', '2_19468']
binaries_mergers = ['2_13471', '1_12336', '1_9849', '1_21627', '2_1139', '1_23993', '2_3273', '2_11151', '1_14486', '1_4409', '1_14573', '1_14884', '1_5823', '1_14426', '1_9772', '1_1626', '2_3729', '1_5811', '1_1645', '1_9618', '1_4456', '2_19468']
max_z = 0.125 #Maximum redshift in the sample.
min_lx = 2e43 #Minimum x-ray luminosity for clusters.
bin_size = 60 #Size in degrees of the bins.
axis_bin = 60 #Size in degrees of the axis bins.
sfr_bin_size = 60 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
max_satellite_mass = 11.5 #Maximum satellite galaxy mass.
classification_threshold = 0.5 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.0 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 1 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 1750 #Maximum physical separation in kpc between BCG and satellite galaxies.
min_phys_sep = 0 #Minimum physical separation in kpc between BCG and satellite galaxies.
mergers = ['1_9618', '1_1626', '1_5811', '1_1645']

show_elliptical = 1 #If 1, will show the elliptical fraction plot.
show_quiescent = 1 #If 1, will show the quiescent fraction plot.
show_ef = 1 #If 1, will show the star forming elliptical fraction plot.
show_sq = 1 #If 1, will show the quiescent spiral fraction plot.
show_ssfr = 0

signal_to_noise = 1 #Minimum signal-to-noise ratio for galaxy spectra.
axis_bin = 60

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

if 0 <= phys_sep <= 3000:
    max_vel = -0.43 * phys_sep + 2000
if 3000 < min_phys_sep < 5000:
    max_vel = 500

#Apply the selection criteria (angular separation and redshift difference).
if 0 < min_phys_sep <= 3000:
    min_vel = -0.43 * min_phys_sep + 2000
    selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (angular_separation > min_phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise) & (z_diff > min_vel / 3e5)
elif min_phys_sep == 0:
    selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise) & (gz_mass < max_satellite_mass)
elif 3000 < min_phys_sep < 5000:
    min_vel = 500
    selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (angular_separation > min_phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise) & (z_diff > min_vel / 3e5)    
else:
    raise ValueError("min_phys_sep must be between 0 and 5000")

selected_counts = [np.sum(selected_galaxies_mask[i]) for i in range(len(reduced_clusters_ra))]
print("Sel", sum(selected_counts))
if all(count < 2 for count in selected_counts):
    print("Not enough data points, skipping iteration.")

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
df_bcg = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec,'bcg_z': reduced_clusters_z, 'bcg_sdss_pa': reduced_clusters_pa, 'sat_id': reduced_clusters_locals_id, 'sat_ra': reduced_clusters_locals_ra, 'sat_dec': reduced_clusters_locals_dec, 'sat_elliptical': reduced_clusters_locals_elliptical, 'sat_spiral': reduced_clusters_locals_spiral, 'sat_mass': reduced_clusters_locals_mass, 'sat_sfr': reduced_clusters_locals_sfr, 'sat_sfr16': reduced_clusters_locals_sfr16, 'sat_sfr84': reduced_clusters_locals_sfr84})
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
sat_elliptical_list = np.concatenate(merged_df['sat_elliptical'].values)
sat_spiral_list = np.concatenate(merged_df['sat_spiral'].values)
sat_mass_list = np.concatenate(merged_df['sat_mass'].values)
star_forming_list = np.concatenate(merged_df['star_forming'].values)
quiescent_list = np.concatenate(merged_df['quiescent'].values)
sfr_list = np.concatenate(merged_df['sat_sfr'].values)
sfr16_list = np.concatenate(merged_df['sat_sfr16'].values)
sfr84_list = np.concatenate(merged_df['sat_sfr84'].values)
sfr_error = (sfr84_list - sfr16_list) / 2

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
    "uu"
    for morph, sfr in zip(sat_type_list, sfr_type_list)]

data = {"angles": sat_majoraxis_list, "types": sat_type_list, "morph_sfr": morph_forming_list}
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
df["uu_angles"] = df["angles"].where(df["morph_sfr"] == "uu")
ef = df["ef_angles"].dropna().reset_index(drop=True)
sf = df["sf_angles"].dropna().reset_index(drop=True)
eq = df["eq_angles"].dropna().reset_index(drop=True)
sq = df["sq_angles"].dropna().reset_index(drop=True)
uu = df["uu_angles"].dropna().reset_index(drop=True)

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
sfr_fraction = np.where(sfr_forming_hist + sfr_quiescent_hist > 0, (sfr_quiescent_hist / (sfr_forming_hist + sfr_quiescent_hist)), 0)
sfr_fraction_errors = np.where(sfr_quiescent_hist + sfr_forming_hist > 0, np.sqrt(sfr_quiescent_hist) / (sfr_quiescent_hist +  sfr_forming_hist), np.nan)
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

#axis_popt_avgsfr, axis_pcov_avgsfr = opt.curve_fit(sine_function, axis_bin_centres, axis_sfr_mean, sigma = axis_sfr_error_mean, p0 = [1, -11, 0], absolute_sigma = True)
#axis_popt_avgsfr_line, axis_pcov_avgsfr_line = opt.curve_fit(horizontal_line, axis_bin_centres, axis_sfr_mean, sigma = axis_sfr_error_mean, absolute_sigma = True)
#axis_trialY_avgsfr = sine_function(axis_trialX, *axis_popt_avgsfr)
#axis_trialY_avgsfr_line = horizontal_line(axis_trialX, *axis_popt_avgsfr_line)
#axis_sfr_chi2_red = chi2_red(axis_bin_centres, axis_sfr_mean, axis_sfr_error_mean, axis_popt_avgsfr, sine_function)
#axis_sfr_chi2_red_line = chi2_red(axis_bin_centres, axis_sfr_mean, axis_sfr_error_mean, axis_popt_avgsfr_line, horizontal_line)

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
    ax.errorbar(bin_centres, ef_fraction, yerr=ef_fraction_err, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Fraction of Star Forming Ellipticals")
    #ax.set_title("Fraction of ellipticals which are star forming as a function of angle")
    ax.set_ylim(np.nanmin(ef_fraction) * 0.8, np.nanmax(ef_fraction) * 1.2)
    ax.plot(trialX, trialY_ef_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_ef_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_ef_frac[0]:.3f} ± {np.sqrt(pcov_ef_frac[0,0]):.3f})') 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    # Add a text box near the vertical line
    ax.text(90, np.nanmax(ef_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    #ax[1].text(0.1, 0.2, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax[1].transAxes, fontsize=8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.show()


#SQ fraction
if show_sq == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(bin_centres, sq_fraction, yerr=sq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Fraction of Quiescent Spirals")
    #ax.set_title("Fraction of spirals which are quiescent as a function of angle")
    ax.set_ylim(np.nanmin(sq_fraction) * 0.8, np.nanmax(sq_fraction) * 1.2)
    ax.plot(trialX, trialY_sq_frac_line, 'g-', label = 'Horiztontal Line Fit') 
    ax.plot(trialX, trialY_sq_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sq_frac[0]:.3f} ± {np.sqrt(pcov_sq_frac[0,0]):.3f})') 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=90, color='black', linestyle='dotted', linewidth=1)
    # Add a text box near the vertical line
    ax.text(90, np.nanmax(sq_fraction) * 1.15, 'Minor Axis', ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    #ax[2].text(0.1, 0.2, f"{bin_size}° Bins\nMinimum LX: {min_lx}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", ha="center", va="center", transform=ax[2].transAxes, fontsize=8, fontweight="normal", bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
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
