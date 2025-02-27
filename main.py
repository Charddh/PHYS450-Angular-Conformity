import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, cosine_function, horizontal_line, chi_squared, chi2_red, assign_morph, calculate_theta

max_z = 0.125 #Maximum redshift in the sample.
min_n = 10 #Minimum number of BCG satellite galaxies.
bin_size = 40 #Size in degrees of the bins.
sfr_bin_size = 40 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10 #Minimum satellite galaxy mass.
classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.25 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 0 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 1500 #Maximum physical separation in kpc between BCG and satellite galaxies.
max_vel = 3000 #Maximum velocity difference in km/s between BCG and satellite galaxies.
signal_to_noise = 3 #Minimum signal-to-noise ratio for galaxy spectra.

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
cluster_df2 = cluster_df[(cluster_df['SCREEN_CLUZSPEC'] < max_z) & (cluster_df['SCREEN_NMEMBERS_W'] > min_n)]

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
gz_sfr = gz_df['SPECSFR_TOT_P50'].values
gz_sfr16 = gz_df['SPECSFR_TOT_P16'].values
gz_sfr84 = gz_df['SPECSFR_TOT_P84'].values
gz_s_n = gz_df['SN_MEDIAN'].values

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
degrees_per_mpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(reduced_clusters_z[:, None]).value

#Apply the selection criteria (angular separation and redshift difference).
selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_mpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) * (gz_s_n > signal_to_noise)

print(np.sum(selected_galaxies_mask))

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

#Import the 
angle_df = pd.read_csv('BCGAngleOffset.csv')
angle_df["clus_id"] = angle_df["clus_id"].str.strip()
df_bcg["bcg_id"] = df_bcg["bcg_id"].str.strip()
print("df_bcg", df_bcg["bcg_id"][~df_bcg["bcg_id"].isin(angle_df["clus_id"])])
print("Length of df_bcg['bcg_id']:", len(df_bcg["bcg_id"]))

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

print(f"Number of satellites in sample: {len(sat_majoraxis_list)}")

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
ef_fraction = np.where(ef_hist + eq_hist > 0, (ef_hist / (ef_hist + eq_hist)), 0)
sq_fraction = np.where(sq_hist + sf_hist > 0, (sq_hist / (sq_hist + sf_hist)), 0)
ef_fraction_err = np.where(ef_hist + eq_hist > 0, np.sqrt(ef_hist) / (ef_hist + eq_hist), np.nan)
sq_fraction_err = np.where(sq_hist + sf_hist > 0, np.sqrt(sq_hist) / (sq_hist + sf_hist), np.nan)
popt_ef_frac, pcov_ef_frac = opt.curve_fit(sine_function, bin_centres, ef_fraction, sigma = ef_fraction_err, p0 = [0.1, 0.75, 0], absolute_sigma = True)
popt_ef_frac_line, pcov_ef_frac_line = opt.curve_fit(horizontal_line, bin_centres, ef_fraction, sigma = ef_fraction_err, absolute_sigma = True)
popt_sq_frac, pcov_sq_frac = opt.curve_fit(sine_function, bin_centres, sq_fraction, sigma = sq_fraction_err, p0 = [0.1, 0.75, 0], absolute_sigma = True)
popt_sq_frac_line, pcov_sq_frac_line = opt.curve_fit(horizontal_line, bin_centres, sq_fraction, sigma = sq_fraction_err, absolute_sigma = True)
trialY_ef_frac = sine_function(trialX, *popt_ef_frac)
trialY_ef_frac_line = horizontal_line(trialX, *popt_ef_frac_line)
trialY_sq_frac = sine_function(trialX, *popt_sq_frac)
trialY_sq_frac_line = horizontal_line(trialX, *popt_sq_frac_line)
chi2_red_ef_frac = chi2_red(bin_centres, ef_fraction, ef_fraction_err, popt_ef_frac, sine_function)
chi2_red_ef_frac_line = chi2_red(bin_centres, ef_fraction, ef_fraction_err, popt_ef_frac_line, horizontal_line)
chi2_red_sq_frac = chi2_red(bin_centres, sq_fraction, sq_fraction_err, popt_sq_frac, sine_function)
chi2_red_sq_frac_line = chi2_red(bin_centres, sq_fraction, sq_fraction_err, popt_sq_frac_line, horizontal_line)

spiral_hist, _ = np.histogram(spirals, bins=bins)
elliptical_hist, _ = np.histogram(ellipticals, bins=bins)
unknown_hist, _ = np.histogram(unknowns, bins=bins)

sfr_forming_hist, _ = np.histogram(sfr_forming, bins=bins)
sfr_quiescent_hist, _ = np.histogram(sfr_quiescent, bins=bins)

sfr_unknown_hist, _ = np.histogram(sfr_unknowns, bins=bins)
sfr_bin_counts, sfr_bin_edges = np.histogram(sat_majoraxis_list, sfr_bins)
sfr_binned, _ = np.histogram(sat_majoraxis_list, sfr_bins, weights = sfr_list)
sfr_mean = sfr_binned / sfr_bin_counts
sfr_err_binned, _ = np.histogram(sat_majoraxis_list, sfr_bins, weights = sfr_error)
sfr_error_mean = sfr_err_binned / sfr_bin_counts
sfr_bin_centres = (sfr_bin_edges[:-1] + sfr_bin_edges[1:]) /2

# Poisson errors for counts
spiral_errors = np.sqrt(spiral_hist)
elliptical_errors = np.sqrt(elliptical_hist)
unknown_errors = np.sqrt(unknown_hist)
sfr_forming_errors = np.sqrt(sfr_forming_hist)
sfr_quiescent_errors = np.sqrt(sfr_quiescent_hist)
sfr_unknown_errors = np.sqrt(sfr_unknown_hist)

# Compute fraction and errors
fraction = np.where(spiral_hist + elliptical_hist > 0, (elliptical_hist / (spiral_hist + elliptical_hist)), 0)
fraction_errors = np.where(elliptical_hist + spiral_hist > 0, np.sqrt(elliptical_hist) / (elliptical_hist + spiral_hist), np.nan)
sfr_fraction = np.where(sfr_forming_hist + sfr_quiescent_hist > 0, (sfr_quiescent_hist / (sfr_forming_hist + sfr_quiescent_hist)), 0)
sfr_fraction_errors = np.where(sfr_quiescent_hist + sfr_forming_hist > 0, np.sqrt(sfr_quiescent_hist) / (sfr_quiescent_hist +  sfr_forming_hist), np.nan)

popt_avgsfr, pcov_avgsfr = opt.curve_fit(sine_function, sfr_bin_centres, sfr_mean, sigma = sfr_error_mean, p0 = [1, -11, 0], absolute_sigma = True)
popt_sfr, pcov_sfr = opt.curve_fit(sine_function, sat_majoraxis_list, sfr_list, sigma = sfr_error, p0 = [1, -11, 0], absolute_sigma = True)
popt_frac, pcov_frac = opt.curve_fit(sine_function, bin_centres, fraction, sigma = fraction_errors, p0 = [0.1, 0.75, 0], absolute_sigma = True)
popt_frac_cos, pcov_frac_cos = opt.curve_fit(cosine_function, bin_centres, fraction, sigma = fraction_errors, p0 = [0.05, 0.85, 0], absolute_sigma = True)
popt_frac_line, pcov_frac_line = opt.curve_fit(horizontal_line, bin_centres, fraction, sigma = fraction_errors, absolute_sigma = True)
popt_sfr_frac, pcov_sfr_frac = opt.curve_fit(sine_function, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, p0 = [0.1, 0, 0], absolute_sigma = True)
popt_sfr_frac_cos, pcov_sfr_frac_cos = opt.curve_fit(cosine_function, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, p0 = [0.05, 0.85, 0], absolute_sigma = True)
popt_sfr_frac_line, pcov_sfr_frac_line = opt.curve_fit(horizontal_line, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, absolute_sigma = True)

trialY_avgsfr = sine_function(trialX, *popt_avgsfr)
trialY_frac = sine_function(trialX, *popt_frac)
trialY_frac_cos = cosine_function(trialX, *popt_frac)
trialY_frac_line = horizontal_line(trialX, *popt_frac_line)
trialY_sfr = sine_function(trialX, *popt_sfr)
trialY_sfr_frac = sine_function(trialX, *popt_sfr_frac)
trialY_sfr_frac_cos = cosine_function(trialX, *popt_sfr_frac_cos)
trialY_sfr_frac_line = horizontal_line(trialX, *popt_sfr_frac_line)

chi2_red_frac = chi2_red(bin_centres, fraction, fraction_errors, popt_frac, sine_function)
chi2_red_frac_cos = chi2_red(bin_centres, fraction, fraction_errors, popt_frac_cos, cosine_function)
chi2_red_frac_line = chi2_red(bin_centres, fraction, fraction_errors, popt_frac_line, horizontal_line)
chi2_red_sfr_frac = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac, sine_function)
chi2_red_sfr_frac_cos = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac_cos, cosine_function)
chi2_red_sfr_frac_line = chi2_red(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr_frac_line, horizontal_line)

print("Elliptical fraction")
print(f"Sinusoid reduced chi squared: {chi2_red_frac:.3f}")
print(f"y = ({popt_frac[0]:.2f} ± {np.sqrt(pcov_frac[0,0]):.2f})sin(x + ({popt_frac[2]:.2f} ± {np.sqrt(pcov_frac[2,2]):.2f})) + ({popt_frac[1]:.2f} ± {np.sqrt(pcov_frac[1,1]):.2f})")
print(f"Cosine reduced chi squared: {chi2_red_frac_cos:.3f}")
print(f"y = ({popt_frac_cos[0]:.2f} ± {np.sqrt(pcov_frac_cos[0,0]):.2f})cos(x + ({popt_frac_cos[2]:.2f} ± {np.sqrt(pcov_frac_cos[2,2]):.2f})) + ({popt_frac_cos[1]:.2f} ± {np.sqrt(pcov_frac_cos[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {chi2_red_frac_line:.3f}")
print(f"y = {popt_frac_line[0]:.2f} ± {np.sqrt(pcov_frac_line[0,0]):.2f}")
print("Quiescent fraction")
print(f"Sinusoid reduced chi squared: {chi2_red_sfr_frac:.3f}")
print(f"y = ({popt_sfr_frac[0]:.2f} ± {np.sqrt(pcov_sfr_frac[0,0]):.2f})sin(x + ({popt_sfr_frac[2]:.2f} ± {np.sqrt(pcov_sfr_frac[2,2]):.2f})) + ({popt_sfr_frac[1]:.2f} ± {np.sqrt(pcov_sfr_frac[1,1]):.2f})")
print(f"Cosine reduced chi squared: {chi2_red_sfr_frac_cos:.3f}")
print(f"y = ({popt_sfr_frac_cos[0]:.2f} ± {np.sqrt(pcov_sfr_frac_cos[0,0]):.2f})cos(x + ({popt_sfr_frac_cos[2]:.2f} ± {np.sqrt(pcov_sfr_frac_cos[2,2]):.2f})) + ({popt_sfr_frac_cos[1]:.2f} ± {np.sqrt(pcov_sfr_frac_cos[1,1]):.2f})")
print(f"Horizontal line reduced chi squared: {chi2_red_sfr_frac_line:.3f}")
print(f"y = {popt_sfr_frac_line[0]:.2f} ± {np.sqrt(pcov_sfr_frac_line[0,0]):.2f}")
"""
# Create figure and subplots
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
ax[0,1].plot(trialX, trialY_frac_cos, 'b-', label = 'Cosine Fit') 
ax[0,1].legend()
ax[0,1].grid(axis="y", linestyle="--", alpha=0.7)
ax[0,1].text(0.7, 0.7, f"{bin_size}° Bins\nMinimum Members: {min_n}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
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
ax[1,1].text(0.7, 0.7, f"{bin_size}° Bins\nMinimum Members: {min_n}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[1, 1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()"""

"""fig, ax = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)

ax[0].errorbar(sat_majoraxis_list, sfr_list, yerr=sfr_error, label="SFR", color="blue", ecolor='red', capsize=2, marker = 'o', markersize = 3, linewidth = 0.5, linestyle = 'None')
ax[0].plot(trialX, trialY_sfr, 'g-', label = 'Sinusoidal Fit') 
ax[0].set_ylabel("SSFR")
ax[0].set_title("Separate Galaxy Angle Distribution")
ax[0].legend()
ax[0].grid(axis="y", linestyle="--", alpha=0.7)

ax[1].errorbar(sfr_bin_centres, sfr_mean, yerr=sfr_error_mean, label="SFR", color="blue", ecolor='red', capsize=2, marker = 'o', markersize = 3, linewidth = 0.5, linestyle = 'None')
ax[1].plot(trialX, trialY_avgsfr, 'g-', label = 'Sinusoidal Fit') 
ax[1].set_xlabel("Angle (degrees)")
ax[1].set_ylabel("SSFR")
ax[1].set_title("Binned Galaxy Angle Distribution")
ax[1].legend()
ax[1].grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()
"""

# Create figure and subplots
fig, ax = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)

#Galaxy counts
ax[0].errorbar(bin_centres, ef_hist, yerr=ef_err, label="Elliptical Star Forming", color="blue", capsize=2)
ax[0].errorbar(bin_centres, eq_hist, yerr=eq_err, label="Elliptical Quiescent", color="green", capsize=2)
ax[0].errorbar(bin_centres, sf_hist, yerr=sf_err, label="Spiral Star Forming", color="red", capsize=2)
ax[0].errorbar(bin_centres, sq_hist, yerr=sq_err, label="Spiral Quiescent", color="grey", capsize=2)
ax[0].set_ylabel("Number of Galaxies")
ax[0].set_title("Galaxy Angle Distribution: Classification + sSFR")
ax[0].legend()
ax[0].grid(axis="y", linestyle="--", alpha=0.7)

#EF fraction
ax[1].errorbar(bin_centres, ef_fraction, yerr=ef_fraction_err, marker='o', linestyle='-', color="purple", label="Elliptical Fraction", capsize=2)
ax[1].set_ylabel("Fraction of Star Forming Ellipticals")
ax[1].set_title("Fraction of ellipticals which are star forming as a function of angle")
ax[1].set_ylim(np.nanmin(ef_fraction) * 0.8, np.nanmax(ef_fraction) * 1.2)
ax[1].plot(trialX, trialY_ef_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[1].plot(trialX, trialY_ef_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_ef_frac[0]:.2f} ± {np.sqrt(pcov_ef_frac[0,0]):.2f})') 
ax[1].legend()
ax[1].grid(axis="y", linestyle="--", alpha=0.7)
ax[1].text(0.1, 0.2, f"{bin_size}° Bins\nMinimum Members: {min_n}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[1].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

#SQ fraction
ax[2].errorbar(bin_centres, sq_fraction, yerr=sq_fraction_err, marker='o', linestyle='-', color="purple", label="Quiescent Fraction", capsize=2)
ax[2].set_xlabel("Angle (degrees)")
ax[2].set_ylabel("Fraction of Quiescent Spirals")
ax[2].set_title("Fraction of spirals which are quiescent as a function of angle")
ax[2].set_ylim(np.nanmin(sq_fraction) * 0.8, np.nanmax(sq_fraction) * 1.2)
ax[2].plot(trialX, trialY_sq_frac_line, 'g-', label = 'Horiztontal Line Fit') 
ax[2].plot(trialX, trialY_sq_frac, 'r-', label = f'Sinusoidal Fit (amplitude = {popt_sq_frac[0]:.2f} ± {np.sqrt(pcov_sq_frac[0,0]):.2f})') 
ax[2].legend()
ax[2].grid(axis="y", linestyle="--", alpha=0.7)
ax[2].text(0.1, 0.2, f"{bin_size}° Bins\nMinimum Members: {min_n}\nz < {max_z}\nMinimum Satellite Mass: {min_satellite_mass}\nClassification threshold: {classification_threshold}\nDebiased: {debiased}\nPhysical Separation: {phys_sep} kpc\nMax Velocity Difference: {max_vel} km/s\nSignal-to-Noise: {signal_to_noise}", 
           ha="center", va="center", transform=ax[2].transAxes, 
           fontsize=8, fontweight="normal", 
           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()