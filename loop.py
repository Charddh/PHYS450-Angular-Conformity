import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, sine_function_2, chi_squared, assign_morph, calculate_theta, calculate_r200, calculate_radial_distance, calculate_velocity_distance
import json

max_z = 0.12 #Maximum redshift in the sample.
min_lx = 1e43 #Minimum x-ray luminosity for clusters.
bin_size = 45 #Size in degrees of the bins.
sfr_bin_size = 45 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
max_satellite_mass = 11.5
classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.25 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 1 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 1000 #Maximum physical separation in kpc between BCG and satellite galaxies.
max_vel = 3000 #Maximum velocity difference in km/s between BCG and satellite galaxies.
signal_to_noise = 1 #Minimum signal-to-noise ratio for galaxy spectra.
axis_bin = 60
mergers = ['1_9618', '1_1626', '1_5811', '1_1645']

max_zs = [0.1, 0.105, 0.110, 0.115, 0.120]
classification_thresholds = [0.5, 1]
debiaseds = [0, 1]
bin_sizes = [45, 60]
max_r200s = [1, 1.5, 2, 2.5, 3, 3.5, 4]
min_r200s = [0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
sfr_thresholds = [-11.0]
min_lxs = [2e43, 3e43, 3.5e43, 3.8e43, 4e43, 4.5e43]
min_vel_diffs = [0, 10, 20, 50, 100]
min_sample = 2

total_iterations = (len(classification_thresholds) * len(debiaseds) * len(bin_sizes) * len(max_r200s) * len(min_r200s) * len(min_lxs) * len(sfr_thresholds) * len(max_zs) * len(min_vel_diffs))

high_sigma = 2.9

best_ef_params = {}
best_eq_params = {}
best_sf_params = {}
best_sq_params = {}
best_q_params = {}
best_f_params = {}
best_e_params = {}
best_s_params = {}
best_fs_params = {}
best_qs_params = {}
best_combo_params = {}

best_ef_sigma = 0
best_eq_sigma = 0
best_sf_sigma = 0
best_sq_sigma = 0
best_q_sigma = 0
best_f_sigma = 0
best_e_sigma = 0
best_s_sigma = 0
best_fs_sigma = 0
best_qs_sigma = 0
best_combo_sigma = 0

high_ef_sigma_params = []
high_eq_sigma_params = []
high_sf_sigma_params = []
high_sq_sigma_params = []
high_q_sigma_params = []
high_f_sigma_params = []
high_e_sigma_params = []
high_s_sigma_params = []
high_fs_sigma_params = []
high_qs_sigma_params = []
high_combo_params = []

cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data
cluster_df = pd.DataFrame({
    name: cluster_data[name].byteswap().newbyteorder()
    for name in cluster_data.dtype.names}) 
gz_df = pd.DataFrame(fits.open("GZDR1SFRMASS.fits")[1].data)
bcg_df = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)
bcg_df["CLUS_ID"] = bcg_df["CLUS_ID"].astype(str).str.strip().str.replace(" ", "")
bcg_df = bcg_df[(bcg_df['GAL_sdss_i_modSX_C2_PA'] > 0) & (~bcg_df['CLUS_ID'].isin(mergers))]
angle_df = pd.read_csv('BCGAngleOffset.csv')
angle_df["clus_id"] = angle_df["clus_id"].str.strip()
trialX = np.linspace(0, 180, 1000)

i = 0
errors = 0
for classification_threshold in classification_thresholds:
    for debiased in debiaseds:
        for bin_size in bin_sizes:
            for min_r200 in min_r200s:
                for max_r200 in max_r200s:
                    for min_lx in min_lxs:
                        for sfr_threshold in sfr_thresholds:
                            for max_z in max_zs:
                                for min_vel_diff in min_vel_diffs:
                                    i += 1
                                    print(f"Iteration: {i} / {total_iterations}")
                                    if (min_r200 > max_r200):
                                        print("min phys sep > max phys sep, skipping iteration")
                                        continue              
                                    cluster_df2 = cluster_df[(cluster_df['SCREEN_CLUZSPEC'] < max_z) & (cluster_df['LX0124'] > min_lx)]
                                    cluster_df2.loc[:, "CLUS_ID"] = cluster_df2["CLUS_ID"].astype(str).str.strip().str.replace(" ", "")
                                    #Extract relevant columns from the filtered clusters DataFrame.
                                    cluster_id = cluster_df2['CLUS_ID'].values
                                    bcg_df2 = bcg_df.merge(
                                        cluster_df2[['CLUS_ID', 'LX0124']],  # Columns to transfer
                                        on='CLUS_ID',                        # Merge key
                                        how='inner',                         # Keep only matching clusters
                                        suffixes=('', '_cluster')            # Avoid column name conflicts
                                    )
                                    reduced_clusters_id = bcg_df2['CLUS_ID'].values
                                    reduced_clusters_ra = bcg_df2['RA_BCG'].values
                                    reduced_clusters_dec = bcg_df2['DEC_BCG'].values
                                    reduced_clusters_z = bcg_df2['CLUZSPEC'].values
                                    reduced_clusters_pa = bcg_df2['GAL_sdss_i_modSX_C2_PA'].values
                                    reduced_clusters_lx = bcg_df2['LX0124'].values
                                    reduced_clusters_r200 = np.array([calculate_r200(lx) for lx in reduced_clusters_lx])

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

                                    if debiased == 1:
                                        gz_elliptical = gz_df['P_EL_DEBIASED'].values
                                        gz_spiral = gz_df['P_CS_DEBIASED'].values
                                    else:
                                        gz_elliptical = gz_df['P_EL'].values
                                        gz_spiral = gz_df['P_CS'].values

                                    ra_diff = reduced_clusters_ra[:, None] - gz_ra  #Broadcast the RA difference calculation.
                                    dec_diff = reduced_clusters_dec[:, None] - gz_dec  #Broadcast the Dec difference calculation.
                                    z_diff = np.abs(reduced_clusters_z[:, None] - gz_z)  #Compute absolute redshift difference.

                                    angular_separation = np.sqrt((ra_diff ** 2) * (np.cos(np.radians(reduced_clusters_dec[:, None])) ** 2) + dec_diff ** 2)

                                    degrees_per_kpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(reduced_clusters_z[:, None]).value
                                    phys_sep_galaxy = angular_separation / degrees_per_kpc
                                    r200_sep_galaxy = phys_sep_galaxy / reduced_clusters_r200[:, None]

                                    max_vel = np.where(
                                        (phys_sep_galaxy >= 0) & (phys_sep_galaxy <= 3000),
                                        -0.43 * phys_sep_galaxy + 2000,
                                        np.where(
                                            (phys_sep_galaxy > 3000) & (phys_sep_galaxy < 5000),
                                            500,
                                            np.nan))

                                    selected_galaxies_mask = (
                                        (r200_sep_galaxy <= max_r200) & 
                                        (r200_sep_galaxy > min_r200) &
                                        (z_diff < max_vel / 3e5) &
                                        (gz_mass > min_satellite_mass) &
                                        (gz_mass < max_satellite_mass) &
                                        (gz_s_n > signal_to_noise) &
                                        (z_diff > min_vel_diff / 3e5))

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
                                    reduced_clusters_locals_r200 = [gz_id[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]

                                    df_bcg = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec,'bcg_z': reduced_clusters_z, 'bcg_sdss_pa': reduced_clusters_pa, 'bcg_r200': reduced_clusters_r200, 'sat_id': reduced_clusters_locals_id, 'sat_ra': reduced_clusters_locals_ra, 'sat_dec': reduced_clusters_locals_dec, 'sat_z': reduced_clusters_locals_z, 'sat_elliptical': reduced_clusters_locals_elliptical, 'sat_spiral': reduced_clusters_locals_spiral, 'sat_mass': reduced_clusters_locals_mass, 'sat_sfr': reduced_clusters_locals_sfr, 'sat_sfr16': reduced_clusters_locals_sfr16, 'sat_sfr84': reduced_clusters_locals_sfr84})
                                    df_bcg["bcg_id"] = df_bcg["bcg_id"].str.strip()

                                    missing_ids = df_bcg["bcg_id"][~df_bcg["bcg_id"].isin(angle_df["clus_id"])]

                                    merged_df = pd.merge(angle_df, df_bcg, left_on='clus_id', right_on='bcg_id', how = 'inner').drop(columns=['bcg_id'])
                                    merged_df['corrected_pa'] = ((((90 - merged_df['spa']) % 360) - merged_df['bcg_sdss_pa']) % 360)
                                    merged_df['theta'] = merged_df.apply(lambda row: calculate_theta(row['bcg_ra'], row['bcg_dec'], row['sat_ra'], row['sat_dec']), axis=1)                                     
                                    if merged_df['theta'].dtypes == 'float64':
                                        print("Error: Skipping itteration")
                                        errors += 1
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
                                    star_forming_list = np.concatenate(merged_df['star_forming'].values)
                                    quiescent_list = np.concatenate(merged_df['quiescent'].values)
                                    r200_list = np.concatenate(merged_df['r/r200'].values)

                                    if len(sat_majoraxis_list) < 2:
                                        print("Not enough data points, skipping iteration.")
                                        continue  # Skip to the next loop iteration

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

                                    bins = np.arange(0, 181, bin_size)
                                    bin_centres = (bins[:-1] + bins[1:]) / 2

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
                                    ef_fraction_err = np.where(total_ellipticals > 0, np.sqrt((ef_hist / total_ellipticals) * (1 - (ef_hist / total_ellipticals)) / total_ellipticals), np.nan)
                                    eq_fraction_err = np.where(total_ellipticals > 0, np.sqrt((eq_hist / total_ellipticals) * (1 - (eq_hist / total_ellipticals)) / total_ellipticals), np.nan)
                                    sq_fraction_err = np.where(total_spirals > 0, np.sqrt((sq_hist / total_spirals) * (1 - (sq_hist / total_spirals)) / total_spirals), np.nan)
                                    sf_fraction_err = np.where(total_spirals > 0, np.sqrt((sf_hist / total_spirals) * (1 - (sf_hist / total_spirals)) / total_spirals), np.nan)
                                    q_fraction_err = np.where(total_gals > 0, np.sqrt(q_fraction * (1 - q_fraction) / total_gals), np.nan) 
                                    f_fraction_err = np.where(total_gals > 0, np.sqrt(f_fraction * (1 - f_fraction) / total_gals), np.nan) 
                                    e_fraction_err = np.where(total_morph > 0, np.sqrt(e_fraction * (1 - e_fraction) / total_morph), np.nan)
                                    s_fraction_err = np.where(total_morph > 0, np.sqrt(s_fraction * (1 - s_fraction) / total_morph), np.nan)
                                    fs_fraction_err = np.where(total_forming > 0, np.sqrt(fs_fraction * (1 - fs_fraction) / total_forming), np.nan)
                                    qs_fraction_err = np.where(total_quiescent > 0, np.sqrt(qs_fraction * (1 - qs_fraction) / total_quiescent), np.nan)
                                    popt_ef_frac, pcov_ef_frac = opt.curve_fit(sine_function_2, bin_centres, ef_fraction, sigma = ef_fraction_err, p0 = [0.03, 0.03], absolute_sigma = True)
                                    popt_eq_frac, pcov_eq_frac = opt.curve_fit(sine_function_2, bin_centres, eq_fraction, sigma = eq_fraction_err, p0 = [0.03, 0.03], absolute_sigma = True)
                                    popt_sq_frac, pcov_sq_frac = opt.curve_fit(sine_function_2, bin_centres, sq_fraction, sigma = sq_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_sf_frac, pcov_sf_frac = opt.curve_fit(sine_function_2, bin_centres, sf_fraction, sigma = sf_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_q_frac, pcov_q_frac = opt.curve_fit(sine_function_2, bin_centres, q_fraction, sigma = q_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_f_frac, pcov_f_frac = opt.curve_fit(sine_function_2, bin_centres, f_fraction, sigma = f_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_e_frac, pcov_e_frac = opt.curve_fit(sine_function_2, bin_centres, e_fraction, sigma = e_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_s_frac, pcov_s_frac = opt.curve_fit(sine_function_2, bin_centres, s_fraction, sigma = s_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_fs_frac, pcov_fs_frac = opt.curve_fit(sine_function_2, bin_centres, fs_fraction, sigma = fs_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    popt_qs_frac, pcov_qs_frac = opt.curve_fit(sine_function_2, bin_centres, qs_fraction, sigma = qs_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                                    trialY_ef_frac = sine_function_2(trialX, *popt_ef_frac)
                                    trialY_eq_frac = sine_function_2(trialX, *popt_eq_frac)
                                    trialY_sq_frac = sine_function_2(trialX, *popt_sq_frac)
                                    trialY_sf_frac = sine_function_2(trialX, *popt_sf_frac)
                                    trialY_q_frac = sine_function_2(trialX, *popt_q_frac)
                                    trialY_f_frac = sine_function_2(trialX, *popt_f_frac)
                                    trialY_e_frac = sine_function_2(trialX, *popt_e_frac)
                                    trialY_s_frac = sine_function_2(trialX, *popt_s_frac)
                                    trialY_fs_frac = sine_function_2(trialX, *popt_fs_frac)
                                    trialY_qs_frac = sine_function_2(trialX, *popt_qs_frac)

                                    if ((abs(popt_eq_frac[0]) / np.sqrt(pcov_eq_frac[0,0])) + (abs(popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) + (abs(popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])) + (abs(popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])) + (abs(popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0]))) > best_combo_sigma:
                                        best_combo_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_combo_sigma = ((abs(popt_eq_frac[0]) / np.sqrt(pcov_ef_frac[0,0])) + (abs(popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) + (abs(popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])) + (abs(popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])) + (abs(popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0])))
                                    if ((abs(popt_eq_frac[0]) / np.sqrt(pcov_eq_frac[0,0])) + (abs(popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) + (abs(popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])) + (abs(popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])) + (abs(popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0]))) > 2 * high_sigma:
                                        high_combo_params.append({"Sigma": ((abs(popt_eq_frac[0]) / np.sqrt(pcov_eq_frac[0,0])) + (abs(popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) + (abs(popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])) + (abs(popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])) + (abs(popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0]))), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})                   

                                    if abs((popt_ef_frac[0]) / np.sqrt(pcov_ef_frac[0,0])) > abs(best_ef_sigma):
                                        best_ef_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_ef_sigma = popt_ef_frac[0] / np.sqrt(pcov_ef_frac[0,0])
                                    if abs((popt_ef_frac[0]) / np.sqrt(pcov_ef_frac[0,0])) > high_sigma:
                                        high_ef_sigma_params.append({"Sigma": abs((popt_ef_frac[0]) / np.sqrt(pcov_ef_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})
                                    
                                    if abs((popt_eq_frac[0]) / np.sqrt(pcov_eq_frac[0,0])) > abs(best_eq_sigma):
                                        best_eq_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_eq_sigma = popt_eq_frac[0] / np.sqrt(pcov_eq_frac[0,0])
                                    if abs((popt_eq_frac[0]) / np.sqrt(pcov_eq_frac[0,0])) > high_sigma:
                                        high_eq_sigma_params.append({"Sigma": abs((popt_eq_frac[0]) / np.sqrt(pcov_eq_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}) 
                                    
                                    if abs((popt_sf_frac[0]) / np.sqrt(pcov_sf_frac[0,0])) > abs(best_sf_sigma):
                                        best_sf_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_sf_sigma = popt_sf_frac[0] / np.sqrt(pcov_sf_frac[0,0])
                                    if abs((popt_sf_frac[0]) / np.sqrt(pcov_sf_frac[0,0])) > high_sigma:
                                        high_sf_sigma_params.append({"Sigma": abs((popt_sf_frac[0]) / np.sqrt(pcov_sf_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) > abs(best_sq_sigma):
                                        best_sq_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_sq_sigma = popt_sq_frac[0] / np.sqrt(pcov_sq_frac[0,0])
                                    if abs((popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) > high_sigma:
                                        high_sq_sigma_params.append({"Sigma": abs((popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])) > abs(best_q_sigma):
                                        best_q_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_q_sigma = popt_q_frac[0] / np.sqrt(pcov_q_frac[0,0])
                                    if abs((popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])) > high_sigma:
                                        high_q_sigma_params.append({"Sigma": abs((popt_q_frac[0]) / np.sqrt(pcov_q_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_f_frac[0]) / np.sqrt(pcov_f_frac[0,0])) > abs(best_f_sigma):
                                        best_f_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_f_sigma = popt_f_frac[0] / np.sqrt(pcov_f_frac[0,0])
                                    if abs((popt_f_frac[0]) / np.sqrt(pcov_f_frac[0,0])) > high_sigma:
                                        high_f_sigma_params.append({"Sigma": abs((popt_f_frac[0]) / np.sqrt(pcov_f_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_e_frac[0]) / np.sqrt(pcov_e_frac[0,0])) > abs(best_e_sigma):
                                        best_e_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_e_sigma = popt_e_frac[0] / np.sqrt(pcov_e_frac[0,0])
                                    if abs((popt_e_frac[0]) / np.sqrt(pcov_e_frac[0,0])) > high_sigma:
                                        high_e_sigma_params.append({"Sigma": abs((popt_e_frac[0]) / np.sqrt(pcov_e_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_s_frac[0]) / np.sqrt(pcov_s_frac[0,0])) > abs(best_s_sigma):
                                        best_s_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_s_sigma = popt_s_frac[0] / np.sqrt(pcov_s_frac[0,0])
                                    if abs((popt_s_frac[0]) / np.sqrt(pcov_s_frac[0,0])) > high_sigma:
                                        high_s_sigma_params.append({"Sigma": abs((popt_s_frac[0]) / np.sqrt(pcov_s_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])) > abs(best_qs_sigma):
                                        best_qs_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_qs_sigma = popt_qs_frac[0] / np.sqrt(pcov_qs_frac[0,0])
                                    if abs((popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])) > high_sigma:
                                        high_qs_sigma_params.append({"Sigma": abs((popt_qs_frac[0]) / np.sqrt(pcov_qs_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})

                                    if abs((popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0])) > abs(best_fs_sigma):
                                        best_fs_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff}
                                        best_fs_sigma = popt_fs_frac[0] / np.sqrt(pcov_fs_frac[0,0])
                                    if abs((popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0])) > high_sigma:
                                        high_fs_sigma_params.append({"Sigma": abs((popt_fs_frac[0]) / np.sqrt(pcov_fs_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max R200": max_r200, "Min R200": min_r200, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold, "Max z": max_z, "Min Vel Diff": min_vel_diff})



results = {
    "Errors": errors,
    "Best sigma for ef fraction": best_ef_sigma,
    "Best parameters for ef fraction sigma": best_ef_params,
    f"Parameters for ef sigma values > {high_sigma}": high_ef_sigma_params,

    "Best sigma for eq fraction": best_eq_sigma,
    "Best parameters for eq fraction sigma": best_eq_params,
    f"Parameters for eq sigma values > {high_sigma}": high_eq_sigma_params,

    "Best sigma for sf fraction": best_sf_sigma,
    "Best parameters for sf fraction sigma": best_sf_params,
    f"Parameters for sf sigma values > {high_sigma}": high_sf_sigma_params,

    "Best sigma for sq fraction": best_sq_sigma,
    "Best parameters for sq fraction sigma": best_sq_params,
    f"Parameters for sq sigma values > {high_sigma}": high_sq_sigma_params,

    "Best sigma for q fraction": best_q_sigma,
    "Best parameters for q fraction sigma": best_q_params,
    f"Parameters for q sigma values > {high_sigma}": high_q_sigma_params,

    "Best sigma for f fraction": best_f_sigma,
    "Best parameters for f fraction sigma": best_f_params,
    f"Parameters for f sigma values > {high_sigma}": high_f_sigma_params,

    "Best sigma for e fraction": best_e_sigma,
    "Best parameters for e fraction sigma": best_e_params,
    f"Parameters for e sigma values > {high_sigma}": high_e_sigma_params,

    "Best sigma for s fraction": best_s_sigma,
    "Best parameters for s fraction sigma": best_s_params,
    f"Parameters for s sigma values > {high_sigma}": high_s_sigma_params,

    "Best sigma for qs fraction": best_qs_sigma,
    "Best parameters for qs fraction sigma": best_qs_params,
    f"Parameters for qs sigma values > {high_sigma}": high_qs_sigma_params,

    "Best sigma for fs fraction": best_fs_sigma,
    "Best parameters for fs fraction sigma": best_fs_params,
    f"Parameters for fs sigma values > {high_sigma}": high_fs_sigma_params,

    "Best sigma for combo": best_combo_sigma,
    "Best parameters for combo": best_combo_params,
    f"Parameters for combo sigma values > {2 * high_sigma}": high_combo_params,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Best parameters for ef fraction sigma:", best_ef_params)
print("Best sigma for ef fraction:", best_ef_sigma)
print(f"Parameters for ef sigma values > {high_sigma}: {high_ef_sigma_params}")

print("Best parameters for eq fraction sigma:", best_eq_params)
print("Best sigma for eq fraction:", best_eq_sigma)
print(f"Parameters for eq sigma values > {high_sigma}: {high_eq_sigma_params}")

print("Best parameters for sf fraction sigma:", best_sf_params)
print("Best sigma for sf fraction:", best_sf_sigma)
print(f"Parameters for sf sigma values > {high_sigma}: {high_sf_sigma_params}")

print("Best parameters for sq fraction sigma:", best_sq_params)
print("Best sigma for sq fraction:", best_sq_sigma)                                
print(f"Parameters for sq sigma values > {high_sigma}: {high_sq_sigma_params}")

print("Best parameters for q fraction sigma:", best_q_params)
print("Best sigma for q fraction:", best_q_sigma)
print(f"Parameters for q sigma values > {high_sigma}: {high_q_sigma_params}")

print("Best parameters for f fraction sigma:", best_f_params)
print("Best sigma for f fraction:", best_f_sigma)
print(f"Parameters for f sigma values > {high_sigma}: {high_f_sigma_params}")

print("Best parameters for e fraction sigma:", best_e_params)
print("Best sigma for e fraction:", best_e_sigma)
print(f"Parameters for e sigma values > {high_sigma}: {high_e_sigma_params}")

print("Best parameters for s fraction sigma:", best_s_params)
print("Best sigma for s fraction:", best_s_sigma)
print(f"Parameters for s sigma values > {high_sigma}: {high_s_sigma_params}")
