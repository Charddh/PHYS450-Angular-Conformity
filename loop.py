import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, sine_function_2, chi_squared, assign_morph, calculate_theta

max_z = 0.125 #Maximum redshift in the sample.
min_lx = 1e43 #Minimum x-ray luminosity for clusters.
bin_size = 45 #Size in degrees of the bins.
sfr_bin_size = 45 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.25 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 1 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 1000 #Maximum physical separation in kpc between BCG and satellite galaxies.
max_vel = 3000 #Maximum velocity difference in km/s between BCG and satellite galaxies.
signal_to_noise = 1 #Minimum signal-to-noise ratio for galaxy spectra.
axis_bin = 60
mergers = ['1_9772', '1_1626', '2_3729', '1_5811', '1_1645', '1_9618', '1_4456', '2_19468']
binaries = ['2_13471', '1_12336', '1_9849', '1_21627', '2_1139', '1_23993', '2_3273', '2_11151', '1_14486', '1_4409', '1_14573', '1_14884', '1_5823', '1_14426']

classification_thresholds = [0.5, 0.6, 0.8, 1]
debiaseds = [0, 1]
bin_sizes = [20, 30, 45, 60]
phys_seps = [500, 750, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000]
min_phys_seps = [0]
sfr_thresholds = [-10.75, -11.0, -11.25, -11.5, -11.75]
min_lxs = [1e42, 0.5e43, 1e43, 5e43]


total_iterations = (len(classification_thresholds) * len(debiaseds) * len(bin_sizes) * len(phys_seps) * len(min_phys_seps) * len(min_lxs) * len(sfr_thresholds))

high_sigma = 2.5

best_params = {}
best_sfr_params = {}
best_sigma_params = {}
high_sigma_params = []
best_sfr_sigma_params = {}
high_sfr_sigma_params = []
best_amplitude = 0
best_sfr_amplitude = 0
best_sigma = 0
best_sfr_sigma = 0

best_ef_params = {}
best_sq_params = {}
best_ef_sigma_params = {}
high_ef_sigma_params = []
best_sq_sigma_params = {}
high_sq_sigma_params = []
best_ef_amplitude = 0
best_sq_amplitude = 0
best_ef_sigma = 0
best_sq_sigma = 0

i = 0
for classification_threshold in classification_thresholds:
    for debiased in debiaseds:
        for bin_size in bin_sizes:
            for phys_sep in phys_seps:
                for min_phys_sep in min_phys_seps:
                    for min_lx in min_lxs:
                        for sfr_threshold in sfr_thresholds:
                            i += 1
                            print(f"Iteration: {i} / {total_iterations}")
                            if (min_phys_sep > phys_sep):
                                print("min phys sep > max phys sep, skipping iteration")
                                continue              
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
                            degrees_per_kpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(reduced_clusters_z[:, None]).value
                            
                            if 0 <= phys_sep <= 3000:
                                max_vel = -0.43 * phys_sep + 2000
                            if phys_sep > 3000:
                                max_vel = 500

                            #Apply the selection criteria (angular separation and redshift difference).

                            if 0 < min_phys_sep <= 3000:
                                min_vel = -0.43 * min_phys_sep + 2000
                                selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (angular_separation > min_phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise) & (z_diff > min_vel / 3e5)
                            elif min_phys_sep == 0:
                                min_vel = 0
                                selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise)
                            elif 3000 < min_phys_sep < 5000:
                                min_vel = 500
                                selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_kpc) & (angular_separation > min_phys_sep * degrees_per_kpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) & (gz_s_n > signal_to_noise) & (z_diff > min_vel / 3e5)    
                            else:
                                print(min_phys_sep)
                                raise ValueError("min_phys_sep must be between 0 and 5000")
                            
                            selected_counts = [np.sum(selected_galaxies_mask[i]) for i in range(len(reduced_clusters_ra))]
                            if sum(selected_counts) < 2:
                                print("Not enough data points, skipping iteration.")
                                continue

                            
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

                            sfr_data = {"angles": sat_majoraxis_list, "types": sfr_type_list}
                            sfr_df=pd.DataFrame(sfr_data)
                            sfr_df["forming_angles"] = sfr_df["angles"].where(sfr_df["types"] == "f")
                            sfr_df["quiescent_angles"] = sfr_df["angles"].where(sfr_df["types"] == "q")
                            sfr_df["unknown_angles"] = sfr_df["angles"].where(sfr_df["types"] == "u")
                            sfr_forming = sfr_df["forming_angles"].dropna().reset_index(drop=True)
                            sfr_quiescent = sfr_df["quiescent_angles"].dropna().reset_index(drop=True)
                            sfr_unknowns = sfr_df["unknown_angles"].dropna().reset_index(drop=True)

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

                            bins = np.arange(0, 181, bin_size)
                            bin_centres = (bins[:-1] + bins[1:]) / 2
                            trialX = np.linspace(0, 180, 1000)

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
                            popt_sq_frac, pcov_sq_frac = opt.curve_fit(sine_function_2, bin_centres, sq_fraction, sigma = sq_fraction_err, p0 = [0.1, 0.75], absolute_sigma = True)
                            trialY_ef_frac = sine_function_2(trialX, *popt_ef_frac)
                            trialY_sq_frac = sine_function_2(trialX, *popt_sq_frac)

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
                
                            popt_sfr, pcov_sfr = opt.curve_fit(sine_function_2, sat_majoraxis_list, sfr_list, sigma = sfr_error, p0 = [1, -11], absolute_sigma = True)
                            popt_frac, pcov_frac = opt.curve_fit(sine_function_2, bin_centres, fraction, sigma = fraction_errors, p0 = [0.1, 0.75], absolute_sigma = True)
                            popt_sfr_frac, pcov_sfr_frac = opt.curve_fit(sine_function_2, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, p0 = [0.1, 0], absolute_sigma = True)
                            trialY_frac = sine_function_2(trialX, *popt_frac)
                            trialY_sfr = sine_function_2(trialX, *popt_sfr)
                            trialY_sfr_frac = sine_function_2(trialX, *popt_sfr_frac)

                            print("popt", popt_frac[0])
                                                        
                            if abs(popt_frac[0]) > abs(best_amplitude):
                                best_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_amplitude = popt_frac[0]
                            if abs(popt_sfr_frac[0]) > abs(best_sfr_amplitude):
                                best_sfr_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_sfr_amplitude = popt_sfr_frac[0]
                            if abs((popt_frac[0]) / np.sqrt(pcov_frac[0,0])) > abs(best_sigma):
                                best_sigma_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_sigma = popt_frac[0] / np.sqrt(pcov_frac[0,0])
                            if abs((popt_frac[0]) / np.sqrt(pcov_frac[0,0])) > high_sigma:
                                high_sigma_params.append({"Sigma": abs((popt_frac[0]) / np.sqrt(pcov_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Lx": min_lx, "Min Velocity": min_vel, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold})                        
                            if abs((popt_sfr_frac[0]) / np.sqrt(pcov_sfr_frac[0,0])) > abs(best_sfr_sigma):
                                best_sfr_sigma_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_sfr_sigma = popt_sfr_frac[0] / np.sqrt(pcov_sfr_frac[0,0])
                            if abs((popt_sfr_frac[0]) / np.sqrt(pcov_sfr_frac[0,0])) > high_sigma:
                                high_sfr_sigma_params.append({"Sigma": abs((popt_sfr_frac[0]) / np.sqrt(pcov_sfr_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Lx": min_lx, "Min Velocity": min_vel, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold})                        
                            
                            if abs(popt_ef_frac[0]) > abs(best_ef_amplitude):
                                best_ef_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_ef_amplitude = popt_ef_frac[0]
                            if abs(popt_sq_frac[0]) > abs(best_sq_amplitude):
                                best_sq_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_sq_amplitude = popt_sq_frac[0]
                            if abs((popt_ef_frac[0]) / np.sqrt(pcov_ef_frac[0,0])) > abs(best_ef_sigma):
                                best_ef_sigma_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_ef_sigma = popt_ef_frac[0] / np.sqrt(pcov_ef_frac[0,0])
                            if abs((popt_ef_frac[0]) / np.sqrt(pcov_ef_frac[0,0])) > high_sigma:
                                high_ef_sigma_params.append({"Sigma": abs((popt_ef_frac[0]) / np.sqrt(pcov_ef_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Lx": min_lx, "Min Velocity": min_vel, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold})
                            if abs((popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) > abs(best_sq_sigma):
                                best_sq_sigma_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold}
                                best_sq_sigma = popt_sq_frac[0] / np.sqrt(pcov_sq_frac[0,0])
                            if abs((popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])) > high_sigma:
                                high_sq_sigma_params.append({"Sigma": abs((popt_sq_frac[0]) / np.sqrt(pcov_sq_frac[0,0])), "Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Max Physical Separation": phys_sep, "Min Physical Separation": min_phys_sep, "Max Velocity": max_vel, "Min Velocity": min_vel, "Min Lx": min_lx, "Number of satellites in sample": len(sat_majoraxis_list), "SFR Threshold": sfr_threshold})

print("Best parameters for ef fraction amplitude:", best_ef_params)
print("Best amplitude value for ef fraction:", best_ef_amplitude)

print("Best parameters for ef fraction sigma:", best_ef_sigma_params)
print("Best sigma for ef fraction:", best_ef_sigma)
print(f"Parameters for ef sigma values > {high_sigma}: {high_ef_sigma_params}")

print("Best parameters for sq fraction amplitude:", best_sq_params)
print("Best amplitude value for sq fraction:", best_sq_amplitude)

print("Best parameters for sq fraction sigma:", best_sq_sigma_params)
print("Best sigma for sq fraction:", best_sq_sigma)                                
print(f"Parameters for sq sigma values > {high_sigma}: {high_sq_sigma_params}")

print("Best parameters for elliptical fraction amplitude:", best_params)
print("Best amplitude value for elliptical fraction:", best_amplitude)

print("Best parameters for elliptical fraction sigma:", best_sigma_params)
print("Best sigma for elliptical fraction:", best_sigma)
print(f"Parameters for elliptical sigma values > {high_sigma}: {high_sigma_params}")

print("Best parameters for quiescent fraction amplitude:", best_sfr_params)
print("Best amplitude value for quiescent fraction:", best_sfr_amplitude)

print("Best parameters for quiescent fraction sigma:", best_sfr_sigma_params)
print("Best sigma for quiescent fraction:", best_sfr_sigma)
print(f"Parameters for quiescent sigma values > {high_sigma}: {high_sfr_sigma_params}")