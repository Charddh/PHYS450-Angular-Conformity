import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, chi_squared, assign_morph, calculate_theta
from sklearn.model_selection import GridSearchCV

max_z = 0.125 #Maximum redshift in the sample.
min_n = 30 #Minimum number of BCG satellite galaxies.
bin_size = 20 #Size in degrees of the bins.
min_satellite_mass = 10 #Minimum satellite galaxy mass
classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.25 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 1 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 1000 #Maximum physical separation in kpc between BCG and satellite galaxies.
max_vel = 3000 #Maximum velocity difference in km/s between BCG and satellite galaxies.
signal_to_noise = 10 #Minimum signal-to-noise ratio for galaxy spectra.

classification_thresholds = [0.5, 0.6]
debiaseds = [0, 1]
bin_sizes = [20, 30 ,40]
min_ns = [20, 30, 40]
phys_seps = [500, 700, 1000]
max_vels = [1000, 2000, 3000]
signal_to_noises = [5, 7, 10]

best_params = {}
best_sfr_params = {}
best_chi2 = np.inf

for classification_threshold in classification_thresholds:
    for debiased in debiaseds:
        for bin_size in bin_sizes:
            for min_n in min_ns:
                for phys_sep in phys_seps:
                    for max_vel in max_vels:
                        for signal_to_noise in signal_to_noises:
                            # Open the FITS file and retrieve the data from the second HDU (Header Data Unit)
                            cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data
                            # Convert the structured array to a dictionary with byte-swapping and endian conversion for each column
                            cluster_df = pd.DataFrame({
                                name: cluster_data[name].byteswap().newbyteorder()  # Apply byte-swapping and endian conversion to each field
                                for name in cluster_data.dtype.names})  # Iterate over each field name in the structured array

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
                            gz_mass = gz_df['LGM_TOT_P50'].values
                            gz_sfr = gz_df['SPECSFR_TOT_P50'].values
                            gz_s_n = gz_df['SN_MEDIAN'].values

                            if debiased == 1:
                                gz_elliptical = gz_df['P_EL_DEBIASED'].values
                                gz_spiral = gz_df['P_CS_DEBIASED'].values
                            else:
                                gz_elliptical = gz_df['P_EL'].values
                                gz_spiral = gz_df['P_CS'].values

                            # Calculate the angular separation and redshift difference using vectorized operations
                            ra_diff = reduced_clusters_ra[:, None] - gz_ra  # Broadcast the RA difference calculation
                            dec_diff = reduced_clusters_dec[:, None] - gz_dec  # Broadcast the Dec difference calculation
                            z_diff = np.abs(reduced_clusters_z[:, None] - gz_z)  # Compute absolute redshift difference

                            # Compute the angular separation using the Haversine formula and the proper scaling
                            angular_separation = np.sqrt((ra_diff ** 2) * (np.cos(np.radians(reduced_clusters_dec[:, None])) ** 2) + dec_diff ** 2)

                            #Number of degrees corresponding to 1 kpc at the redshift of each cluster.
                            degrees_per_mpc = (1 / 3600) * cosmo.arcsec_per_kpc_proper(reduced_clusters_z[:, None]).value

                            # Apply the selection criteria (angular separation and redshift difference)
                            selected_galaxies_mask = (angular_separation < phys_sep * degrees_per_mpc) & (z_diff < max_vel / 3e5) & (gz_mass > min_satellite_mass) * (gz_s_n > signal_to_noise)

                            # Create a list of Galaxy IDs for each cluster
                            reduced_clusters_locals_id = [gz_id[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_ra = [gz_ra[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_dec = [gz_dec[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_z = [gz_z[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_elliptical = [gz_elliptical[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_spiral = [gz_spiral[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_mass = [gz_mass[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]
                            reduced_clusters_locals_sfr = [gz_sfr[selected_galaxies_mask[i]] for i in range(len(reduced_clusters_ra))]

                            gz_id_list = [item for sublist in reduced_clusters_locals_id for item in sublist]
                            gz_ra_list = [item for sublist in reduced_clusters_locals_ra for item in sublist]
                            gz_dec_list = [item for sublist in reduced_clusters_locals_dec for item in sublist]
                            gz_z_list = [item for sublist in reduced_clusters_locals_z for item in sublist]

                            df_gz = pd.DataFrame({'gz_id': gz_id_list,'gz_ra': gz_ra_list,'gz_dec': gz_dec_list,'gz_z': gz_z_list})
                            df_bcg = pd.DataFrame({'bcg_id': reduced_clusters_id,'bcg_ra': reduced_clusters_ra,'bcg_dec': reduced_clusters_dec,'bcg_z': reduced_clusters_z, 'bcg_sdss_pa': reduced_clusters_pa, 'sat_id': reduced_clusters_locals_id, 'sat_ra': reduced_clusters_locals_ra, 'sat_dec': reduced_clusters_locals_dec, 'sat_elliptical': reduced_clusters_locals_elliptical, 'sat_spiral': reduced_clusters_locals_spiral, 'sat_mass': reduced_clusters_locals_mass, 'sat_sfr': reduced_clusters_locals_sfr})

                            angle_df = pd.read_csv('BCGAngleOffset.csv')
                            angle_df["clus_id"] = angle_df["clus_id"].str.strip()
                            df_bcg["bcg_id"] = df_bcg["bcg_id"].str.strip()

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

                            data = {"angles": sat_majoraxis_list, "types": sat_type_list}
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

                            df_angles = pd.concat([spirals, ellipticals, unknowns], axis=1, keys=["spirals", "ellipticals", "unknowns"])
                            df.to_csv('angles.csv', index=False)

                            bins = np.arange(0, 181, bin_size)

                            spiral_hist, _ = np.histogram(spirals, bins=bins)
                            elliptical_hist, _ = np.histogram(ellipticals, bins=bins)
                            unknown_hist, _ = np.histogram(unknowns, bins=bins)

                            sfr_forming_hist, _ = np.histogram(sfr_forming, bins=bins)
                            sfr_quiescent_hist, _ = np.histogram(sfr_quiescent, bins=bins)
                            sfr_unknown_hist, _ = np.histogram(sfr_unknowns, bins=bins)

                            # Poisson errors for counts
                            sfr_forming_errors = np.sqrt(sfr_forming_hist)
                            sfr_quiescent_errors = np.sqrt(sfr_quiescent_hist)
                            sfr_unknown_errors = np.sqrt(sfr_unknown_hist)

                            # Compute fraction and errors
                            fraction = np.where(spiral_hist + elliptical_hist > 0, (elliptical_hist / (spiral_hist + elliptical_hist)), 0)
                            fraction_errors = np.where(elliptical_hist + spiral_hist > 0, np.sqrt(elliptical_hist) / (elliptical_hist + spiral_hist), np.nan)
                            sfr_fraction = np.where(sfr_forming_hist + sfr_quiescent_hist > 0, (sfr_quiescent_hist / (sfr_forming_hist + sfr_quiescent_hist)), 0)
                            sfr_fraction_errors = np.where(sfr_quiescent_hist + sfr_forming_hist > 0, np.sqrt(sfr_quiescent_hist) / (sfr_quiescent_hist +  sfr_forming_hist), np.nan)
                            bin_centres = (bins[:-1] + bins[1:]) / 2 #Bin midpoints

                            popt, pcov = opt.curve_fit(sine_function, bin_centres, fraction, sigma = fraction_errors, p0 = [0.1, 0], absolute_sigma = True)
                            popt_sfr, pcov_sfr = opt.curve_fit(sine_function, bin_centres, sfr_fraction, sigma = sfr_fraction_errors, p0 = [0.1, 0], absolute_sigma = True)
                            trialX = np.linspace(0, 180, 1000)
                            trialY = sine_function(trialX, *popt)
                            trialY_sfr = sine_function(trialX, *popt_sfr)
                            chi2 = chi_squared(bin_centres, fraction, fraction_errors, popt, sine_function)
                            chi2_sfr = chi_squared(bin_centres, sfr_fraction, sfr_fraction_errors, popt_sfr, sine_function)
                            
                            if chi2 < best_chi2:
                                best_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Min N": min_n, "Physical Separation": phys_sep, "Max Velocity": max_vel, "Signal to Noise": signal_to_noise}
                                best_chi2 = chi2
                            if chi2_sfr < best_chi2:
                                best_sfr_params = {"Classification": classification_threshold, "Debiased": debiased, "Bin Size": bin_size, "Min N": min_n, "Physical Separation": phys_sep, "Max Velocity": max_vel, "Signal to Noise": signal_to_noise}
                                best_chi2 = chi2_sfr

print("Best parameters for elliptical fraction:", best_params)
print("Best chi-squared value for elliptical fraction:", best_chi2)
print("Best parameters for quiescent fraction:", best_sfr_params)
print("Best chi-squared value for quiescent fraction:", best_chi2)