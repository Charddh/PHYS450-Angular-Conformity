import numpy as np #this is a module that has array functionality 
import matplotlib.pyplot as plt #graph plotting module
from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy
import scipy.optimize as opt
from astropy.io import fits
import pandas as pd
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from functions import sine_function, sine_function_2, chi_squared, assign_morph, calculate_theta, calculate_r200, calculate_radial_distance, calculate_velocity_distance
import json

max_z = 0.115 #Maximum redshift in the sample.
min_lx = 3.8e43 #Minimum x-ray luminosity for clusters.
bin_size = 60 #Size in degrees of the bins.
sfr_bin_size = 60 #Size in degrees of the bins for the SFR plot.
min_satellite_mass = 10.2 #Minimum satellite galaxy mass.
max_satellite_mass = 11.5 #Maximum satellite galaxy mass.
classification_threshold = 1 #If 1, will classify based on highest number. Else, will classify based on probability threshold.
sfr_threshold = -11.00 #Threshold of specific star formation rate considered as the boundary between active and quiescent galaxies.
debiased = 0 #If 1, will use debiased classifications. Else, will use raw classifications.
phys_sep = 3000 #Maximum physical separation in kpc between BCG and satellite galaxies.
max_vel = 3000 #Maximum velocity difference in km/s between BCG and satellite galaxies.
signal_to_noise = 1 #Minimum signal-to-noise ratio for galaxy spectra.
axis_bin = 60
mergers = ['1_9618', '1_1626', '1_5811', '1_1645']
min_vel_diff = 0 

use_r200 = 1 #If 1, will use r200 to calculate physical separation. Else, will use physical separation directly.
max_r200 = 6
min_r200 = 0


show_eq_amp = 1
show_sq_amp = 1
show_q_amp = 1
show_s_amp = 1
show_qs_amp = 1
show_fs_amp = 1
show_combined = 1

show_ef_amp = 0
show_sf_amp = 0
show_f_amp = 0
show_e_amp = 0

#For separate plots
first_step = 1
step = 0.08

#For combined plot
"""first_step = 1
step = 0.2"""

if use_r200 == 1:
    extra_steps = [(0.45, 0), (0.55, 0.1), (0.65, 0.2)]
    normal_steps = [(x, x - first_step) for x in np.arange(first_step, 4.5 + step, step)]
    fixed_pairs = extra_steps + normal_steps
else:
    fixed_pairs = [(x, x - 1000) for x in range(1000, 3100, 100)]

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

ef_amp = []
eq_amp = []
sf_amp = []
sq_amp = []
q_amp = []
f_amp = []
e_amp = []
s_amp = []
qs_amp = []
fs_amp = []
r200_list = []
ef_err_list = []
eq_err_list = []
sf_err_list = []
sq_err_list = []
q_err_list = []
f_err_list = []
e_err_list = []
s_err_list = []
qs_err_list = []
fs_err_list = []

#for phys_sep, min_phys_sep in fixed_pairs:
for max_r200, min_r200 in fixed_pairs:
    #print(f"Processing: phys_sep = {phys_sep}, min_phys_sep = {min_phys_sep}")
    print(f"Processing: phys_sep = {max_r200}, min_phys_sep = {min_r200}")          
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

    max_vel = np.where((r200_sep_galaxy >= 0) & (r200_sep_galaxy <= 2),-750 * r200_sep_galaxy + 2000,np.where((r200_sep_galaxy > 2) & (r200_sep_galaxy < 4),500,np.nan))

    #max_vel = np.where((phys_sep_galaxy >= 0) & (phys_sep_galaxy <= 3000),-0.43 * phys_sep_galaxy + 2000,np.where((phys_sep_galaxy > 3000) & (phys_sep_galaxy < 5000), 500,np.nan))

    if use_r200 == 1:
        selected_galaxies_mask = (
            (r200_sep_galaxy <= max_r200) & 
            (r200_sep_galaxy > min_r200) &
            (z_diff < max_vel / 3e5) &
            (gz_mass > min_satellite_mass) &
            (gz_mass < max_satellite_mass) &
            (gz_s_n > signal_to_noise) &
            (z_diff > min_vel_diff / 3e5))
    else:
        selected_galaxies_mask = (
            (phys_sep_galaxy <= phys_sep) & 
            (phys_sep_galaxy > min_phys_sep) &
            (z_diff < max_vel / 3e5) &
            (gz_mass > min_satellite_mass) &
            (gz_mass < max_satellite_mass) &
            (gz_s_n > signal_to_noise) &
            (z_diff > min_vel_diff / 3e5))
        
    selected_counts = [np.sum(selected_galaxies_mask[i]) for i in range(len(reduced_clusters_ra))]
    print("Sel", sum(selected_counts))

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
    merged_df['sat_majoraxis_angle'] = merged_df.apply(lambda row: [((row['corrected_pa'] - theta) % 360) % 180 for theta in row['theta']], axis=1)
    merged_df['radial_sep'] = merged_df.apply(lambda row: calculate_radial_distance(row['bcg_ra'], row['bcg_dec'], row['bcg_z'], row['sat_ra'], row['sat_dec']), axis=1)
    merged_df['vel_diff'] = merged_df.apply(lambda row: calculate_velocity_distance(row['bcg_z'], row['sat_z']), axis=1)
    merged_df['r/r200'] = merged_df['radial_sep'] / merged_df['bcg_r200']
    merged_df['r/r200_copy'] = merged_df['r/r200'].copy()
    merged_df2 = merged_df.explode("r/r200").reset_index(drop=True)
    merged_df_clean = merged_df2.dropna(subset=['r/r200'])
    r200_loop = (merged_df2['r/r200'].max() + merged_df2['r/r200'].min()) / 2
    r200_list.append(r200_loop)

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
    vel_diff_list = np.concatenate(merged_df['vel_diff'].values)
    radial_sep_list = np.concatenate(merged_df['radial_sep'].values)
    r200_list_graph = np.concatenate(merged_df['r/r200_copy'].values)

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

    ef_amp.append(abs(popt_ef_frac[0]))
    eq_amp.append(abs(popt_eq_frac[0]))
    sf_amp.append(abs(popt_sf_frac[0]))
    sq_amp.append(abs(popt_sq_frac[0]))
    q_amp.append(abs(popt_q_frac[0]))
    f_amp.append(abs(popt_f_frac[0]))
    e_amp.append(abs(popt_e_frac[0]))
    s_amp.append(abs(popt_s_frac[0]))
    qs_amp.append(abs(popt_qs_frac[0]))
    fs_amp.append(abs(popt_fs_frac[0]))


    ef_err_list.append(np.sqrt(pcov_ef_frac[0,0]))
    eq_err_list.append(np.sqrt(pcov_eq_frac[0,0]))
    sf_err_list.append(np.sqrt(pcov_sf_frac[0,0]))
    sq_err_list.append(np.sqrt(pcov_sq_frac[0,0]))
    q_err_list.append(np.sqrt(pcov_q_frac[0,0]))
    f_err_list.append(np.sqrt(pcov_f_frac[0,0]))
    e_err_list.append(np.sqrt(pcov_e_frac[0,0]))
    s_err_list.append(np.sqrt(pcov_s_frac[0,0]))
    qs_err_list.append(np.sqrt(pcov_qs_frac[0,0]))
    fs_err_list.append(np.sqrt(pcov_fs_frac[0,0]))

    ef_err_list_sig = 3 * np.array(ef_err_list)
    eq_err_list_sig = 3 * np.array(eq_err_list)
    sf_err_list_sig = 3 * np.array(sf_err_list)
    sq_err_list_sig = 3 * np.array(sq_err_list)
    q_err_list_sig = 3 * np.array(q_err_list)
    f_err_list_sig = 3 * np.array(f_err_list)
    e_err_list_sig = 3 * np.array(e_err_list)
    s_err_list_sig = 3 * np.array(s_err_list)
    qs_err_list_sig = 3 * np.array(qs_err_list)
    fs_err_list_sig = 3 * np.array(fs_err_list)

valid_indices_eq = np.where((np.array(eq_amp) - np.array(eq_err_list_sig)) > 0)
filtered_r200_eq = np.array(r200_list)[valid_indices_eq]
filtered_eq_amp = np.array(eq_amp)[valid_indices_eq]
filtered_eq_err = np.array(eq_err_list_sig)[valid_indices_eq]
mask_filtered_eq = (filtered_eq_amp != 0.03) & (filtered_eq_amp != 0.1)
mask_unfiltered_eq = (np.array(eq_amp) != 0.03) & (np.array(eq_amp) != 0.1)

valid_indices_sq = np.where((np.array(sq_amp) - np.array(sq_err_list_sig)) > 0)
filtered_r200_sq = np.array(r200_list)[valid_indices_sq]
filtered_sq_amp = np.array(sq_amp)[valid_indices_sq]
filtered_sq_err = np.array(sq_err_list_sig)[valid_indices_sq]
mask_filtered_sq = (filtered_sq_amp != 0.03) & (filtered_sq_amp != 0.1)
mask_unfiltered_sq = (np.array(sq_amp) != 0.03) & (np.array(sq_amp) != 0.1)

valid_indices_q = np.where((np.array(q_amp) - np.array(q_err_list_sig)) > 0)
filtered_r200_q = np.array(r200_list)[valid_indices_q]
filtered_q_amp = np.array(q_amp)[valid_indices_q]
filtered_q_err = np.array(q_err_list_sig)[valid_indices_q]
mask_filtered_q = (filtered_q_amp != 0.03) & (filtered_q_amp != 0.1)
mask_unfiltered_q = (np.array(q_amp) != 0.03) & (np.array(q_amp) != 0.1)

valid_indices_s = np.where((np.array(s_amp) - np.array(s_err_list_sig)) > 0)
filtered_r200_s = np.array(r200_list)[valid_indices_s]
filtered_s_amp = np.array(s_amp)[valid_indices_s]
filtered_s_err = np.array(s_err_list_sig)[valid_indices_s]
mask_filtered_s = (filtered_s_amp != 0.03) & (filtered_s_amp != 0.1)
mask_unfiltered_s = (np.array(s_amp) != 0.03) & (np.array(s_amp) != 0.1)

valid_indices_qs = np.where((np.array(qs_amp) - np.array(qs_err_list_sig)) > 0)
filtered_r200_qs = np.array(r200_list)[valid_indices_qs]
filtered_qs_amp = np.array(qs_amp)[valid_indices_qs]
filtered_qs_err = np.array(qs_err_list_sig)[valid_indices_qs]
mask_filtered_qs = (filtered_qs_amp != 0.03) & (filtered_qs_amp != 0.1)
mask_unfiltered_qs = (np.array(qs_amp) != 0.03) & (np.array(qs_amp) != 0.1)

valid_indices_fs = np.where((np.array(fs_amp) - np.array(fs_err_list_sig)) > 0)
filtered_r200_fs = np.array(r200_list)[valid_indices_fs]
filtered_fs_amp = np.array(fs_amp)[valid_indices_fs]
filtered_fs_err = np.array(fs_err_list_sig)[valid_indices_fs]
mask_filtered_fs = (filtered_fs_amp != 0.03) & (filtered_fs_amp != 0.1)
mask_unfiltered_fs = (np.array(fs_amp) != 0.03) & (np.array(fs_amp) != 0.1)

print("eq_amp:", filtered_eq_amp)
print("sq_amp:", filtered_sq_amp)
print("q_amp:", filtered_q_amp)
print("s_amp:", filtered_s_amp)
print("qs_amp:", filtered_qs_amp)
print("fs_amp:", filtered_fs_amp)

if show_eq_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(filtered_r200_eq[mask_filtered_eq], filtered_eq_amp[mask_filtered_eq], yerr=filtered_eq_err[mask_filtered_eq], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_eq], np.array(eq_amp)[mask_unfiltered_eq], yerr=np.array(eq_err_list)[mask_unfiltered_eq], marker='o', markersize = 10, linestyle='none', color="purple", label="Quiescent Fraction Amplitude", capsize=2)    
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylim(0, 3 * max(eq_amp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylabel("Amplitude of Quiescent Elliptical Fit", fontsize=16) 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_sq_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(filtered_r200_sq[mask_filtered_sq], filtered_sq_amp[mask_filtered_sq], yerr=filtered_sq_err[mask_filtered_sq], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_sq], np.array(sq_amp)[mask_unfiltered_sq], yerr=np.array(sq_err_list)[mask_unfiltered_sq], marker='o', markersize = 10, linestyle='none', color="purple", label="Quiescent Fraction Amplitude", capsize=2)    
    ax.set_ylim(0, 3 * max(sq_amp))
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Amplitude of Quiescent Spiral Fit", fontsize=16) 
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_q_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(filtered_r200_q[mask_filtered_q], filtered_q_amp[mask_filtered_q], yerr=filtered_q_err[mask_filtered_q], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_q], np.array(q_amp)[mask_unfiltered_q], yerr=np.array(q_err_list)[mask_unfiltered_q], marker='o', markersize = 10, linestyle='none', color="purple", label="Quiescent Fraction Amplitude", capsize=2)    
    ax.set_ylim(0, 3 * max(q_amp))
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylabel("Amplitude of Quiescent Galaxies Fit", fontsize=16) 
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_s_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(filtered_r200_s[mask_filtered_s], filtered_s_amp[mask_filtered_s], yerr=filtered_s_err[mask_filtered_s], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_s], np.array(s_amp)[mask_unfiltered_s], yerr=np.array(s_err_list)[mask_unfiltered_s], marker='o', markersize = 10, linestyle='none', color="purple", label="Spiral Fraction Amplitude", capsize=2)    
    ax.set_ylim(0, 3 * max(s_amp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylabel("Amplitude of Spiral Galaxies Fit", fontsize=16)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16) 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_qs_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(filtered_r200_qs[mask_filtered_qs], filtered_qs_amp[mask_filtered_qs], yerr=filtered_qs_err[mask_filtered_qs], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_qs], np.array(qs_amp)[mask_unfiltered_qs], yerr=np.array(qs_err_list)[mask_unfiltered_qs], marker='o', markersize = 10, linestyle='none', color="purple", label="Quiescent Spiral Fraction Amplitude", capsize=2)    
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylim(0, 3 * max(qs_amp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylabel("Amplitude of Quiescent Spiral Galaxies Fit", fontsize=16) 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_fs_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.errorbar(filtered_r200_fs[mask_filtered_fs], filtered_fs_amp[mask_filtered_fs], yerr=filtered_fs_err[mask_filtered_fs], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_fs], np.array(fs_amp)[mask_unfiltered_fs], yerr=np.array(fs_err_list)[mask_unfiltered_fs], marker='o', markersize = 10, linestyle='none', color="purple", label="Star-Forming Spiral Fraction Amplitude", capsize=2)    
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylim(0, 3 * max(fs_amp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylabel("Amplitude of Star-Forming Spiral Galaxies Fit", fontsize=16) 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_combined == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    ax.plot(np.array(r200_list)[mask_unfiltered_sq], np.array(sq_amp)[mask_unfiltered_sq], marker='o', markersize = 10, linestyle='-', color="purple", label="Spiral Quiescent Fraction Amplitude")  
    ax.plot(np.array(r200_list)[mask_unfiltered_eq], np.array(eq_amp)[mask_unfiltered_eq], marker='x', markersize = 10, linestyle='-', color="blue", label="Elliptical Quiescent Fraction Amplitude")    
    ax.plot(np.array(r200_list)[mask_unfiltered_q], np.array(q_amp)[mask_unfiltered_q], marker='D', markersize = 10, linestyle='-', color="orange", label="Quiescent Fraction Amplitude")  
    ax.plot(np.array(r200_list)[mask_unfiltered_qs], np.array(qs_amp)[mask_unfiltered_qs], marker='s', markersize = 10, linestyle='-', color="grey", label="Quiescent Spiral Fraction Amplitude")  
    ax.plot(np.array(r200_list)[mask_unfiltered_fs], np.array(fs_amp)[mask_unfiltered_fs], marker='p', markersize = 10, linestyle='-', color="red", label="Star-Forming Spiral Fraction Amplitude")  
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel(r"R/R$_{200}$", fontsize=16)
    ax.set_ylim(0, 1.5 * max(fs_amp))
    ax.set_ylabel("Amplitude", fontsize=16) 
    ax.legend(fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_e_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    valid_indices = np.where((np.array(e_amp) - np.array(e_err_list_sig)) > 0)
    filtered_r200 = np.array(r200_list)[valid_indices]
    filtered_e_amp = np.array(e_amp)[valid_indices]
    filtered_e_err = np.array(e_err_list_sig)[valid_indices]
    ax.errorbar(filtered_r200, filtered_e_amp, yerr=filtered_e_err, marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(r200_list, e_amp, yerr=e_err_list, marker='o', markersize = 10, linestyle='none', color="purple", label="Elliptical Fraction Amplitude", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$")
    ax.set_ylim(0, 3 * max(e_amp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylabel("Amplitude of Elliptical Galaxies Fit") 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_f_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    valid_indices = np.where((np.array(f_amp) - np.array(f_err_list_sig)) > 0)
    filtered_r200 = np.array(r200_list)[valid_indices]
    filtered_f_amp = np.array(f_amp)[valid_indices]
    filtered_f_err = np.array(f_err_list_sig)[valid_indices]
    ax.errorbar(filtered_r200, filtered_f_amp, yerr=filtered_f_err, marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(r200_list, f_amp, yerr=f_err_list, marker='o', markersize = 10, linestyle='none', color="purple", label="Star-Forming Fraction Amplitude", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylim(0, 3 * max(f_amp))
    ax.set_ylabel("Amplitude of Star-Forming Galaxies Fit") 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_sf_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)
    valid_indices = np.where((np.array(sf_amp) - np.array(sf_err_list_sig)) > 0)
    filtered_r200 = np.array(r200_list)[valid_indices]
    filtered_sf_amp = np.array(sf_amp)[valid_indices]
    filtered_sf_err = np.array(sf_err_list_sig)[valid_indices]
    ax.errorbar(filtered_r200, filtered_sf_amp, yerr=filtered_sf_err, marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(r200_list, sf_amp, yerr=sf_err_list, marker='o', linestyle='none', markersize = 10, color="purple", label="Star-Forming Fraction Amplitude", capsize=2)
    ax.set_xlabel(r"R/R$_{200}$")
    ax.set_ylim(0, 3 * max(sf_amp))
    ax.set_ylabel("Amplitude of Star Forming Spiral Fit") 
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()

if show_ef_amp == 1:
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), constrained_layout=True, dpi=200)    
    ax.errorbar(filtered_r200_ef[mask_filtered_ef], filtered_ef_amp[mask_filtered_ef], yerr=filtered_ef_err[mask_filtered_ef], marker='o', markersize = 10, linestyle='none', color="purple", capsize=2, ecolor="black")
    ax.errorbar(np.array(r200_list)[mask_unfiltered_ef], np.array(ef_amp)[mask_unfiltered_ef], yerr=np.array(ef_err_list)[mask_unfiltered_ef], marker='o', markersize = 10, linestyle='none', color="purple", label="Star-Forming Fraction Amplitude", capsize=2)    
    ax.set_xlabel(r"R/R$_{200}$")
    ax.set_ylim(0, 3 * max(ef_amp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_ylabel("Amplitude of Star Forming Elliptical Fit") 
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth = 2)
    plt.show()