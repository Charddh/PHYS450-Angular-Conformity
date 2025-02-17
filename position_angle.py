import pandas as pd
import numpy as np
import ast

angle_df = pd.read_csv('BCGAngleOffset.csv')
bcg_df = pd.read_csv('bcg.csv')

angle_df["clus_id"] = angle_df["clus_id"].str.strip()
bcg_df["bcg_id"] = bcg_df["bcg_id"].str.strip()

merged_df = pd.merge(angle_df, bcg_df, left_on='clus_id', right_on='bcg_id', how = 'inner').drop(columns=['bcg_id'])

merged_df['corrected_pa'] = ((((90 - merged_df['spa']) % 360) - merged_df['bcg_sdss_pa']) % 360)

#merged_df['sat_dec'] = pd.to_numeric(merged_df['sat_dec'], errors='coerce')
#merged_df['sat_ra'] = pd.to_numeric(merged_df['bcg_ra'], errors='coerce')

print(merged_df['sat_dec'].dtypes)

def calculate_theta(bcg_ra, bcg_dec, gal_ra, gal_dec):
    """
    Compute the angle (theta) in radians between a BCG and satellite galaxy.
    """
    if (isinstance(gal_ra, str) and gal_ra.strip() == "[]") or (isinstance(gal_dec, str) and gal_dec.strip() == "[]"):
        return []
    print(type(gal_ra))
    try:
        if isinstance(gal_ra, str):
            gal_ra = ast.literal_eval(gal_ra)
        if isinstance(gal_dec, str):
            gal_dec = ast.literal_eval(gal_dec)
    except (ValueError, SyntaxError):
        return ['Error']
    gal_ra = np.array(gal_ra, dtype=float)
    gal_dec = np.array(gal_dec, dtype=float)
    avg_dec = np.radians((bcg_dec + gal_dec)/2)
    delta_ra = np.radians(bcg_ra - gal_ra)*np.cos(avg_dec)
    delta_dec = np.radians(bcg_dec - gal_dec)
    return np.arctan2(delta_ra, delta_dec)

merged_df['theta'] = merged_df.apply(lambda row: calculate_theta(row['bcg_ra'], row['bcg_dec'], row['sat_ra'], row['sat_dec']), axis=1)

