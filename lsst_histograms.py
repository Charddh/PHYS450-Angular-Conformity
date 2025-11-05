import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

cluster_data1 = pd.read_csv("VII110Atable3.csv")
df_clus1 = pd.DataFrame(cluster_data1)
cluster_data2 = pd.read_csv("VII110Atable4.csv")
df_clus2 = pd.DataFrame(cluster_data2)
cluster_data3 = pd.read_csv("VII110Atable5.csv")
df_clus3 = pd.DataFrame(cluster_data3)
cluster_data4 = pd.read_csv("VII110Atable6.csv")
df_clus4 = pd.DataFrame(cluster_data4)
abell_df = pd.concat(
    [cluster_data1, cluster_data2, cluster_data3, cluster_data4],
    ignore_index=True
)

cluster_data1 = pd.read_csv("JA+A685A106emain.csv")
erosita_df = pd.DataFrame(cluster_data1)

cluster_data1 = pd.read_csv("JApJS2241cat_dr8.csv")
df_clus1 = pd.DataFrame(cluster_data1)
cluster_data2 = pd.read_csv("JApJS2241sva1exp.csv")
df_clus2 = pd.DataFrame(cluster_data2)
redmapper_df = pd.concat(
    [cluster_data1, cluster_data2],
    ignore_index=True
)

cluster_data1 = pd.read_csv("JAJ14752table2.csv")
cluster_df = pd.DataFrame(cluster_data1)
cdfs_df = cluster_df[cluster_df["Cl"].astype(str).str.contains("4|5", na=False)]

print("abell", abell_df.columns)
print("erosita", erosita_df.columns)
print("redmapper", redmapper_df.columns)

gal_data = pd.read_csv("lsst_dp1.csv")
galaxy_df = pd.DataFrame(gal_data)

show_spatial = 0
show_spatial_nearby = 0
show_histograms = 0
show_cdfs = 1

gal_coords = SkyCoord(ra=galaxy_df['coord_ra'].values*u.deg,
                      dec=galaxy_df['coord_dec'].values*u.deg)

abell_coords = SkyCoord(ra=abell_df['_RA_icrs'].values*u.deg,
                      dec=abell_df['_DE_icrs'].values*u.deg)
erosita_coords = SkyCoord(ra=erosita_df['RAJ2000'].values*u.deg,
                      dec=erosita_df['DEJ2000'].values*u.deg)
redmapper_coords = SkyCoord(ra=redmapper_df['RAJ2000'].values*u.deg,
                      dec=redmapper_df['DEJ2000'].values*u.deg)
cdfs_coords = SkyCoord(ra=cdfs_df['RAJ2000'].values*u.deg,
                      dec=cdfs_df['DEJ2000'].values*u.deg)

radius = 3 * u.arcmin
# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = abell_coords.match_to_catalog_sky(gal_coords)
# Filter galaxies within the radius
mask_near = d2d < radius
abell_nearby = abell_df[mask_near].copy()

print("an", abell_nearby["z"])

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = erosita_coords.match_to_catalog_sky(gal_coords)
# Filter galaxies within the radius
mask_near = d2d < radius
erosita_nearby = erosita_df[mask_near].copy()

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = redmapper_coords.match_to_catalog_sky(gal_coords)
# Filter galaxies within the radius
mask_near = d2d < radius
redmapper_nearby = redmapper_df[mask_near].copy()

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = cdfs_coords.match_to_catalog_sky(gal_coords)
# Filter galaxies within the radius
mask_near = d2d < radius
cdfs_nearby = cdfs_df[mask_near].copy()

if show_cdfs == 1:
    plt.figure(figsize=(8, 6))
    ra = 53.16
    dec = -28.10
    radius = 1
    plt.scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    plt.scatter(abell_nearby['_RA_icrs'], abell_nearby['_DE_icrs'], label="Abell")
    plt.scatter(erosita_nearby['RAJ2000'], erosita_nearby['DEJ2000'], label="eROSITA")
    plt.scatter(redmapper_nearby['RAJ2000'], redmapper_nearby['DEJ2000'], label="redMaPPer")
    plt.scatter(cdfs_nearby['RAJ2000'], cdfs_nearby['DEJ2000'], label="CDFS Clusters", color = "gold")
    plt.title("ECDFS")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.xlim(ra - radius, ra + radius)
    plt.ylim(dec - radius, dec + radius)
    plt.legend()
    plt.show()

if show_spatial == 1:
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    radius = 1.0

    # EDFS
    ra = 59.10
    dec = -48.73
    axes[0,0].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[0,0].scatter(abell_df['_RA_icrs'], abell_df['_DE_icrs'], label="Abell")
    axes[0,0].scatter(erosita_df['RAJ2000'], erosita_df['DEJ2000'], label="eROSITA")
    axes[0,0].scatter(redmapper_df['RAJ2000'], redmapper_df['DEJ2000'], label="redMaPPer")
    axes[0,0].set_title("EDFS")
    axes[0,0].set_xlabel("RA")
    axes[0,0].set_ylabel("Dec")
    axes[0,0].set_xlim(ra - radius, ra + radius)
    axes[0,0].set_ylim(dec - radius, dec + radius)
    axes[0,0].legend()

    # ECDFS
    ra = 53.16
    dec = -28.10
    axes[0,1].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[0,1].scatter(abell_df['_RA_icrs'], abell_df['_DE_icrs'], label="Abell")
    axes[0,1].scatter(erosita_df['RAJ2000'], erosita_df['DEJ2000'], label="eROSITA")
    axes[0,1].scatter(redmapper_df['RAJ2000'], redmapper_df['DEJ2000'], label="redMaPPer")
    axes[0,1].set_title("ECDFS")
    axes[0,1].set_xlabel("RA")
    axes[0,1].set_ylabel("Dec")
    axes[0,1].set_xlim(ra - radius, ra + radius)
    axes[0,1].set_ylim(dec - radius, dec + radius)
    axes[0,1].legend()

    # LELF
    ra = 37.98
    dec = 7.015
    axes[1,0].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[1,0].scatter(abell_df['_RA_icrs'], abell_df['_DE_icrs'], label="Abell")
    axes[1,0].scatter(erosita_df['RAJ2000'], erosita_df['DEJ2000'], label="eROSITA")
    axes[1,0].scatter(redmapper_df['RAJ2000'], redmapper_df['DEJ2000'], label="redMaPPer")
    axes[1,0].set_title("LELF")
    axes[1,0].set_xlabel("RA")
    axes[1,0].set_ylabel("Dec")
    axes[1,0].set_xlim(ra - radius, ra + radius)
    axes[1,0].set_ylim(dec - radius, dec + radius)
    axes[1,0].legend()

    # LGLF
    ra = 95.0
    dec = -25.0
    axes[1,1].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[1,1].scatter(abell_df['_RA_icrs'], abell_df['_DE_icrs'], label="Abell")
    axes[1,1].scatter(erosita_df['RAJ2000'], erosita_df['DEJ2000'], label="eROSITA")
    axes[1,1].scatter(redmapper_df['RAJ2000'], redmapper_df['DEJ2000'], label="redMaPPer")
    axes[1,1].set_title("LGLF")
    axes[1,1].set_xlabel("RA")
    axes[1,1].set_ylabel("Dec")
    axes[1,1].set_xlim(ra - radius, ra + radius)
    axes[1,1].set_ylim(dec - radius, dec + radius)
    axes[1,1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if show_spatial_nearby == 1:
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    radius = 1.0

    # EDFS
    ra = 59.10
    dec = -48.73
    axes[0,0].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[0,0].scatter(abell_nearby['_RA_icrs'], abell_nearby['_DE_icrs'], label="Abell")
    axes[0,0].scatter(erosita_nearby['RAJ2000'], erosita_nearby['DEJ2000'], label="eROSITA")
    axes[0,0].scatter(redmapper_nearby['RAJ2000'], redmapper_nearby['DEJ2000'], label="redMaPPer")
    axes[0,0].set_title("EDFS")
    axes[0,0].set_xlabel("RA")
    axes[0,0].set_ylabel("Dec")
    axes[0,0].set_xlim(ra - radius, ra + radius)
    axes[0,0].set_ylim(dec - radius, dec + radius)
    axes[0,0].legend()

    # ECDFS
    ra = 53.16
    dec = -28.10
    axes[0,1].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[0,1].scatter(abell_nearby['_RA_icrs'], abell_nearby['_DE_icrs'], label="Abell")
    axes[0,1].scatter(erosita_nearby['RAJ2000'], erosita_nearby['DEJ2000'], label="eROSITA")
    axes[0,1].scatter(redmapper_nearby['RAJ2000'], redmapper_nearby['DEJ2000'], label="redMaPPer")
    axes[0,1].set_title("ECDFS")
    axes[0,1].set_xlabel("RA")
    axes[0,1].set_ylabel("Dec")
    axes[0,1].set_xlim(ra - radius, ra + radius)
    axes[0,1].set_ylim(dec - radius, dec + radius)
    axes[0,1].legend()

    # LELF
    ra = 37.98
    dec = 7.015
    axes[1,0].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[1,0].scatter(abell_nearby['_RA_icrs'], abell_nearby['_DE_icrs'], label="Abell")
    axes[1,0].scatter(erosita_nearby['RAJ2000'], erosita_nearby['DEJ2000'], label="eROSITA")
    axes[1,0].scatter(redmapper_nearby['RAJ2000'], redmapper_nearby['DEJ2000'], label="redMaPPer")
    axes[1,0].set_title("LELF")
    axes[1,0].set_xlabel("RA")
    axes[1,0].set_ylabel("Dec")
    axes[1,0].set_xlim(ra - radius, ra + radius)
    axes[1,0].set_ylim(dec - radius, dec + radius)
    axes[1,0].legend()

    # LGLF
    ra = 95.0
    dec = -25.0
    axes[1,1].scatter(galaxy_df['coord_ra'], galaxy_df['coord_dec'], label="LSST DP1", s = 1)
    axes[1,1].scatter(abell_nearby['_RA_icrs'], abell_nearby['_DE_icrs'], label="Abell")
    axes[1,1].scatter(erosita_nearby['RAJ2000'], erosita_nearby['DEJ2000'], label="eROSITA")
    axes[1,1].scatter(redmapper_nearby['RAJ2000'], redmapper_nearby['DEJ2000'], label="redMaPPer")
    axes[1,1].set_title("LGLF")
    axes[1,1].set_xlabel("RA")
    axes[1,1].set_ylabel("Dec")
    axes[1,1].set_xlim(ra - radius, ra + radius)
    axes[1,1].set_ylim(dec - radius, dec + radius)
    axes[1,1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if show_histograms == 1:
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    radius = 1.0

    # EDFS
    ra = 59.10
    dec = -48.73
    galaxy = galaxy_df.loc[(galaxy_df["coord_ra"] > ra - radius) & (galaxy_df["coord_ra"] < ra + radius) & (galaxy_df["coord_dec"] > dec - radius) & (galaxy_df["coord_dec"] < dec + radius), "bpz_z_median"].copy()
    abell = abell_nearby.loc[(abell_nearby["_RA_icrs"] > ra - radius) & (abell_nearby["_RA_icrs"] < ra + radius) & (abell_nearby["_DE_icrs"] > dec - radius) & (abell_nearby["_DE_icrs"] < dec + radius), "z"].copy()
    erosita = erosita_nearby.loc[(erosita_nearby["RAJ2000"] > ra - radius) & (erosita_nearby["RAJ2000"] < ra + radius) & (erosita_nearby["DEJ2000"] > dec - radius) & (erosita_nearby["DEJ2000"] < dec + radius), "zBest"].copy()
    redmapper = redmapper_nearby.loc[(redmapper_nearby["RAJ2000"] > ra - radius) & (redmapper_nearby["RAJ2000"] < ra + radius) & (redmapper_nearby["DEJ2000"] > dec - radius) & (redmapper_nearby["DEJ2000"] < dec + radius), "zlambda"].copy()
    # Example: top-left subplot
    ax[0,0].hist(galaxy, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax[0,0].set_xlabel("Redshift (z)")
    ax[0,0].set_ylabel("Count")
    ax[0,0].set_title("EDFS")
    # Add vertical lines for Abell, eROSITA, and Redmapper
    for z in abell:
        ax[0,0].axvline(z, color='red', linestyle='--', linewidth=2, label='Abell')
    for z in erosita:
        ax[0,0].axvline(z, color='green', linestyle='-.', linewidth=2, label='eROSITA')
    for z in redmapper:
        ax[0,0].axvline(z, color='purple', linestyle=':', linewidth=2, label='Redmapper')
    # Avoid duplicate legend labels
    handles, labels = ax[0,0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[0,0].legend(unique.values(), unique.keys())
    ax[0,0].grid(True, linestyle='--', alpha=0.6)

    # ECDFS
    ra = 53.16
    dec = -28.10
    galaxy = galaxy_df.loc[(galaxy_df["coord_ra"] > ra - radius) & (galaxy_df["coord_ra"] < ra + radius) & (galaxy_df["coord_dec"] > dec - radius) & (galaxy_df["coord_dec"] < dec + radius), "bpz_z_median"].copy()
    abell = abell_nearby.loc[(abell_nearby["_RA_icrs"] > ra - radius) & (abell_nearby["_RA_icrs"] < ra + radius) & (abell_nearby["_DE_icrs"] > dec - radius) & (abell_nearby["_DE_icrs"] < dec + radius), "z"].copy()
    erosita = erosita_nearby.loc[(erosita_nearby["RAJ2000"] > ra - radius) & (erosita_nearby["RAJ2000"] < ra + radius) & (erosita_nearby["DEJ2000"] > dec - radius) & (erosita_nearby["DEJ2000"] < dec + radius), "zBest"].copy()
    redmapper = redmapper_nearby.loc[(redmapper_nearby["RAJ2000"] > ra - radius) & (redmapper_nearby["RAJ2000"] < ra + radius) & (redmapper_nearby["DEJ2000"] > dec - radius) & (redmapper_nearby["DEJ2000"] < dec + radius), "zlambda"].copy()
    print(redmapper)
    # Example: top-left subplot
    ax[0,1].hist(galaxy, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax[0,1].set_xlabel("Redshift (z)")
    ax[0,1].set_ylabel("Count")
    ax[0,1].set_title("ECDFS")
    # Add vertical lines for Abell, eROSITA, and Redmapper
    for z in abell:
        ax[0,1].axvline(z, color='red', linestyle='--', linewidth=2, label='Abell')
    for z in erosita:
        ax[0,1].axvline(z, color='green', linestyle='-.', linewidth=2, label='eROSITA')
    for z in redmapper:
        ax[0,1].axvline(z, color='purple', linestyle=':', linewidth=2, label='Redmapper')
    # Avoid duplicate legend labels
    handles, labels = ax[0,1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[0,1].legend(unique.values(), unique.keys())
    ax[0,1].grid(True, linestyle='--', alpha=0.6)

    # LELF
    ra = 37.98
    dec = 7.015
    galaxy = galaxy_df.loc[(galaxy_df["coord_ra"] > ra - radius) & (galaxy_df["coord_ra"] < ra + radius) & (galaxy_df["coord_dec"] > dec - radius) & (galaxy_df["coord_dec"] < dec + radius), "bpz_z_median"].copy()
    abell = abell_nearby.loc[(abell_nearby["_RA_icrs"] > ra - radius) & (abell_nearby["_RA_icrs"] < ra + radius) & (abell_nearby["_DE_icrs"] > dec - radius) & (abell_nearby["_DE_icrs"] < dec + radius), "z"].copy()
    erosita = erosita_nearby.loc[(erosita_nearby["RAJ2000"] > ra - radius) & (erosita_nearby["RAJ2000"] < ra + radius) & (erosita_nearby["DEJ2000"] > dec - radius) & (erosita_nearby["DEJ2000"] < dec + radius), "zBest"].copy()
    redmapper = redmapper_nearby.loc[(redmapper_nearby["RAJ2000"] > ra - radius) & (redmapper_nearby["RAJ2000"] < ra + radius) & (redmapper_nearby["DEJ2000"] > dec - radius) & (redmapper_nearby["DEJ2000"] < dec + radius), "zlambda"].copy()
    
    # Example: top-left subplot
    ax[1,0].hist(galaxy, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax[1,0].set_xlabel("Redshift (z)")
    ax[1,0].set_ylabel("Count")
    ax[1,0].set_title("LELF")
    # Add vertical lines for Abell, eROSITA, and Redmapper
    for z in abell:
        ax[1,0].axvline(z, color='red', linestyle='--', linewidth=2, label='Abell')
    for z in erosita:
        ax[1,0].axvline(z, color='green', linestyle='-.', linewidth=2, label='eROSITA')
    for z in redmapper:
        ax[1,0].axvline(z, color='purple', linestyle=':', linewidth=2, label='Redmapper')
    # Avoid duplicate legend labels
    handles, labels = ax[1,0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[1,0].legend(unique.values(), unique.keys())
    ax[1,0].grid(True, linestyle='--', alpha=0.6)

    # LGLF
    ra = 95.0
    dec = -25.0
    galaxy = galaxy_df.loc[(galaxy_df["coord_ra"] > ra - radius) & (galaxy_df["coord_ra"] < ra + radius) & (galaxy_df["coord_dec"] > dec - radius) & (galaxy_df["coord_dec"] < dec + radius), "bpz_z_median"].copy()
    abell = abell_nearby.loc[(abell_nearby["_RA_icrs"] > ra - radius) & (abell_nearby["_RA_icrs"] < ra + radius) & (abell_nearby["_DE_icrs"] > dec - radius) & (abell_nearby["_DE_icrs"] < dec + radius), "z"].copy()
    erosita = erosita_nearby.loc[(erosita_nearby["RAJ2000"] > ra - radius) & (erosita_nearby["RAJ2000"] < ra + radius) & (erosita_nearby["DEJ2000"] > dec - radius) & (erosita_nearby["DEJ2000"] < dec + radius), "zBest"].copy()
    redmapper = redmapper_nearby.loc[(redmapper_nearby["RAJ2000"] > ra - radius) & (redmapper_nearby["RAJ2000"] < ra + radius) & (redmapper_nearby["DEJ2000"] > dec - radius) & (redmapper_nearby["DEJ2000"] < dec + radius), "zlambda"].copy()
    # Example: top-left subplot
    ax[1,1].hist(galaxy, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax[1,1].set_xlabel("Redshift (z)")
    ax[1,1].set_ylabel("Count")
    ax[1,1].set_title("LGLF")
    # Add vertical lines for Abell, eROSITA, and Redmapper
    for z in abell:
        ax[1,1].axvline(z, color='red', linestyle='--', linewidth=2, label='Abell')
    for z in erosita:
        ax[1,1].axvline(z, color='green', linestyle='-.', linewidth=2, label='eROSITA')
    for z in redmapper:
        ax[1,1].axvline(z, color='purple', linestyle=':', linewidth=2, label='Redmapper')
    # Avoid duplicate legend labels
    handles, labels = ax[1,1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[1,1].legend(unique.values(), unique.keys())
    ax[1,1].grid(True, linestyle='--', alpha=0.6)
    # Adjust layout and display
    plt.tight_layout()
    plt.show()