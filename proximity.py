from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd


"""
#Abell
cluster_data1 = pd.read_csv("VII110Atable3.csv")
df_clus1 = pd.DataFrame(cluster_data1)

cluster_data2 = pd.read_csv("VII110Atable4.csv")
df_clus2 = pd.DataFrame(cluster_data2)

cluster_data3 = pd.read_csv("VII110Atable5.csv")
df_clus3 = pd.DataFrame(cluster_data3)

cluster_data4 = pd.read_csv("VII110Atable6.csv")
df_clus4 = pd.DataFrame(cluster_data4)

df_clus = pd.concat(
    [cluster_data1, cluster_data2, cluster_data3, cluster_data4],
    ignore_index=True
)
#print(df_clus.columns)
#print("gal", df_bcg.iloc[0])

gal_data = pd.read_csv("lsst_table.csv")
df_gal = pd.DataFrame(gal_data)

new_row = pd.DataFrame({
    'coord_ra': [3.79],
    'coord_dec': [-23.88]
})

#df_gal = pd.concat([df_gal, new_row], ignore_index=True)

print("cols", df_clus['_DE_icrs'])

clus_coords = SkyCoord(ra=df_clus['_RA_icrs'].values*u.deg,
                      dec=df_clus['_DE_icrs'].values*u.deg)

gal_coords = SkyCoord(ra=df_gal['coord_ra'].values*u.deg,
                      dec=df_gal['coord_dec'].values*u.deg)

# Set a threshold for "nearby": say 1 arcminute
radius = 12 * u.arcmin

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = gal_coords.match_to_catalog_sky(clus_coords)

# Filter galaxies within the radius
mask_near = d2d < radius

nearby_galaxies = df_gal[mask_near].copy()
nearby_galaxies['nearest_bcg_index'] = idx[mask_near]
nearby_galaxies['angular_sep_arcsec'] = d2d[mask_near].arcsec

print("gals", nearby_galaxies)"""

#redMaPPer
"""cluster_data1 = pd.read_csv("JApJS2241cat_dr8.csv")
df_clus1 = pd.DataFrame(cluster_data1)

cluster_data2 = pd.read_csv("JApJS2241sva1exp.csv")
df_clus2 = pd.DataFrame(cluster_data2)

df_clus = pd.concat(
    [cluster_data1, cluster_data2],
    ignore_index=True
)
#print(df_clus.columns)
#print("gal", df_bcg.iloc[0])

gal_data = pd.read_csv("lsst_table.csv")
df_gal = pd.DataFrame(gal_data)

#df_gal = pd.concat([df_gal, new_row], ignore_index=True)

print("cols", df_clus['RAJ2000'])

clus_coords = SkyCoord(ra=df_clus['RAJ2000'].values*u.deg,
                      dec=df_clus['DEJ2000'].values*u.deg)

gal_coords = SkyCoord(ra=df_gal['coord_ra'].values*u.deg,
                      dec=df_gal['coord_dec'].values*u.deg)

# Set a threshold for "nearby": say 1 arcminute
radius = 12 * u.arcmin

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = gal_coords.match_to_catalog_sky(clus_coords)

# Filter galaxies within the radius
mask_near = d2d < radius

nearby_galaxies = df_gal[mask_near].copy()
nearby_galaxies['nearest_bcg_index'] = idx[mask_near]
nearby_galaxies['angular_sep_arcsec'] = d2d[mask_near].arcsec

print("gals", nearby_galaxies)"""

"""#SRG/eROSITA 
cluster_data1 = pd.read_csv("JA+A685A106emain.csv")
df_clus = pd.DataFrame(cluster_data1)

#print(df_clus.columns)
#print("gal", df_bcg.iloc[0])

gal_data = pd.read_csv("lsst_table.csv")
df_gal = pd.DataFrame(gal_data)

#df_gal = pd.concat([df_gal, new_row], ignore_index=True)

print("cols", df_clus['RAJ2000'])

clus_coords = SkyCoord(ra=df_clus['RAJ2000'].values*u.deg,
                      dec=df_clus['DEJ2000'].values*u.deg)

gal_coords = SkyCoord(ra=df_gal['coord_ra'].values*u.deg,
                      dec=df_gal['coord_dec'].values*u.deg)

# Set a threshold for "nearby": say 1 arcminute
radius = 12 * u.arcmin

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = gal_coords.match_to_catalog_sky(clus_coords)

# Filter galaxies within the radius
mask_near = d2d < radius

nearby_galaxies = df_gal[mask_near].copy()
nearby_galaxies['nearest_bcg_index'] = idx[mask_near]
nearby_galaxies['angular_sep_arcsec'] = d2d[mask_near].arcsec

print("gals", nearby_galaxies)"""

#MACS
cluster_data1 = pd.read_csv("2010MNRAS.407...83E.csv")
df_clus = pd.DataFrame(cluster_data1)

#print(df_clus.columns)
#print("gal", df_bcg.iloc[0])

gal_data = pd.read_csv("lsst_table.csv")
df_gal = pd.DataFrame(gal_data)

#df_gal = pd.concat([df_gal, new_row], ignore_index=True)

print("cols", df_clus['RA'])

clus_coords = SkyCoord(ra=df_clus['RA'].values*u.deg,
                      dec=df_clus['Dec'].values*u.deg)

gal_coords = SkyCoord(ra=df_gal['coord_ra'].values*u.deg,
                      dec=df_gal['coord_dec'].values*u.deg)

# Set a threshold for "nearby": say 1 arcminute
radius = 12 * u.arcmin

# For each galaxy, find the nearest BCG and distance
idx, d2d, d3d = gal_coords.match_to_catalog_sky(clus_coords)

# Filter galaxies within the radius
mask_near = d2d < radius

nearby_galaxies = df_gal[mask_near].copy()
nearby_galaxies['nearest_bcg_index'] = idx[mask_near]
nearby_galaxies['angular_sep_arcsec'] = d2d[mask_near].arcsec

print("gals", nearby_galaxies)