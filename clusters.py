"""import pandas as pd
from astropy.io import fits

# Load the FITS table
cluster_data = fits.open("catCluster-SPIDERS_RASS_CLUS-v3.0.fits")[1].data

# Convert to Pandas DataFrame
df = pd.DataFrame(cluster_data)

# Now you can use head()
print(df.columns)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df['RA'], df['DEC'], s=10)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('RA vs Dec Scatter Plot')
plt.grid(True)
plt.show()"""

# Access astronomical databases
from pyvo import registry  # version >=1.6

# Moc and HEALPix tools
from mocpy import MOC

# Coordinates manipulation
from astropy.coordinates import SkyCoord

# Sky visualization
from ipyaladin import Aladin  # version >=0.4.0

# For plots
import matplotlib.pyplot as plt

# the catalogue name in VizieR
CATALOGUE = "J/A+A/688/A187"

# each resource in the VO has an identifier, called ivoid. For vizier catalogs,
# the VO ids can be constructed like this:
catalogue_ivoid = f"ivo://CDS.VizieR/{CATALOGUE}"
# the actual query to the registry
voresource = registry.search(ivoid=catalogue_ivoid)[0]

# We can print metadata information about the catalogue
#print(voresource.describe(verbose=True))

tables = voresource.get_tables()
#print(f"In this catalogue, we have {len(tables)} tables.")
#for table_name, table in tables.items():
    #print(f"{table_name}: {table.description}")

# We can also extract the tables names for later use
tables_names = list(tables.keys())

# we get the conesearch  service associated to the first table
conesearch_interface = voresource.get_interface(service_type='conesearch', keyword='J/A+A/688/A187/mcxcii', lax=True)
# if you get a TypeError about an unexpected keyword, check that you installed pyvo>=1.6
conesearch_service = conesearch_interface.to_service()
#print(conesearch_service)
conesearch_radius = 1 / 60.0  # in degrees
conesearch_center = (0.029500, 8.274400)
conesearch_records = conesearch_service.search(pos=conesearch_center,sr=conesearch_radius,)
print(conesearch_records)

# retrieve the MOC
catalogue_coverage = MOC.from_vizier_table(CATALOGUE)
#catalogue_coverage.display_preview()