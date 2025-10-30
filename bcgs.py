import pandas as pd
from astropy.io import fits

"""# Load the FITS table
bcg_data = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

# Convert to Pandas DataFrame
df = pd.DataFrame(bcg_data)

# Now you can use head()
print(df.columns)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df['RA_BCG'], df['DEC_BCG'], s=10)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('RA vs Dec Scatter Plot')
plt.grid(True)
plt.show()

import pandas as pd
from astropy.io import fits

# Load the FITS table
bcg_data = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

# Convert to Pandas DataFrame
df = pd.DataFrame(bcg_data)

# Now you can use head()
print(df.columns)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df['RA_BCG'], df['DEC_BCG'], s=10)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('RA vs Dec Scatter Plot')
plt.grid(True)
plt.show()4

# Load the FITS table
bcg_data = pd.DataFrame(fits.open("SpidersXclusterBCGs-v2.0.fits")[1].data)

# Convert to Pandas DataFrame
df = pd.DataFrame(bcg_data)

# Now you can use head()
print(df.columns)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df['RA_BCG'], df['DEC_BCG'], s=10)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('RA vs Dec Scatter Plot')
plt.grid(True)
plt.show()"""

# Load the FITS table
bcg_data = bcg_data = pd.read_csv("stott_2008_bcgs.csv")

# Convert to Pandas DataFrame
df = pd.DataFrame(bcg_data)

print(df.columns)

print(len(df['RAJ2000']))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df['RAJ2000'], df['DEJ2000'], s=10)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('RA vs Dec Scatter Plot')
plt.grid(True)
plt.show()


