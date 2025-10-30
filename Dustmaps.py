from dustmaps import config
from dustmaps.bayestar import BayestarWebQuery
from astropy.coordinates import SkyCoord
import astropy.units as u

# configure dustmaps to fetch web map (it will cache locally)
config['data_dir'] = '/tmp/dustmaps'  # change as needed

# Bayestar (Green et al. 2019) - covers Dec > -30 deg
bayestar = BayestarWebQuery(version='bayestar2019')
c = SkyCoord(ra=180*u.deg, dec=-20*u.deg, frame='icrs')   # example coord
ebv = bayestar(c)   # returns E(B-V) vs distance (array of samples / profile)
print('Bayestar E(B-V) profile (RA,Dec)=', ebv)


"""#Zucker et al 2025 for Dec < -30 deg
decaps = DecapsQuery()
c = SkyCoord(ra=270*u.deg, dec=-45*u.deg, frame='icrs')  # example in south
ebv = decaps(c)
print('DECaPS E(B-V) profile (RA,Dec)=', ebv)"""