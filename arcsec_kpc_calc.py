from astropy.cosmology import FlatLambdaCDM

# very useful for converting on-sky distances in arcseconds to kpc at a given redshift (i.e. distance between 2 points at same redshift) or obtaining the luminosity distance to an object for a given redshift
# could instead manually use Ned Wright's cosmology calculator online (http://www.astro.ucla.edu/~wright/CosmoCalc.html enter desired parameters and press "flat")


#below sets up the comsology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)#H0=70 and OmegaM=0.3 are reasonable parameters to use (what I use in all of my papers) - as it's flat this means OmegaLambda=0.7


z=0.0453#redshift - will also take arrays as well as single values I think
print('cosmology')


lumdist=cosmo.luminosity_distance(z)#luminosity distance, i.e the distance to a galaxy, gives answer in Mpc
print('luminosity dist',lumdist,'this is an object that gives value and units so cant be used in a calculation with a normal number so turn to value with lumdist.value')
print('lumdist.value',lumdist.value)
arcperkpc=cosmo.arcsec_per_kpc_proper(z)#converts distance in arcseconds on sky between 2 points at the same redshfit to kpc
print('arcsec per kpc',arcperkpc)
print('arcperkpc.value',arcperkpc.value)






#exit() #uncomment this to stop the code here