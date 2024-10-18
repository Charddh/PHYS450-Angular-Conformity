#########################
#Author: John Stott Lancaster University
#This is a quick python tutorial for performing simple data tasks for astrophysics. It is designed to help MPhys students.
#This is for Python 3.x I recommend using Anaconda download of Python (https://www.anaconda.com/products/individual) as this keeps a version of python that is separate to any others you may already have installed on your computer (this can avoid an issue if you have a mac - as messing with the preloaded version of python can upset your operating system).
#I run python code with iPython using a text editor to edit my code but you may prefer an IDE instead as they are probably more user friendly. An IDE is just a window that contains a window for the code text editor and a window for the python output, plus usually a way of keeping track of your variables. I'd avoid Jupyter notebooks though as they require too much interaction and can cause problems with the order in which things are run etc.
#########################
#List of tasks:
#1. preamble at start of python file
#2. a few basic tips on data types and arrays
#3a. read in data
#3b. read out data
#4. scatter plot
#5. np.where function to isolate subsamples within data
#6. fit a line to some points
#7. histogram, make and plot
#8. for loop
#9. if statement 
#10. cosmology, i.e. distance to a given redshift or on-sky distance between 2 points at the same redshift
#11. fit a function to some data
#12. interpolation 
#13. list files in directory  
#14. splitting and inserting variables into strings


########################
#1. preamble
#the purpose of the lines below is to load various modules into python to improve functionality for data analysis as the standard python is quite basic


#the 2 lines below clear all variables in ipython every time you run the code. This is not needed if you are not using ipython and will cause your code to fail so I have commented them out
#from IPython import get_ipython
#get_ipython().run_line_magic('reset','-sf')


import numpy as np #this is a module that has array functionality 


import matplotlib.pyplot as plt #graph plotting module


from astropy.io import ascii #astropy is an astronomy module but here we are just importing its data read in functionality. If astropy not included in your installation you can see how to get it here https://www.astropy.org/


from astropy.table import Table, Column, MaskedColumn# this is astropy's way to make a data table, for reading out data


from astropy.cosmology import FlatLambdaCDM# this is the cosmology module from astropy


import glob #allows you to read the names of files in folders 


from scipy.optimize import curve_fit #allows you to fit your own user defined curves to data


from sys import exit #allows you to put an exit() to stop your code in a certain place, good for debugging or only running part of a script


#you can also write your own modules and place them in the PYTHONPATH for use in all of your scripts


########################
#2. Basics


#variable names: python variable names are case sensitive so be careful a is not the same as A


#Types of data:
#Python complains if you get your data types mixed up so you need to be careful. The 3 most common types you will come across are integers, floats and strings


a=10 #integer
b=10.0 #float - can also just write b=10.  
c='ten'#string


print('a=',a,'b=',b,'c=',c)


#type tells you what type a variable is (works for arrays too). A more extreme version of this is "a?" which gives full details
print('type(a)',type(a),'type(b)',type(b),'type(c)',type(c))


#multiplying integers and floats
print('a*a=',a*a,'b*b=',b*b,'a*b',a*b)


#Maths operators are: plus +, minus -, multiply * and divide /
#raise to the power is ** square root is np.sqrt()
print('raise to power and square root')
print('b**2.0=',b**2.0,'np.sqrt(b)=',np.sqrt(b))




#exit() #uncomment this to stop the code here


#Arrays:
#python without any modules imported works with lists which are inferior to arrays and so try and do everything in numpy arrays. NB lists don't have the mathemtical operations that arrays have i.e. you can't multiply lists by a number and get the more logical answer you get with np.arrays and you can't multiply lists together  
#Python also uses Tuples - e.g. the output of np.where. Tuple vs list: List has mutable nature i.e., list can be changed or modified after its creation according to needs whereas tuple has immutable nature i.e., tuple can’t be changed or modified after its creation.




#to make an array a
a=np.array([4,5,6,7,8])


print('this is array a',a)


b=[1,2,3,4,5]#this is a list
print('this is list b',b)


#turn list b into an array
b=np.array(b)#now its a list


print('turn it into an array b=np.array(b)',b)


#length of an array
print('length of array b, len(b)',len(b))




#to index an array (i.e. select 1 or more elements of the array) remembering python starts counting at element 0


print('a=',a)
print('a[0]=',a[0],' python starts counting at element 0')
print('a[2]=',a[2])
print('a[-1]=',a[-1],'as [-1] is the last element of any array')


#to slice an array i.e. just look at some parts of it
print('a[0:2]=',a[0:2],' Note: The result includes the start index, but excludes the end index. Unlike IDL!')




#change array type


print(a)
print('array dtype',a.dtype)
print('change to float with np.astype')
a=a.astype('float64')#float64 as np float is 64 bit
print(a)
print('array dtype',a.dtype)


#make an array of consecutive numbers
b=np.arange(20)#20 element array with integers from 0-19. Can obviously change this to a float etc with astype
print('20 element array with values from 0-19')
print(b)


#make an empty array
#if you want to make an empty into which you will inset values
a=np.empty(10,dtype=int)#creates an array with 10 elements (you can also specify the dtype, default is float I think), however in certain cases this array is created containing values and so if you aren't going to overwrite them all it may be better to use np.zeros(a) in which case each element will be set to zero  
print('created an empty array of 10 elements',a)
a=np.zeros(10,dtype=int)
print('created an empty array of 10 elements all set to zero',a)


#arrays can also contain many dimensions i.e. a 3x2 array is created by:
a=np.empty((2,3),dtype=int)
print('created a 3x2 array',a)
print('note that array was not truly empty and so use np.zeros instead')
a=np.zeros((2,3),dtype=int)
print(a)


#array multiplication
print('array multiplication (or division)')
a=np.array([1.,2.,3.,4.,5.])
b=np.array([2.,1.,2.,1,2.])


print('a=',a)
print('b=',b)


print('a*2.',a*2.)


print('a*b',a*b)
print('can also add+, subtract-, divide/, raise to power**, square root arrays np.sqrt() in an analogous way')


#number of decimals
#if you want to round your values (i.e. for displaying a value in the the legend of a plot) use np.round()
b=2.3456
print('unrounded',b, 'rounded to 2 decimal places',np.round(b))


#exit() #uncomment this to stop the code here


########################
#3a. Read in data
#this is important!


print('read in an ascii or text table')


datafile='coma_data.cat'#this is a limited number of galaxies (40) within at the same redshift as the Coma galaxy cluster from SDSS data - I'm using it for a new experiment on the Phys363 Astrolab


print('ascii.read from astropy is very good for reading in tables but there, if astropy not included in your installation you can see how to get it here https://www.astropy.org/')
data = ascii.read(datafile)


print('these are the columns it has found, it may not find them depending on format of table, in which case it would call them col1, col2 etc',data.columns)


#if it can't find column names use data = ascii.read(datafile,names=colnames) colnames=['x','y'] etc


#convert the columns of the table to np.arrays
#NB you don't have to convert all of them just the ones you're interested in
ra=np.array(data['RA'])#RA
dec=np.array(data['DEC'])#DEC
z=np.array(data['Z'])#Redshift
z_err=np.array(data['Z_ERR'])#Redshift error
umag=np.array(data['modelMag_u'])# u band magnitude
gmag=np.array(data['modelMag_g'])# g band magnitude
rmag=np.array(data['modelMag_r'])# r band magnitude
imag=np.array(data['modelMag_i'])# i band magnitude
zmag=np.array(data['modelMag_z'])# z band magnitude




print('print out ra',ra)


print('you can also read in data with np.genfromtxt but less user friendly')




#you may also want to read in a .fits table or a .fits image at some point but you can look that up if needed


#exit() #uncomment this to stop the code here


########################
#3a. Read out data




dataout = Table([ra,dec], names=['#RA', 'DEC'])#can use a hash symbol within quotes
ascii.write(dataout, 'new_data.cat', overwrite=True)




#exit() #uncomment this to stop the code here


########################
#4. Plot Data
#below we are using matplotlib to plot
#using the catalogue we read in above
plt.plot(gmag,umag-gmag,'go',label='galaxies')#g means green and o means use circle points - there are a huge number of options here for colours, symbols, linestyles etc so look up matplotlib
#e.g. symbols (markers) are here https://matplotlib.org/api/markers_api.html
#colours are here https://matplotlib.org/3.1.0/gallery/color/named_colors.html
#linestyles https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
plt.xlim(11,19)#x range: it will do this automotically to fit the size of the data but I'm just showing it explicitly
plt.ylim(0.5,2.5)#y range: see above
plt.xlabel('g')#x axis title
plt.ylabel('u-g')#y axis title
plt.legend(loc="lower left")#if your data has labels (i.e. label='galaxies' as above) then this will automatically make a legend for the plot
plt.savefig('cmr.png')#file name you can save as png and pdf perhaps other formats too.
plt.show()#you can comment this out if you don't want to show this plot every time you run the code
plt.close()






#exit() #uncomment this to stop the code here
########################
#5. np.where - this is a really useful function!
#using the catalogue we read in above
bright=np.where(gmag < 12.5)#bright now contains the indices of the array where gmag is less than 12.5. N.B The condition for equals is == not = e.g. to find where x is equal to 1 do np.where(x == 1). Can use multiple conditions separated by &
u_g=umag-gmag#make this a single variable to aid indexing


redseq=np.where(u_g > (gmag*(-0.05)+2.4))#can also use the equation of a line in this case to split the red sequence from the blue cloud
bluecld=np.where(u_g < (gmag*(-0.05)+2.4))#can also use the equation of a line
#for multiple conditions use & (and) and | (or).


xvals=np.arange(10,dtype=float)+10.#making some x values so we can plot the line above on the graph


plt.plot(gmag[bright],u_g[bright],'ys',label='brightest cluster galaxy',markersize=10)
plt.plot(gmag[redseq],u_g[redseq],'r*',label='red sequence',markersize=10)
plt.plot(gmag[bluecld],u_g[bluecld],'bo',label='blue cloud',markersize=7)
plt.plot(xvals,xvals*(-0.05)+2.4,linestyle='dashed',color='m')#make a dashed magenta line
plt.xlim(11,19)#x range it will do this automotically to fit the size of the data but I'm just showing it explicitly
plt.ylim(0.5,2.5)#y range see above
plt.xlabel('g')
plt.ylabel('u-g')
plt.legend(loc="lower left")#if your data has labels (i.e. label='' as above) then this will automatically make a legend for the plot
plt.savefig('cmr_separate.png')#file name you can save as png and pdf perhaps other formats too.
plt.show()#you can comment this out if you don't want to show this plot every time you run the code
plt.close()


#Length of a a=numpy.where(blah blah) print(len(a[0])) as its a tuple
#if you want to index by just the first value of the where tuple then do b[a[0][0]]
#i.e. where a was supposed to return np.where the minimum of an array
#was but returned 2 values as they where both equidistant from the
#minimum requested.


#Also if you want to say something should happen if np.where to return nothing if len(a[0]) == 0


#np.where multiple conditions np.where((A < B) & (C>D))






#exit() #uncomment this to stop the code here
########################
#6. Fit a line


redseq_fit = np.polyfit(gmag[redseq], u_g[redseq], 1)# 1st order polynomial fit of red sequence i.e. a straight line, change number to increase order of fit


plt.plot(gmag[redseq],u_g[redseq],'r*',label='red sequence',markersize=10)#plotting just red sequence
plt.plot(xvals,xvals*redseq_fit[0]+redseq_fit[1],linestyle='dashed',color='k',label='red sequence fit')#inputting the fit parameters from above
plt.xlim(11,19)#x range it will do this automotically to fit the size of the data but I'm just showing it explicitly
plt.ylim(1.4,2.2)#y range see above
plt.xlabel('g')
plt.ylabel('u-g')
plt.legend(loc="lower left")#if your data has labels (i.e. label='' as above) then this will automatically make a legend for the plot
plt.savefig('redseq_fit.png')#file name you can save as png and pdf perhaps other formats too.
plt.show()#you can comment this out if you don't want to show this plot every time you run the code
plt.close()










#exit() #uncomment this to stop the code here
########################
#7. histogram


binwidth=0.5
bins=np.arange(min(gmag), max(gmag) + binwidth, binwidth)


histvals,bin_edges=np.histogram(gmag,bins=bins)


print(histvals)
print(bin_edges)


xvals=bin_edges[0:-1]+binwidth/2.


plt.bar(xvals, histvals, width=binwidth,label='galaxies')
plt.legend(loc="best")#chooses the best location for the legend
plt.xlabel('g')
plt.ylabel('N')
plt.savefig('hist.png')#file name you can save as png and pdf perhaps other formats too.
plt.show()#you can comment this out if you don't want to show this plot every time you run the code
plt.close()








#exit() #uncomment this to stop the code here
########################
#8. for loop


#simple example of a for loop - looping through a procedure i number of times


a=np.arange(10)


print('for loop')


for i in range(len(a)):
    print(a[i])#indent is important
#commands in for loop are those that are indented. Once code below is not indented then no longer in for loop


#while loops are similar to for loops but end after a condition is met


print('while loop')
i=1
while i < 4:
    print(a[i])#indent is important
    i=i+1#indent is important


#exit() #uncomment this to stop the code here
########################
#9 if statement


a=10
b=5


print('a=',a,'b=',b)
if a > b:
    print('a is bigger than b')
else:
    print('b is bigger than a')




a=5
b=10
print('a=',a,'b=',b)
if a > b:
    print('a is bigger than b')
else:
    print('b is bigger than a')


#exit() #uncomment this to stop the code here
########################
#10. Cosmology
# very useful for converting on-sky distances in arcseconds to kpc at a given redshift (i.e. distance between 2 points at same redshift) or obtaining the luminosity distance to an object for a given redshift
# could instead manually use Ned Wright's cosmology calculator online (http://www.astro.ucla.edu/~wright/CosmoCalc.html enter desired parameters and press "flat")


#below sets up the comsology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)#H0=70 and OmegaM=0.3 are reasonable parameters to use (what I use in all of my papers) - as it's flat this means OmegaLambda=0.7


z=0.5#redshift - will also take arrays as well as single values I think
print('cosmology')


lumdist=cosmo.luminosity_distance(z)#luminosity distance, i.e the distance to a galaxy, gives answer in Mpc
print('luminosity dist',lumdist,'this is an object that gives value and units so cant be used in a calculation with a normal number so turn to value with lumdist.value')
print('lumdist.value',lumdist.value)
arcperkpc=cosmo.arcsec_per_kpc_proper(z)#converts distance in arcseconds on sky between 2 points at the same redshfit to kpc
print('arcsec per kpc',arcperkpc)
print('arcperkpc.value',arcperkpc.value)






#exit() #uncomment this to stop the code here


#####################
#11 Fit a user defined function to some data


randomdata = np.random.normal(loc=0, scale=2.0, size=(100))#this makes a random normal distribution of data, can also make random uniform data (np.random.uniform)


binwidth=0.5
histvals,bin_edges=np.histogram(randomdata,bins=np.arange(min(randomdata), max(randomdata) + binwidth, binwidth))


xvals=bin_edges[0:-1]+binwidth/2.#xvals of histogram


plt.bar(xvals, histvals, width=binwidth,label='random data')#plot of random data




#now create a simple guassian function to fit to the data; A module to be imported is basically file with functions in it and so you can keep this separate from your main script if you like and then import it.
#####
def gaussian_function(x, x0, I, sig):
# returns a gaussian, given x values an offset, a normalisation and a width
    import numpy as np
    from numpy import pi, sqrt, exp
    
    func = (I/sig/sqrt(2.0*pi)) * exp((-0.5*(x-x0)**2)/(sig**2)) #gaussian


    return func
#####




#use curve_fit to fit your gaussian function to the random data
p0=[0.,10.,2.]#guess at initial parameters x0, I, sig


##curve_fit outputs 2 arrays coefficients and covariance matrix which you can then convert to errors
coeffs, cov = curve_fit(gaussian_function, xvals, histvals,p0=p0)




plt.plot(xvals, gaussian_function(xvals,coeffs[0],coeffs[1],coeffs[2]),'g',label='fit')#over plot the fitted gaussian on the data


plt.legend(loc="best")
plt.xlabel('x')
plt.ylabel('N')
plt.savefig('gaussfit.png')
plt.show()
plt.close()




#exit() #uncomment this to stop the code here


##########
#12 interpolate
#An interpoltation is when you plot the same data but with different x values. Very useful for when you have xvalues that are measured data points and a non-parametric model to compare with. You can find out the value of the model at each x value of the measured data and compare the difference in measured and model y values 


xvalsshift=xvals+0.2#create an offset with the xvals above


fitgauss=gaussian_function(xvals,coeffs[0],coeffs[1],coeffs[2])#plot gaussian fit from above


yinterp=np.interp(xvalsshift,xvals,fitgauss)#interpolate gaussian fit from above onto the new shifted xvals


plt.plot(xvals, gaussian_function(xvals,coeffs[0],coeffs[1],coeffs[2]),'go',label='original curve')
plt.plot(xvalsshift,yinterp,'bs',label='same curve interpolated to different x values')
plt.legend(loc="best")
plt.xlabel('x')
plt.ylabel('N')
plt.savefig('interpolation.png')
plt.show()
plt.close()


#exit() #uncomment this to stop the code here
#####################
#13 list file names in directory in array/list




filelist=glob.glob('*.png')




print(filelist)
#this is very useful as you could create a for loop around your main script such that it performs the same calculations on a different data file each time. see below.




#exit() #uncomment this to stop the code here
#####################
#14 splitting strings and inserting variables into strings
#using file list above we can name files


namesplit = np.char.split(filelist,sep ='.')#split up the file names from their types at the full stop. Output is an array of lists.


print(namesplit)
for i in range(len(namesplit)):
    newstring=f'this file is called: {namesplit[i][0]}'#this can be used to make new output files based on the original input files.
    print(newstring)
    
exit() #uncomment this to stop the code here
#####################

