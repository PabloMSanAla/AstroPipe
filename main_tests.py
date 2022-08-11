#%%


from AstroPipe.classes import image,SExtractor,AstroGNU,directories
from AstroPipe.masking import automatic_mask, ds9_region_masking 
from AstroPipe.utilities import where, check_print, redshift_to_kpc, make_parser,mag_limit, closest
from AstroPipe.sbprofile import isophotal_photometry,isophotal_photometry_fix,rectangular_photometry
from AstroPipe.sbprofile import aaron_break_finder



from astropy.io import fits
import numpy as np
import os 
from astropy.table import Table
import astropy.units as u





''' Initialize images to process'''
index = 3
# index =  make_parser().parse_args().index
objects = Table.read('objects.fits')

object_name = objects['Name'][index].strip()
file = f'/Users/pmsa/Documents/PhD/Projects/Objetos_INT/RAW/{object_name}.fits'
hdu = 0

object_name = 'ngc4244'
file = '/Users/pmsa/Documents/INT/galaxies/Lapalma/20220427/ngc4244/g/ngc4244_g.fits'
file = '/Users/pmsa/Downloads/wetransfer_ngc3556_combined_band_g_clipped-fit_2022-04-30_2201/ngc3556_g.fits'
hdu = 0





file_extension = file.split('.')[-1]
folders = directories(object_name)
folders.set_regions('/Users/pmsa/Documents/PhD/Projects/Objetos_INT/Regions/')


AstroPipe_steps = ['SExtractor','NoiseChisel', 'ds9' , 'FABADA','Photometry_fit',
                    'Photometry_man','Save']


img = image(file,hdu=hdu)

zp = 22.5

coords = {'ra': objects['RA'][index] , 'dec':  objects['DEC'][index]}   
distance = redshift_to_kpc(objects['redshift'][index]).value

# coords = {'ra':	184.373579,'dec':+37.807111}
coords = {'ra':167.879029, 'dec': +55.674122}
# distance = 4345.1 #kpc
distance = 9550 #kpc

img.obj(coords["ra"], coords["dec"])


sex_config = {
                "CHECKIMAGE_TYPE": "SEGMENTATION",
                "DEBLEND_MINCONT": 0.1,
                "DEBLEND_NTHRESH": 32,
                "BACK_SIZE": 64,
                "SATUR_LEVEL": 50000.0,
                "MAG_ZEROPOINT": 22.5,
                'PHOT FLUXFRAC': 0.5,
                'DETECT_THRESH': 1.5,
            }

#%% PSF

#%% MASKING

'''Create Final Mask '''
folders.set_mask(os.path.join(folders.out,img.name+'_mask.fits'))
if not os.path.exists(folders.mask):

    check_print(f'Creating Mask for {img.name}...')

    automatic_mask(img,folders,sex_config = sex_config)

    # ds9_region_masking(img,folders)

else:
    check_print(f'Reading Mask from {folders.mask}...')
    img.set_mask(1-fits.getdata(folders.mask))
    img.set_std(fits.getheader(folders.mask)['STD'])


# %% PROFILE



folders.set_profile(os.path.join(folders.out,img.name+'_profile.fits'))

img.get_morphology(width=(1200,1200))


if not os.path.exists(folders.profile):
    check_print(f'Creating Profile for {img.name}...')
    img.get_background(max_r=1200)
    check_print(str(img.bkg))
    profile = isophotal_photometry(img,r_eff=2.5*img.r_eff,
                        plot=True,save=folders,max_r=1500)

    check_print(f'From pa={img.pa} eps={img.eps} ...')
    img.set_morphology(pa=np.nanpercentile(profile['pa'].value,99),
                    eps=np.nanpercentile(profile['ellipticity'],99))
    check_print(f'to pa={img.pa} eps={img.eps} ...')
    check_print(f'Creating Fix Profile for {img.name}...')
    profile_fix = isophotal_photometry_fix(img,
                        plot=True, save=folders,max_r=1500)

    check_print(f'Creating Rectangular Profile for {img.name}...')
    profile_rec = rectangular_photometry(img,
                        plot=True, save=folders,max_r=1500)

else:
    check_print(f'Reading Profile from {folders.profile}...')
    profile = Table.read(folders.profile, 
                units=[u.arcsec]+3*[u.mag/u.arcsec**2]+[u.deg]+[None]+2*[u.deg])
    profile_fix = Table.read(folders.profile.split('.')[-2]+'_fixed.fits', 
                units=[u.arcsec]+3*[u.mag/u.arcsec**2])
    profile_rec = Table.read(folders.profile.split('.')[-2]+'_rect.fits', 
                units=[u.arcsec]+3*[u.mag/u.arcsec**2])
    img.set_background(profile.meta['background'])

#%% Let`s play with the profile
from astropy.modeling.functional_models import Moffat2D
from scipy.signal import convolve2d
from AstroPipe.plotting import show
# add value 

# data = img.data.data + img.mu_to_counts(22)

# add white noise

# data = img.data.data + np.random.normal(0,5*img.std,img.data.shape)

# add gradient 
xx, yy = np.meshgrid(np.arange(0,img.data.shape[1]),
                    np.arange(0,img.data.shape[0]))
zz = np.sqrt((xx-img.pix[0])**2 + (yy-img.pix[1])**2)
zz /= zz.max()
zz *= img.mu_to_counts(24)
data = img.data.data + zz

# convolution with moffat 
# size = (100,100)
# xx, yy = np.meshgrid(np.arange(0,size[0]),
#                     np.arange(0,size[1]))
                    
# psf_model = Moffat2D(1,size[0]/2,size[1]/2,0.1,0.7)
# psf = psf_model(xx,yy)
# psf /= np.sum(psf)
# psf[np.where(psf<np.nanpercentile(psf,90))] = 0

data = convolve2d(img.data.data,psf,mode='same')
show([img.data.data,data])
img.set_data(data)
pf_pert = isophotal_photometry_fix(img)

#%%
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

plt.figure()
ax = plt.subplot(111)
ax.invert_yaxis()

ax.plot(profile_fix['radius'], profile_fix['surface_brightness'],'g-',label='Real')
ax.fill_between(profile_fix['radius'],profile_fix['surface_brightness']+profile_fix['sb_err_upper'],
                profile_fix['surface_brightness']-profile_fix['sb_err_low'],color='g',alpha=0.5)

ax.plot(pf_pert['radius'], pf_pert['surface_brightness'],'r-',label='Perturbed')
ax.fill_between(pf_pert['radius'].value,pf_pert['surface_brightness'].value+pf_pert['sb_err_upper'].value,
                pf_pert['surface_brightness'].value-pf_pert['sb_err_low'].value,color='r',alpha=0.5)
ax.set_title('Image * PSF{Moffat(1,0.1,0.7)}')
ax.set_xlabel('Radius [arcsec]')
ax.set_ylabel('$\mu [mag \cdot arcsec^{-2}]$')
ax.legend()
plt.tight_layout()
plt.savefig('profile_psf.jpg',dpi=200)
#%% PLOTTING
check_print('Plotting Profile...')

import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

save_fig = True
extent = np.array([-img.pix[0],img.data.shape[1]-img.pix[0],
        -img.pix[1],img.data.shape[0]-img.pix[1]])
extent *= img.pixel_scale


trunc_arg = aaron_break_finder(profile['radius'].value,profile['surface_brightness'].value)

trunc_aper = patches.Ellipse((0,0),
                2*profile['radius'].value[trunc_arg[0]],
                2*(profile['radius'].value[trunc_arg[0]] * (1 - profile['ellipticity'][trunc_arg[0]])),
                profile['pa'].value[trunc_arg[0]],
                color='white',alpha=0.8,fill=False)


fig = plt.figure(figsize=(14,7))
ax1 = plt.subplot2grid((5,2),(0,0),rowspan=5)
ax2 = plt.subplot2grid((5,2),(0,1),rowspan=3)
ax4 = plt.subplot2grid((5,2),(3,1),rowspan=1,sharex=ax2)
ax5 = plt.subplot2grid((5,2),(4,1),rowspan=1,sharex=ax2)

fig.suptitle(object_name,fontsize=15)

ax1.imshow(img.data, origin='lower',norm=LogNorm(vmin=0.0001,vmax=0.999),
            cmap='nipy_spectral_r',extent=extent)

ax1.add_patch(trunc_aper)
ax1.scatter(img.pix[0], img.pix[1],c='r',s=30,marker='x',alpha=0.5)
ax1.set_xlabel('distance $[arcsec]$')
ax1.set_ylabel('distance $[arcsec]$')


ax2.plot(profile_fix['radius'],profile_fix['surface_brightness'],'g-',label='Ellip-Aper')
ax2.fill_between(profile_fix['radius'].value, 
       profile_fix['surface_brightness'].value - profile_fix['sb_err_low'].value,
       profile_fix['surface_brightness'].value + profile_fix['sb_err_upper'].value,
       alpha=0.4,color='g')

# ax2.plot(profile_rec['radius'],profile_rec['surface_brightness'],'cs',label='Rect-Aper')
# ax2.fill_between(profile_rec['radius'].value, 
#        profile_rec['surface_brightness'].value - profile_rec['sb_err_low'].value,
#        profile_rec['surface_brightness'].value + profile_rec['sb_err_upper'].value,
#        alpha=0.4,color='c')

ax2.plot(profile['radius'],profile['surface_brightness'],'r-',label='Astropy')
ax2.fill_between(profile['radius'].value, 
        profile['surface_brightness'].value - profile['sb_err_low'].value,
        profile['surface_brightness'].value + profile['sb_err_upper'].value,
        alpha=0.4,color='r')
ax2.axhline(img.counts_to_mu(img.bkg),label='Background',ls='--',c='b',alpha=0.7)
ax2.axhline(mag_limit(img.std),label='Mag-Limit',ls='-.',c='g',alpha=0.7)

for arg in trunc_arg:
    ax2.axvline(profile['radius'].value[arg],color='darkorange',ls=':',label='Disk Break')

ax2.set_ylabel('$\mu [mag*arcsec^{-2}]$',fontsize=15)
ax2.legend(fontsize=10)
ax2.invert_yaxis()

ax2.set_ylim([np.max([np.nanmax(profile['surface_brightness'].value),img.counts_to_mu(img.bkg)+0.5]),
    np.nanmin(profile['surface_brightness'].value)-0.5])
ax2.set_xlim([-10,profile['radius'].value[closest(profile['surface_brightness'].value,ax2.get_ylim()[0])]+10])
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%2d'))



ax4.plot(profile['radius'],profile['pa'],'r-')
ax4.axhline(img.pa,c='indianred',ls=':')
ax4.set_ylabel('PA [deg]',fontsize = 12)
ax4.yaxis.set_tick_params(labelsize=12)
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

ax5.plot(profile['radius'],profile['ellipticity'],'r-')
ax5.axhline(img.eps,c='indianred',ls=':')
ax5.set_ylabel('Eps',fontsize = 12)
ax5.yaxis.set_tick_params(labelsize=12)
ax5.set_xlabel('R [$arcsec$]')


ax6 = ax2.twiny()

ax6.plot(profile['radius']*np.pi*distance/(3600*180),
        profile['surface_brightness'],alpha=0)
ax6.set_xlabel('Radius [kpc]')


ax1.set_xlim([-round(ax2.get_xlim()[1]/10+1)*10,round(ax2.get_xlim()[1]/10+1)*10])
ax1.set_ylim([-round(ax2.get_xlim()[1]/10+1)*10,round(ax2.get_xlim()[1]/10+1)*10])

for ax in [ax2, ax4]:
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_yticks(ax.get_yticks()[1:]) 


plt.subplots_adjust(top=0.895,
bottom=0.1,
left=0.07,
right=0.94,
hspace=0.1,
wspace=0.1)

# plt.tight_layout()

# if save_fig:
    # fig.savefig(os.path.join(folders.out,object_name+'_photometry.jpg'),dpi=200)

x_lim = ax2.get_xlim()
#%% Aaron Watkins Break Finder

from scipy import stats
from scipy.ndimage import median_filter
from scipy.signal import argrelextrema

def derivative(x,y,n=4):
    """
    Computes de slope from the n adjacent points using 
    linear regression
    """
    der = np.zeros_like(x)
    for i in range(len(x)):
        if i<n:
            slope = stats.linregress(x[:i+n],y[:i+n])[0]
        elif len(x)-i<n:
            slope = stats.linregress(x[i-n:],y[i-n:])[0]
        else:
            slope = stats.linregress(x[i-n:i+n],y[i-n:i+n])[0]
        der[i] = slope
    return der

def aaron_break_finder(rad,mu,mu_lim=21,n=4,p=5):
    """
    Finds the disk breaks in the surface brightness profile as 
    seen in Watkins et al. (2019)
    """
    index = np.where(mu>mu_lim)[0]
    der = derivative(rad[index],mu[index],n=n)
    der = median_filter(der,size=int(p*len(mu)/100))
    cum_sum = np.cumsum(der-np.mean(der))
    maximum = index[0] + argrelextrema(cum_sum, np.greater)[0]
    minimum = index[0] + argrelextrema(cum_sum, np.less)[0]
    return np.sort(np.append(maximum,minimum))

mu = profile['surface_brightness']
mu_err = np.max([profile['sb_err_upper'],
            profile['sb_err_low']],axis=0)[np.isfinite(mu)]
rad = profile['radius']#[np.isfinite(mu)]
# mu = mu[np.isfinite(mu)]

n,p,min,max = 4,5,21,30.5
index = where([mu>min,np.isfinite(mu),mu<max])[0]
der = derivative(rad[index],mu[index],n=n)
der = median_filter(der,size=int(p*len(mu)/100))
second_der =  median_filter(derivative(rad[index],der,n=n),size=int(p*len(mu)/100))
cum_sum = np.cumsum(der-np.mean(der))
maximum = rad[index][argrelextrema(cum_sum, np.greater)[0]]
minimum = rad[index][argrelextrema(cum_sum, np.less)[0]]



fig = plt.figure(figsize=(10,12))
fig.suptitle('Watkins et al 2019 Break Finder\n'
f'for {img.name}',fontsize=15)
ax = plt.subplot(411)
ax2 = plt.subplot(412,sharex=ax)
ax3 = plt.subplot(413,sharex=ax)
ax4 = plt.subplot(414,sharex=ax)

ax.plot(rad,mu,'r-')
ax.set_ylabel('$\mu [mag*arcsec^{-2}]$',fontsize=15)
ax.invert_yaxis()

ax2.plot(rad[index],der,'g-')
ax2.set_ylabel('$\delta \mu / \delta r$')

ax3.plot(rad[index],second_der,'y-')
ax3.set_ylabel('$\delta^2 \mu / \delta r^2$')

ax4.plot(rad[index],cum_sum,'b-')
for point in np.append(maximum,minimum):
    ax.axvline(point,c='c',ls=':')
    ax2.axvline(point,c='c',ls=':')
    ax3.axvline(point,c='c',ls=':')
    ax4.axvline(point,c='c',ls=':')

ax4.set_xlabel('R [arcsec]')
ax4.set_ylabel('Cumulative Sum',fontsize=14)
plt.tight_layout()
ax.set_xlim(x_lim)

plt.savefig(img.name+'_break_finder.jpg',dpi=200)

#%% FFT testing

import scipy.fft  as fft
from scipy.ndimage import median_filter

def LP_gaussian(R0, datShape):
    base = np.zeros(datShape)
    center = datShape[0]/2
    for x in range(len(base)):
        base[x] = np.exp(((-(x-center)**2)/(2*(R0**2))))
    return base

def gauss_filter(data,radius=10):
    data_spec = np.fft.fftshift(np.fft.fft(data))
    filtered = np.abs(np.fft.ifft(np.fft.ifftshift(data_spec * LP_gaussian(radius, data.shape))))
    return filtered

def step_func(size):
    step = np.ones(size)
    step[:int(size/2)] = 0
    return step

def aroon_break_finder(radius,mu,size=5,point=0):
    mu = mu[point:]
    radius = radius[point:]
    mu_smooth = median_filter(mu,size=size)
    dmu_drad = np.gradient(mu_smooth,radius)
    cum_sum = np.cumsum(dmu_drad-np.nanmean(dmu_drad))
    return cum_sum

def closest(data,value):
    return np.argmin(np.abs(data-value))

mu = profile['surface_brightness']
mu_err = np.max([profile['sb_err_upper'],
            profile['sb_err_low']],axis=0)[np.isfinite(mu)]
rad = profile['radius'][np.isfinite(mu)]
mu = mu[np.isfinite(mu)]

mu_filtered = gauss_filter(mu,radius=50)

mu_fft_filtered = np.fft.fft(mu_filtered)
mu_fft = np.fft.fft(mu)
mu_freq = np.fft.fftfreq(mu.size,d=np.mean(np.diff(rad)))



plt.figure()
plt.plot(mu_freq, np.abs(mu_fft),'ro')
plt.plot(mu_freq, np.abs(mu_fft_filtered),'b.')
plt.yscale('log')

plt.figure()
ax = plt.subplot(111)
ax.plot(rad,mu,'ro')
ax.plot(rad,mu_filtered,'b-')
ax.invert_yaxis()

index_filt = np.multiply(mu_freq>-0.028,mu_freq<0.028)
filt = index_filt
mu_hp = np.abs(np.fft.ifft(np.fft.ifftshift(mu_fft*filt)))

plt.figure(figsize=(8,4))
ax = plt.subplot(121)
ax2 = plt.subplot(122)
ax.plot(rad,mu_hp,'r')
ax.invert_yaxis()
ax2.plot(mu_freq,np.abs(mu_fft),'b.',alpha=0.3)
ax2.plot(mu_freq*filt,np.abs(mu_fft)*filt,'ro')
ax2.set_yscale('log')
plt.tight_layout()

#%% Yago Finding Break Testing


from scipy.ndimage import median_filter
from fabada import fabada

mu = profile['surface_brightness']
intensity = img.mu_to_counts(mu)
# mu = intensity
mu_err = np.max([profile['sb_err_upper'],
            profile['sb_err_low']],axis=0)[np.isfinite(mu)]
rad = profile['radius'][np.isfinite(mu)]
mu = mu[np.isfinite(mu)]

# mu = mu_filtered

n=1
dmu_drad = np.gradient(fabada(mu,n*mu_err**2),rad)
ddmu_drad = np.gradient(fabada(mu/dmu_drad,n*mu_err**2),rad)

# dmu_drad = np.gradient(mu,rad)
# ddmu_drad = np.gradient(dmu_drad,rad)


cum_sum = np.cumsum(dmu_drad - np.mean(dmu_drad))
cum_sum_div = np.cumsum(mu/dmu_drad - np.mean(mu/dmu_drad))

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(311)
ax2 = plt.subplot(312,sharex=ax)
ax3 = plt.subplot(313,sharex=ax)
ax.invert_yaxis()


ax.plot(rad,mu,'b')
# ax.set_xlim([rad[75],rad[-1]])
ax.set_ylabel('$\mu [mag \cdot arcsec^{-2}]$',fontsize=14)

ax2.plot(rad,mu/dmu_drad,'r-')
# ax2.axhline(np.mean(dmu_drad),ls=':',c='coral')
# ax2.axvline(rad[closest(dmu_drad,np.mean(dmu_drad))],ls=':',c='coral')
# ax2.set_ylim([0,0.15])
ax2.set_ylabel('$\delta ln(\mu) / \delta r$',fontsize=14)

ax3.plot(rad,cum_sum_div,'r-')
ax3.set_ylabel('Cumulative Sum',fontsize=14)
 
plt.tight_layout()
# ax3.plot(rad,(mu/dmu_drad)/ddmu_drad)
# ax3.axhline(0,c='b',ls='-.',alpha=0.5)
# ax3.axhline(n*mu_err,'b:',alpha=0.5)
# ax3.axhline(-n*mu_err,'b:',alpha=0.5)
# ax3.plot(rad,(ddmu_drad/mu_err)**2)

# ax3.axhline(1,ls=':')
# ax3.set_ylim([-0.2,5])

# ax3.set_ylim([-0.01,0.01])
# ax3.axvline(rad[closest(dmu_drad,np.mean(dmu_drad))],c='r')
# ax3.axvline(rad[np.argmin(cum_sum)],c='b')

index_reg = mu>21
cum_sum_reg = np.cumsum(dmu_drad[index_reg] - np.mean(dmu_drad[index_reg]))
plt.figure()
plt.title('Above 21 mag arcsec^-2')
plt.plot(rad[index_reg],cum_sum_reg)
plt.xlabel('Radius [arcsec]')
 



#%% Fit the profile each time with 1,2,3,... sersic

from astropy.modeling import models, fitting
from matplotlib.animation import FuncAnimation 


fit = fitting.LevMarLSQFitter()

# initialize a linear model
intensity = img.mu_to_counts(mu).value

step = np.arange(10,len(mu),1,dtype=int)

model_init = models.Sersic1D() 
models_fitted = []
chi2,chi2_derivative = np.inf,0
chi2_previous = np.inf

converged = False
k = 0

while not converged:

    models_fitted.append(fit(model_init, rad, intensity))

    mu_fitted = img.counts_to_mu(models_fitted[-1](rad))
    chi2 = np.mean((mu_fitted - mu)**2/mu_err**2)

    model_init = models.Sersic1D() 
    for m in models_fitted:
        model_init += m 
    
    chi2_derivative = chi2 - chi2_previous  
    print(chi2,chi2_previous,chi2_derivative)
    chi2_previous = chi2
    k+=1  
    if chi2_derivative > 0:
        converged = True
        final_model = models_fitted[-2]
        
# Plotting results 


fig = plt.figure()
ax = plt.subplot(211)
ax = plt.subplot2grid((3,1),(0,0),rowspan=2)
ax2 = plt.subplot2grid((3,1),(2,0),sharex=ax)
ax2.set_xlabel('Radius [arcsec]')
ax2.set_ylabel('Residual',fontsize=11)
ax.set_ylabel('SB mag*arcsec^-2',fontsize=13)
ax.invert_yaxis()
ax.plot(rad,mu,'ro')

p, = ax.plot([],[],'b-')
r, = ax2.plot([],[],'c-.',label='Mean = 0')

ax2.set_yscale('log')
ax2.set_ylim([0,1000])

plt.tight_layout()

save_model = []


def fitter(frame):
# fit the data with the fitter
    mu_fitted = img.counts_to_mu(models_fitted[frame](rad))
    p.set_data(rad,mu_fitted)
    residual = (mu_fitted-mu)**2/mu_err**2
    r.set_data(rad,residual)
    return p,r

anim = FuncAnimation(fig, fitter, frames = len(models_fitted), interval = 500, blit = True)

#%% ¿Cum sum, really? Test with mock data

ratio = 3
x1,x2 = np.linspace(0,50,np.int(ratio*100)), np.linspace(0,50,100)

y1 = 18 + 0.3*x1
y2 = y1[-1]+ 0.15*x2
y = np.concatenate((y1,y2))
x = np.concatenate((x1,x1[-1]+x2))

y_smooth = median_filter(y,size=5)
dy_dx = np.gradient(y_smooth,x)
x = x[np.isfinite(dy_dx)]
y = y[np.isfinite(dy_dx)]
dy_dx = dy_dx[np.isfinite(dy_dx)]

cum_sum = np.cumsum(dy_dx - np.mean(dy_dx))


# step = np.hstack((np.ones(len(x)), -1*np.ones(len(x))))
# cum_sum = np.convolve(dy_dx - np.mean(dy_dx), step, mode='valid')

fig = plt.figure()
ax = plt.subplot(311)
ax2 = plt.subplot(312,sharex=ax)
ax3 = plt.subplot(313,sharex=ax)
ax.invert_yaxis()
fig.suptitle(f'Size ratio = {ratio}')
ax.plot(x,y,'b')

ax2.plot(x,dy_dx,'ro')
ax2.axhline(np.mean(dy_dx),ls=':',c='coral')
ax2.axvline(x[closest(dy_dx,np.mean(dy_dx))],ls=':',c='coral')


ax3.plot(x,cum_sum[:-1])
ax3.axvline(x[closest(y,np.mean(y))],c='r')
ax3.axvline(x[np.argmin(cum_sum)],c='b')

 