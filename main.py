#%%


from AstroPipe.classes import image,SExtractor,AstroGNU,directories
from AstroPipe.masking import automatic_mask, ds9_region_masking
from AstroPipe.plotting import plot_ellipses 
import AstroPipe.utilities as ut 
from AstroPipe.sbprofile import isophotal_photometry,isophotal_photometry_fix,rectangular_photometry
from AstroPipe.sbprofile import break_estimation



from astropy.io import fits
import numpy as np
import os 
from astropy.table import Table
import astropy.units as u
import glob
from astroquery.sdss import SDSS

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

from astropy.visualization import ImageNormalize, LogStretch

# %matplotlib widget



''' Initialize images to process'''
path = '/Volumes/G-Drive/PhD/Data/AMIGAS'
# path = '/scratch/pmsa/AMIGAS'

# index =  ut.make_parser().parse_args().index
# index = 0


objects = Table.read(os.path.join(path,'Objects.fits'))
index = np.where(objects['Galaxy']=='CIG340')[0][0]

object_name = objects['Galaxy'][index].strip()
file = glob.glob(os.path.join(path,object_name,f'{object_name}*.fit*'))[0]
hdu = np.where(np.array([len(i.shape) for i in fits.open(file)])!=0)[0][0]

ut.check_print(f'Processing {file}...')

file_extension = file.split('.')[-1]
folders = directories(object_name,path=os.path.dirname(file))
folders.set_regions('/Volumes/G-Drive/PhD/Data/AMIGAS/Regions/')


AstroPipe_steps = ['SExtractor','NoiseChisel', 'ds9' , 'FABADA','Photometry_fit',
                    'Photometry_man','Save']


img = image(file, hdu=hdu, zp=objects['Zero_point'][index])



coords = {'ra': objects['RA'][index] , 'dec':  objects['DEC'][index]}   
img.obj(coords["ra"], coords["dec"])


try:
    distance = ut.redshift_to_kpc(objects['redshift'][index])
except:
    print('No redshitf found')

try:
    radius = SDSS.query_crossid(img.SkyCoord,photoobj_fields=['PetroRad_r'])['PetroRad_r'].data[0]//img.pixel_scale
except:
    radius = 140
    ut.check_print('No SDSS data found')

img.set_maglim(objects['Depth'][index])
if  img.maglim == 0:
    img.set_maglim(28.5)

save_plot = True

plot_dict = {'width':ut.kpc_to_arcsec(50,distance)/img.pixel_scale,
            'vmax':img.mu_to_counts(18),
            'vmin':img.mu_to_counts(np.round(np.nanpercentile(objects['Depth'],90)))}

#%% Crop Image to light up process.
width = 50*radius
if width > 7000 or width < 4000:
    width = 7000

if any(width//2 > np.array(img.pix)):
    width = np.array(np.min(img.pix))*2

if not any(width > np.array(img.data.shape)):

    ut.check_print(f'Croping image {img.name}...')
    ut.check_print(f'... from {img.data.shape} to {(width,width)} pixels')

    ut.crop(img, width=[width//2]*2,
            out=os.path.join(folders.out,f'{img.name}_crop.fits'))

width = ut.kpc_to_arcsec(50,distance)/img.pixel_scale
if save_plot: 
    img.show(width=plot_dict['width'],vmin=plot_dict['vmin'],
                            vmax=plot_dict['vmax'])
    plt.savefig(os.path.join(folders.temp,f'{object_name}_crop.jpg'),dpi=200)


#%% PSF

#%% MASKING

'''Create Final Mask '''
folders.set_mask(os.path.join(folders.out,img.name+'_mask.fits'))
if not os.path.exists(folders.mask):

    ut.check_print(f'Creating Mask for {img.name}...')

    automatic_mask(img,folders)

    ut.check_print(f'Manual Masking for {img.name}...')

    ds9_region_masking(img,folders)
    
    masked = img.data.data
    masked[np.where(img.data.mask == True)] = np.nan
    masked_file = os.path.join(folders.out,img.name+'_masked.fits')
    fits.PrimaryHDU(masked,header=img.header).writeto(
            masked_file,overwrite=True)

    ut.check_print(f'Masked image saved in {masked_file}...')


else:
    ut.check_print(f'Reading Mask from {folders.mask}...')
    img.set_mask(1-fits.getdata(folders.mask))
    try:
        img.set_std(fits.getheader(folders.mask)['STD'])
    except:
        ut.check_print(f'Not STD information found from previous mask...')



if save_plot:
    img.show(width=plot_dict['width'],vmin=plot_dict['vmin'],
                            vmax=plot_dict['vmax'])

    plt.savefig(os.path.join(folders.temp,f'{object_name}_masked.jpg'),dpi=200)

    


#%% PROFILE

folders.set_profile(os.path.join(folders.out,img.name+'_profile.fits'))


img.get_morphology(width=ut.kpc_to_arcsec(25,distance)/img.pixel_scale)
ut.check_print(f'Morphology found pa={img.pa} eps={img.eps} r_eff={img.r_eff}...')
ut.check_print(f'Redifinding center from {img.pix}...')
img.pix = ut.find_center(img)
ut.check_print(f'... to {img.pix}')




if save_plot:
    im = img.show(width=plot_dict['width'],vmin=plot_dict['vmin'],
                            vmax=plot_dict['vmax'])
    
    ellipse_patch = patches.Ellipse(img.pix,
                2*img.r_eff,
                2*img.r_eff * (1 - img.eps),
                img.pa,
                color='black',alpha=0.8,fill=False)
    plt.scatter(img.pix[0],img.pix[1],c='b',marker='x',s=20)
    im.axes.add_patch(ellipse_patch)
    plt.savefig(os.path.join(folders.temp,f'{object_name}_morphology.jpg'),dpi=200)

if not os.path.exists(folders.profile):
    ut.check_print(f'Creating Profile for {img.name}...')
    img.get_background(out=os.path.join(folders.temp,f'{img.name}_background.jpg'))
    ut.check_print('Background estimation of {:^e} ({:2.2f} mag*arcsec^2)'.format(
                                img.bkg,img.counts_to_mu(img.bkg)))
    

    profile = isophotal_photometry(img,r_eff=ut.kpc_to_arcsec(10,distance)/img.pixel_scale,
                        zp=img.zp,
                        max_r=ut.kpc_to_arcsec(200,distance)/img.pixel_scale,
                        plot=True,save=folders,fix_center=True)

    morph_index = np.where(np.isfinite(profile['surface_brightness']))[0][-1]
    morph_index = ut.closest(profile['radius'],profile['radius'][morph_index]/2)

    
    img.set_morphology(pa=profile['pa'][morph_index].value,
                    eps=profile['ellipticity'][morph_index])
    ut.check_print(f'Morphology uptdated pa={img.pa} eps={img.eps}')

    profile_fix = isophotal_photometry_fix(img,zp=img.zp,
                        max_r=ut.kpc_to_arcsec(200,distance)/img.pixel_scale,
                        plot=True, save=folders)

    profile_rec = rectangular_photometry(img,zp=img.zp,width=np.int32(10/img.pixel_scale),
                        max_r = ut.kpc_to_arcsec(200,distance)/img.pixel_scale,
                        plot=True, save=folders)
    ut.check_print(f"Profile saved in {folders.profile}...")

else:
    ut.check_print(f'Reading Profile from {folders.profile}...')
    profile = Table.read(folders.profile, 
                units=[u.arcsec]+3*[u.mag/u.arcsec**2]+[u.deg]+[None]+2*[u.deg])
    profile_fix = Table.read(folders.profile.split('.')[-2]+'_fixed.fits', 
                units=[u.arcsec]+3*[u.mag/u.arcsec**2]+[u.deg]+[None]+2*[u.deg])
    profile_rec = Table.read(folders.profile.split('.')[-2]+'_rect.fits', 
                units=[u.arcsec]+3*[u.mag/u.arcsec**2])
    img.set_background(profile.meta['background'])


#%% Remove units

new_profile = {}
for key in profile.columns:
    new_profile[key] = profile[key].value

new_profile_fix = {}
for key in profile_fix.columns:
    new_profile_fix[key] = profile_fix[key].value

new_profile_rec = {}
for key in profile_rec.columns:
    new_profile_rec[key] = profile_rec[key].value


profile = new_profile
profile_fix = new_profile_fix
profile_rec = new_profile_rec

#%% Break Finding 
import pandas as pd 

file = '/users/pmsa/Documents/PhD/Projects/AMIGA/Breaks_stats_radius.csv'

breaks = break_estimation(profile['radius'],profile['surface_brightness'],
            rms=img.bkg, skyrms=img.bkg, rin=20,rout=150,
            zp=img.zp, pixel_scale=img.pixel_scale)


breaks = pd.read_csv(file,delimiter=';')

radial_break = breaks['Radius'][int(np.argwhere(
                    np.array(breaks['Galaxy'].str.contains(
                    img.name.split('_')[0]))))]

if np.isnan(radial_break):
    radial_break = -999

#%% PLOTTING
ut.check_print('Plotting Profile...')
import matplotlib
# from AstroPipe.plotting import plot_ellipses
from astropy.table import Table


save_fig = True
extent = np.array([-img.pix[0],img.data.shape[1]-img.pix[0],
        -img.pix[1],img.data.shape[0]-img.pix[1]])
extent *= img.pixel_scale


# trunc_arg = aaron_break_finder(profile['radius'].value,profile['surface_brightness'].value)
trunc_arg = [ut.closest(profile['radius'],radial_break)]
trunc_aper = patches.Ellipse((0,0),
                2*profile['radius'][trunc_arg[0]],
                2*(profile['radius'][trunc_arg[0]] * (1 - profile['ellipticity'][trunc_arg[0]])),
                profile['pa'][trunc_arg[0]],
                color='darkorange',alpha=0.8,fill=False,lw=1)

fig = plt.figure(figsize=(11.5,4))
ax1 = plt.subplot2grid((5,3),(0,0),rowspan=5)
ax1_2 = plt.subplot2grid((5,3),(0,1),rowspan=5)
ax2 = plt.subplot2grid((5,3),(0,2),rowspan=3)
ax4 = plt.subplot2grid((5,3),(3,2),rowspan=1,sharex=ax2)
ax5 = plt.subplot2grid((5,3),(4,2),rowspan=1,sharex=ax2)

kwargs = {'origin':'lower','cmap':'nipy_spectral',
          'extent':extent,'interpolation':'none',
          'vmin':19,'vmax':28.5}
fontsize=11

fig.suptitle(img.name,fontsize=15)

transparent = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
gray = matplotlib.colors.colorConverter.to_rgba('black',alpha = 0.4)
cmap = matplotlib.colors.ListedColormap([transparent, gray])

ax1.imshow(img.counts_to_mu(img.data.data - img.bkg), **kwargs)
ax1.imshow(img.data.mask, origin='lower',
            cmap=cmap,extent=extent)

ax1.set_xlabel('distance $[arcsec]$',fontsize=fontsize,labelpad=-1)
ax1.set_ylabel('distance $[arcsec]$',fontsize=fontsize,labelpad=-2)

im = ax1_2.imshow(img.counts_to_mu(img.data.data - img.bkg), **kwargs)
ax1_2.imshow(img.data.mask, origin='lower',
            cmap=cmap,extent=extent)

ax1_2 = plot_ellipses(profile, ax=ax1_2, max_r=180, step=5,lw=0.4,alpha=0.9)
ax1_2.set_xlabel('distance $[arcsec]$',fontsize=fontsize,labelpad=-1)
ax1_2.set_yticklabels([])
ax1_2.add_patch(trunc_aper)


ax2.plot(profile_fix['radius'],profile_fix['surface_brightness'],'g-',label='Ellip-Aper')
ax2.fill_between(profile_fix['radius'],profile_fix['surface_brightness']+profile_fix['sb_err_upper'],
                profile_fix['surface_brightness']-profile_fix['sb_err_low'],color='g',alpha=0.5)

ax2.plot(profile_rec['radius'],profile_rec['surface_brightness'],'c-',label='Rect-Aper')
ax2.fill_between(profile_rec['radius'],profile_rec['surface_brightness']+profile_rec['sb_err_upper'],
                profile_rec['surface_brightness']-profile_rec['sb_err_low'],color='c',alpha=0.5)

ax2.plot(profile['radius'],profile['surface_brightness'],'r-',label='Astropy')
ax2.fill_between(profile['radius'], 
        profile['surface_brightness'] - profile['sb_err_low'],
        profile['surface_brightness'] + profile['sb_err_upper'],
        alpha=0.4,color='r')

for arg in trunc_arg:
    ax2.axvline(profile['radius'][arg],color='darkorange',ls=':',label='Break')

ax2.set_ylabel('$\mu [mag*arcsec^{-2}]$',fontsize=fontsize,labelpad=-1)
ax2.legend(fontsize=10)
ax2.invert_yaxis()

ax2.set_ylim([np.nanmax([np.nanmax(profile['surface_brightness']),img.maglim+1.5]),
    np.nanmin(profile['surface_brightness'])-0.5])
ax2.set_xlim([-10,profile['radius'][ut.closest(profile['surface_brightness'],ax2.get_ylim()[0])]+10])


bar_ax = ax2.inset_axes([ -7, kwargs['vmin'], 6, kwargs['vmax']-kwargs['vmin']],
            transform=ax2.transData)
bar = fig.colorbar(im, cax=bar_ax)
bar_ax.axis('off')
bar_ax.invert_yaxis()


ax4.plot(profile['radius'],profile['pa'],'r-')
ax4.axhline(img.pa,c='indianred',ls=':')
ax4.set_ylabel('PA [deg]',fontsize=fontsize-1,labelpad=-1)

ax5.plot(profile['radius'],profile['ellipticity'],'r-')
ax5.axhline(img.eps,c='indianred',ls=':')
ax5.set_ylabel('Eps',fontsize=fontsize-1,labelpad=-1)
ax5.set_xlabel('R [$arcsec$]',fontsize=fontsize,labelpad=-1)    


ax6 = ax2.twiny()

ax6.plot(profile['radius']*np.pi*distance/(3600*180),
        profile['surface_brightness'],alpha=0)
ax6.set_xlabel('Radius [kpc]',fontsize=fontsize-1,labelpad=2)


ax1.set_xlim([-round(ax2.get_xlim()[1]/10+1)*10,round(ax2.get_xlim()[1]/10+1)*10])
ax1.set_ylim([-round(ax2.get_xlim()[1]/10+1)*10,round(ax2.get_xlim()[1]/10+1)*10])
ax1_2.set_xlim([-round(ax2.get_xlim()[1]/10+1)*10,round(ax2.get_xlim()[1]/10+1)*10])
ax1_2.set_ylim([-round(ax2.get_xlim()[1]/10+1)*10,round(ax2.get_xlim()[1]/10+1)*10])

for ax in [ax2,ax4,ax5]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(labelsize=10)

for ax in [ax2, ax4]:
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_yticks(ax.get_yticks()[1:]) 

for ax in (ax1,ax1_2,ax2,ax4,ax5,ax6):
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)

plt.subplots_adjust(top=0.895,
bottom=0.1,
left=0.035,
right=0.95,
hspace=0.3,
wspace=0.0)

# plt.tight_layout()

if save_fig:
    fig.savefig(os.path.join(folders.out,object_name+'_photometry.jpg'),dpi=200)

x_lim = ax2.get_xlim()
plt.show()
#%% Aaron Watkins Break Finder
'''
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
            profile['sb_err_low']],axis=0)
rad = profile['radius']

index = ut.where([mu>min,np.isfinite(mu),mu<max])[0]

# mu = median_filter(mu[index],size=int(5*len(mu)/100))
# rad_new = np.linspace(np.nanmin(rad),np.nanmax(rad),500)
# mu_new = np.interp(rad_new,rad[index],mu)
# index = np.ones_like(mu_new,dtype=bool)

# rad = rad_new
# mu = mu_new


n,p,min,max = 4,5,21,30.5


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

#%% Convolution with step function


step = np.zeros(10)
step[len(step)//2:] = 1
 
conv = np.convolve(der,step,mode='same')

plt.figure()
plt.plot(rad[index],conv,'r-')



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

 '''



