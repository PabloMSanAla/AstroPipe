#%%

from AstroPipe.classes import Image,SExtractor,AstroGNU,Directories
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
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

# %matplotlib widget



''' Initialize images to process'''


path = '/Volumes/G-Drive/PhD/Data/AMIGAS-dejar'
objects_file = '/Users/pmsa/Documents/PhD/Projects/AMIGA/Objects.fits'
breakFile = '/Users/pmsa/Documents/PhD/Projects/AMIGA/Breaks_stats_radius.csv'


# path = '/scratch/pmsa/AMIGAS'
# objects_file = '/scratch/pmsa/AMIGAS/Objects.fits'
# breakFile = '/scratch/pmsa/AMIGAS/Breaks_stats_radius.csv'

# index =  ut.make_parser().parse_args().index
# index = 0


objects = Table.read(objects_file)
index = np.where(objects['Galaxy']=='CIG947')[0][0]

object_name = objects['Galaxy'][index].strip()
file = glob.glob(os.path.join(path,object_name,f'{object_name}*.fit*'))[0]

# file = glob.glob(os.path.join(path,f'{object_name}*.fit*'))[0]

hdu = np.where(np.array([len(i.shape) for i in fits.open(file)])!=0)[0][0]

ut.check_print(f'Processing {file}...')

file_extension = file.split('.')[-1]
folders = Directories(object_name,path=os.path.dirname(file))
folders.set_regions('/Volumes/G-Drive/PhD/Data/AMIGAS/Regions/')


#Steps: Crop, Masking, PSF, Photometry, Break, Plot
AstroPipe_steps = []


img = Image(file, hdu=hdu, zp=objects['Zero_point'][index])



coords = {'ra': objects['RA'][index] , 'dec':  objects['DEC'][index]}   
img.obj(coords["ra"], coords["dec"])

img.set_extinction(objects['Av(rSDSS)'][index])
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

img.bkg = 0
# AstroPipe_steps += ['Photometry'] 
if not os.path.exists(folders.profile) or ('Photometry' in AstroPipe_steps):
    ut.check_print(f'Creating Profile for {img.name}...')
    img.get_background(out=os.path.join(folders.temp,f'{img.name}_background.jpg'))
    ut.check_print('Background estimation of {:^e} ({:2.2f} mag*arcsec^2)'.format(
                                img.bkg,img.counts_to_mu(img.bkg)))
    

    profile = isophotal_photometry(img,r_eff=ut.kpc_to_arcsec(10,distance)/img.pixel_scale,
                        zp=img.zp,
                        max_r=np.ceil(img.bkg_radius/100)*100,
                        plot=True,save=folders,fix_center=True)

    morph_index = np.where((profile['radius'].value>ut.kpc_to_arcsec(5,distance))&(
                            (profile['radius'].value<ut.kpc_to_arcsec(20,distance))))
    
    img.set_morphology(pa=np.mean(profile['pa'][morph_index].value),
                    eps=np.mean(profile['ellipticity'][morph_index]))
    ut.check_print(f'Morphology uptdated pa={img.pa} eps={img.eps}')

    profile_fix = isophotal_photometry_fix(img,zp=img.zp,
                        max_r=np.ceil(img.bkg_radius/100)*100,
                        plot=True, save=folders)

    profile_rec = rectangular_photometry(img,zp=img.zp,width=np.int32(10/img.pixel_scale),
                        max_r=np.ceil(img.bkg_radius/100)*100,
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

breaks = Table.read(breakFile,format='ascii.csv',delimiter=';')
break_index = np.where(breaks['Galaxy']==img.name.split('_')[0])


# Revisar  1047

if 'Break' in AstroPipe_steps:
    rin = profile['radius'][np.where(profile['surface_brightness']>20.5)[0][0]]
    rout = profile['radius'][np.where(profile['surface_brightness']>25)[0][0]]
    auto_breaks = break_estimation(profile['radius'],profile['surface_brightness'],
                rms=img.bkg, skyrms=img.bkg, rin=rin,rout=rout,
                zp=img.zp, pixel_scale=img.pixel_scale)

    for b,column in zip(auto_breaks,breaks.columns[2:]):
        breaks[column][np.where(breaks['Galaxy']==img.name.split('_')[0])].mask = False
        breaks[column][np.where(breaks['Galaxy']==img.name.split('_')[0])] = np.int32(b)

    breaks.write(breakFile,overwrite=True,format='ascii.csv',delimiter=';')

radial_break = breaks['Radius'][break_index][0]

if np.isnan(radial_break):
    radial_break = -999

 #%% PLOTTING
import matplotlib

from fabada import fabada
from astropy.stats import sigma_clipped_stats


manual_limits =  {'CIG154':100,
                  'CIG279':200,
                  'CIG329':240,
                  'CIG568':123,
                  'CIG772':139,
                  'CIG947':549,
                  'CIG971':300}

run_fabada = '613' in img.name
save_fig = True

extent = np.array([-img.pix[0],img.data.shape[1]-img.pix[0],
        -img.pix[1],img.data.shape[0]-img.pix[1]])
extent *= img.pixel_scale

r25 = profile['radius'][ut.closest(profile['surface_brightness'],25)]
alpha25 = 2.7


nans = np.isnan(profile['surface_brightness'].data)
if nans.any():
    index_sb =  ut.closest(alpha25*r25,profile['radius'][nans])
    index_limit = np.where(nans)[0][index_sb]
    index_offset = index_sb+1
else:
    index_limit = len(profile['radius']) -1 
    index_offset = 0

if img.name.split('_')[0] in  manual_limits.keys(): 
    index_limit = ut.closest(profile['radius'],manual_limits[img.name.split('_')[0]])
    index_offset = 0

force_limits = {'x': profile['radius'][index_limit]+np.diff(profile['radius'])[index_limit],
                'y': profile['surface_brightness'][index_limit-index_offset]+0.5}

force_limits = {'x': profile['radius'][-1],
                'y': profile['surface_brightness'][np.isfinite(profile['surface_brightness'])][-1]+0.3}
# force_limits = True

mean,median,std = sigma_clipped_stats(img.data.data,mask=img.data.mask,sigma=2.5)

if not run_fabada: mu_image = img.counts_to_mu(img.data.data - img.bkg - mean) - img.Av
if run_fabada:  
    smooth = fabada(img.data.data-img.bkg - mean,(1.3*std)**2,verbose=True)
    mu_image = img.counts_to_mu(smooth) - img.Av
    mean,median,std = sigma_clipped_stats(smooth,mask=img.data.mask,sigma=2.5)

kwargs = {'origin':'lower','cmap':'nipy_spectral',
        'extent':extent,'interpolation':'none',
        'vmin':profile['surface_brightness'][~np.isnan(profile['surface_brightness'])][0],
        # 'vmax':np.nanpercentile(mu_image,75)}
        'vmax':np.ceil(img.counts_to_mu(std))+0.3}


ut.check_print('Plotting Profile...')


# trunc_arg = aaron_break_finder(profile['radius'].value,profile['surface_brightness'].value)

for i in range(2):
    trunc_arg = [ut.closest(profile['radius'],radial_break)]
    trunc_aper = patches.Ellipse((0,0),
                2*profile['radius'][trunc_arg[0]],
                2*(profile['radius'][trunc_arg[0]] * (1 - profile['ellipticity'][trunc_arg[0]])),
                profile['pa'][trunc_arg[0]],
                color='black',ls='--',alpha=0.8,fill=False,lw=2)

    fig = plt.figure(figsize=(11.5,4))
    ax1 = plt.subplot2grid((5,3),(0,0),rowspan=5)
    ax1_2 = plt.subplot2grid((5,3),(0,1),rowspan=5)
    ax2 = plt.subplot2grid((5,3),(0,2),rowspan=3)
    ax4 = plt.subplot2grid((5,3),(3,2),rowspan=1,sharex=ax2)
    ax5 = plt.subplot2grid((5,3),(4,2),rowspan=1,sharex=ax2)

    
    nans =  np.isnan(profile['surface_brightness'])
    if not nans.any():
        index_lim = len(profile['surface_brightness'])-1
    else:
        index_lim = np.where(nans)[0][0]-1

    fontsize=11

    transparent = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
    gray = matplotlib.colors.colorConverter.to_rgba('black',alpha = 0.4)
    cmap = matplotlib.colors.ListedColormap([transparent, gray])

    ax1.imshow(mu_image, **kwargs)


    ax1.set_xlabel('distance [arcsec]',fontsize=fontsize,labelpad=-1)
    ax1.set_ylabel('distance [arcsec]',fontsize=fontsize,labelpad=-2)

    if '613' in img.name:
        ax1.arrow(-30,-85,0,20,head_width=8,width=2,fc='k')
        ax1.arrow(15,120,-15,-25,head_width=8,width=2,fc='k')

    ax1.text(0.99,1.05,img.name.split('_')[0].replace('CIG','CIG '),fontsize=13,
        transform = ax1.transAxes,ha='right',va='top',fontweight='bold')

    im = ax1_2.imshow(mu_image, **kwargs)
    ax1_2.imshow(img.data.mask, origin='lower',
                cmap=cmap,extent=extent)

    ax1_2.set_xlabel('distance [arcsec]',fontsize=fontsize,labelpad=-1)
    ax1_2.set_yticklabels([])

    if i==1:
        ax2.plot(profile_fix['radius'],profile_fix['surface_brightness']-img.Av,'g-',label='Fixed $\\epsilon$, PA',alpha=0.6)
        ax2.fill_between(profile_fix['radius'],profile_fix['surface_brightness']-img.Av+profile_fix['sb_err_upper'],
                        profile_fix['surface_brightness']-img.Av-profile_fix['sb_err_low'],color='g',alpha=0.4)

        ax2.plot(profile_rec['radius'],profile_rec['surface_brightness']-img.Av,'c-',label='Rectangular',alpha=0.6)
        ax2.fill_between(profile_rec['radius'],profile_rec['surface_brightness']-img.Av+profile_rec['sb_err_upper'],
                        profile_rec['surface_brightness']-img.Av-profile_rec['sb_err_low'],color='c',alpha=0.4)

    ax2.plot(profile['radius'],profile['surface_brightness']-img.Av,'r-',label='Dynamic $\\epsilon$, PA',lw=2)
    ax2.fill_between(profile['radius'], 
            profile['surface_brightness']-img.Av - profile['sb_err_low'],
            profile['surface_brightness']-img.Av + profile['sb_err_upper'],
            alpha=0.4,color='r')

    if radial_break != -999:
        ax2.axvline(profile['radius'][trunc_arg],color='black',ls='--',label='Break')

    if i==1:
        ax2.legend(fontsize=8,loc='upper right',frameon=False,  bbox_to_anchor=(0.90, 0.95))
    ax2.set_ylabel('$\mu\,[\mathrm{mag\,arcsec}^{-2}]$',fontsize=fontsize,labelpad=3)
    ax2.invert_yaxis()

    ax4.plot(profile['radius'],profile['pa'],'r-')
    ax4.axhline(img.pa,c='indianred',ls=':')
    ax4.set_ylabel('PA [deg]',fontsize=fontsize-1,labelpad=3)
    ax4.set_ylim(np.floor(np.nanmin(profile['pa'])),np.ceil(np.nanmax(profile['pa'])))
    ax4.locator_params(axis='y',nbins=2)

    ax5.plot(profile['radius'],profile['ellipticity'],'r-')
    ax5.axhline(img.eps,c='indianred',ls=':')
    ax5.set_ylabel('$\\epsilon$',fontsize=fontsize-1,labelpad=3)
    ax5.set_xlabel('radius [arcsec]',fontsize=fontsize,labelpad=-1)    
    ax5.set_ylim(np.floor(np.nanmin(profile['ellipticity'])),np.ceil(np.nanmax(profile['ellipticity'])))
    ax5.locator_params(axis='y',nbins=2)

    ax6 = ax2.twiny()

    ax6.plot(profile['radius']*np.pi*distance/(3600*180),
            profile['surface_brightness'],alpha=0)
    ax6.set_xlabel('radius [kpc]',fontsize=fontsize-1,labelpad=2)


    # ax1.set_xlim([-round(x_lim[1]/10+1)*10,round(x_lim[1]/10+1)*10])
    # ax1.set_ylim([-round(x_lim[1]/10+1)*10,round(x_lim[1]/10+1)*10])
    # ax1_2.set_xlim([-round(x_lim[1]/10+1)*10,round(x_lim[1]/10+1)*10])
    # ax1_2.set_ylim([-round(x_lim[1]/10+1)*10,round(x_lim[1]/10+1)*10])
    
    
    # R25 limit
    for ax in [ax1,ax1_2]:
        for lim in [ax.set_xlim,ax.set_ylim]:
            lim(-alpha25*r25,alpha25*r25)


    ax1_2 = plot_ellipses(profile, ax=ax1_2, max_r=force_limits['x']*1.1, step=5,lw=0.6,alpha=0.9)
    if radial_break !=-999: ax1_2.add_patch(trunc_aper)


    bar_ax = ax2.inset_axes([0.9*force_limits['x'], kwargs['vmin'], 0.05*force_limits['x'], kwargs['vmax']-kwargs['vmin']],
                transform=ax2.transData)
    bar = fig.colorbar(im, cax=bar_ax)
    bar_ax.axis('off')
    bar_ax.invert_yaxis()

    for ax in [ax2,ax4,ax5]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(labelsize=8)

    for ax in [ax2, ax4]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:]) 

    # Set tick size
    for ax in (ax1,ax1_2,ax2,ax4,ax5,ax6):
        ax.yaxis.set_tick_params(labelsize=10)
        ax.xaxis.set_tick_params(labelsize=10)
    
     
    if force_limits:
        ax2.set_ylim([force_limits['y'],np.nanmin(profile['surface_brightness'])-0.5])
        ax2.set_xlim([-force_limits['x']//30,force_limits['x']])
    ax6.set_xlim(np.multiply(ax2.get_xlim(),np.pi*distance/(3600*180)))

    plt.subplots_adjust(top=0.915,
    bottom=0.09,
    left=0.045,
    right=0.95,
    hspace=0.3,
    wspace=0.0)

    # plt.tight_layout()

    if save_fig:
        fig.savefig(os.path.join(folders.out,object_name+'_photometry'+'_all'*i+'.jpg'),dpi=300)


    # plt.show()
#%% Aaron Watkins Break Finder
'New sky background subtraction method'

from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from astropy.stats import sigma_clipped_stats, sigma_clip
from AstroPipe.sbprofile import find_mode, derivative

from AstroPipe.plotting import show
import scipy.stats as stats 


# Find radius where an asintote starts in a profile
def find_radius_asintote(x,y,eps):
    # Find the first point where the slope of the line is 0
    # This is the point where the asintote starts
    # x,y are the profile
    # Returns the radius where the asintote starts

    # Find the slope of the line between each point
    slope = derivative(x,y)

    # Find the first point where the slope is 0
    # This is the point where the asintote starts
    # This is the point where the slope changes sign
    sign_change = np.where(np.diff(np.sign(slope)))[0]

    #If there is no sign change, return the point with the closes slope to eps
    if len(sign_change) == 0:
        return x[np.nanargmin(np.abs(slope - eps))]
    else:
        # Return the radius where the asintote starts
        return x[sign_change]

def res_sum_squares(dmdr, cog, slope, abcissa):

    y2 = abcissa+slope*dmdr    
    rms = np.mean(np.sqrt((cog-y2)**2))

    return rms, y2 

def asymtotic_fit_radius(x,y,eps=1e-5):
    '''
    Find the asymptotic radius of a profile
    fitting a line and finding the intersection with the y axis
    '''
    xx = np.array(x)
    yy = np.array(y)
    want = xx == xx
    for i in range(3):
        xx = xx[want]
        yy = yy[want]
        fit = stats.linregress(xx, yy)
        rms, y2 = res_sum_squares(xx, yy, fit[0], fit[1])  # This was a custom function I wrote
        want = np.abs(yy - y2) <= 3*rms
    
    return -fit[1]/fit[0]


IMG = img.copy()
pa =  IMG.pa * np.pi/180
eps = IMG.eps 
center=IMG.pix
zp=IMG.zp
rad = None ; max_r = 3000; growth_rate = 1.03
plot=True
if rad is None:
    rad = np.ones(1)

data = IMG.data 
flatten = IMG.data[IMG.data.mask==False].flatten()
flatten = flatten[np.isfinite(flatten)]
mode, results = find_mode(flatten)
std = results.values['sigma']

intensity = np.zeros(0)
intensity_std = np.zeros(0)
ellip_apertures = []
previous_mask = np.zeros_like(IMG.data)
converge = False
maxr,stability = np.NaN, 0


while not converge:
    
    if len(ellip_apertures) > 1:
        previous_mask = mask
    ellip_apertures.append(EllipticalAperture((center[0],center[1]), rad[-1], (1-eps)*rad[-1], pa))
    mask = ellip_apertures[-1].to_mask(method='center').to_image(data.shape)

    index = ut.where([data.mask==False,mask!=0,previous_mask==0])
    clipped = sigma_clip(data.data[index],sigma=3,maxiters=3)
    intensity= np.append(intensity, np.ma.median(clipped))
    intensity_std = np.append(intensity_std, np.nanstd(clipped)/np.sqrt(np.size(clipped)))
    
    # print(f'maxr = {maxr:.1f}; rad={rad[-1]:.2f}; intesity={intensity[-1]:.2f}', end=',')
    
    if intensity[-1] < mode + std and np.isnan(maxr):
        index = intensity < mode + std
        dIdr = derivative(rad[index],intensity[index])
        if (any(np.sign(dIdr[1:]/dIdr[:-1]) == -1
             ) or (np.abs(intensity[-1]/mode - 1) < 1e-1)
            ) and np.isnan(maxr):
            stability += 1
            if stability > 5:
                maxr = asymtotic_fit_radius(rad[index],dIdr,eps=mode+1)
                print(f'maxr={maxr:.2f}; rad={rad[-1]:.2f}; intesity={intensity[-1]:.2f}')
        else: stability = 0
    if rad[-1] > 1.1*maxr:
        converge = True
        print(maxr,len(rad),len(intensity))
        break

    rad = np.append(rad, rad[-1]*growth_rate)

#%%

cumintensity = np.nancumsum(intensity)
dcumint = derivative(rad,cumintensity)
index = np.where(cumintensity > cumintensity[-1]/2)[0]
maxr = asymtotic_fit_radius(rad[index],cumintensity[index],eps=mode+std)

if plot:
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,5), sharex=True)
    ax1.plot(rad,cumintensity)
    ax2.plot(rad,dcumint)
    ax1.set_xscale('log'); ax2.set_xscale('log') 
    ax1.axvline(maxr,ls='--',color='r')

firstd = derivative(rad,intensity)
secondd = derivative(rad,firstd)

index = intensity < mode + std
skyradius1 = asymtotic_fit_radius(rad[index],secondd[index])
skyradius2 = asymtotic_fit_radius(rad[index],firstd[index])

skyradii = np.sort([skyradius1,skyradius2])
aperfactor = np.nanmax([0.01,float(np.round((60 - np.diff(skyradii)) / (np.sum(skyradii)),3))])

width = float(np.nanmax([np.diff(skyradii),60]))

bkg_aperture = EllipticalAnnulus((IMG.pix[0],IMG.pix[1]),
                     (1-aperfactor)*skyradii[0], (1+aperfactor)*skyradii[1], 
                    (1-0.6*IMG.eps)*(1-aperfactor)*skyradii[0], None,
                    IMG.pa*np.pi/180)

mask_aper = bkg_aperture.to_mask(method='center').to_image(IMG.data.shape)
mask_aper = np.ma.array(mask_aper,mask=1-mask_aper)
aper_values = IMG.data*mask_aper
aper_values = aper_values[np.where(~aper_values.mask)].flatten()
localsky, gauss_fit = find_mode(aper_values)



out = os.path.join(folders.temp,f'{img.name}_background.jpg')
if out: 
    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot2grid((3,3),(0,0))
    ax2 = plt.subplot2grid((3,3),(1,0),sharex=ax1)
    ax3 = plt.subplot2grid((3,3),(2,0),sharex=ax1)
    ax4 = plt.subplot2grid((3,3),(0,1),rowspan=3,colspan=2)
    
    fontsize = 12
    ax1.plot(rad,intensity,'.', label='Flux')
    ax1.set_ylabel('Intensity (ADUs)',fontsize=fontsize)
    ax1.axhline(mode,ls='--',c='k',label='Mode')
    ax1.axhline(mode-std,ls=':',c='k',label='Mode $\pm \sigma$')
    ax1.axhline(mode+std,ls=':',c='k')
    ax1.set_ylim([mode-std*0.5,mode+std*1.4])
    arglim = np.nanargmin(np.abs(intensity-mode-std))
    ax1.set_xlim([rad[arglim],1.1*np.nanmax(np.append(rad[-1],skyradii))])

    ax2.plot(rad,firstd,'.')
    ax2.set_ylabel('$dI/dr$',fontsize=fontsize)
    ax2.axhline(0,ls='--',c='k')
    ax2.axvline(skyradius2,ls='--',c='r',label='Sign change')
    ax2.set_ylim([firstd[arglim],-firstd[arglim]/10])

    ax3.plot(rad,secondd,'.')
    ax3.set_ylabel('$d^2I/dr^2$',fontsize=fontsize)
    ax3.set_xlabel('Radius (pixels)',fontsize=fontsize)
    ax3.axhline(0,ls='--',c='k')
    ax3.axvline(skyradius1,ls='--',c='r',label='Sign change')
    ax3.set_ylim([-secondd[arglim]/10,secondd[arglim]])

    for ax in [ax1,ax2,ax3]:
        ax.axvline(bkg_aperture.a_in,ls='-.',c='magenta',label='Sky annulus')
        ax.axvline(bkg_aperture.a_out,ls='-.',c='magenta')   


    show(IMG.data - 2*localsky, vmin=-localsky, mask=True, ax=ax4)
    imwidth=skyradii[0]+width//2
    ax4.set_xlim([center[0]-imwidth,center[0]+imwidth])
    ax4.set_ylim([center[1]-imwidth,center[1]+imwidth])
    ax4.text(0.02, 1, os.path.basename(os.path.splitext(out)[0]), horizontalalignment='left',
            verticalalignment='bottom', transform=ax4.transAxes, fontweight='bold',fontsize='large')
    bkg_aperture.plot() 
    ax1.axhline(localsky,ls='-.',c='magenta')
    ax4.text(1, 1, f'localsky={localsky:.3e}', horizontalalignment='right',
                verticalalignment='bottom', transform=ax4.transAxes,fontsize='large',color='magenta')
    plt.tight_layout()


# %%

fig,ax = plt.subplots(1,1,figsize=(12,8))
for sky in [-1.65, -0.3,-0.5, -0.65, -0.72, -0.82, -0.9, 0.4]:
    ax.plot(rad, IMG.zp - 2.*np.log10((intensity-sky)/IMG.pixel_scale**2),label=f'{np.round(sky,2)}')
ax.legend(ncol=4)
ax.invert_yaxis()
ax.set_xlabel('Radius (pixels)')
ax.set_ylabel('Magnitude')
ax.set_title(f'Changes due to sky subtraction mode={mode:.2f} local={localsky:.2f} $\sigma$={std:.2f}')
plt.tight_layout()



#%%  Break errors

breaks = Table.read(breakFile,format='ascii.csv',delimiter=';')

for i,gname in enumerate(breaks['Galaxy']):
    profileFile = glob.glob(f'/Volumes/G-Drive/PhD/Data/AMIGAS/{gname}/AstroPipe_{gname}/{gname}*profile.fits')[0]
    profile = Table.read(profileFile)
    if np.isfinite(breaks['Radius'][i]):
        argument = np.nanargmin(np.abs(profile['radius']-breaks['Radius'][i]))
        sb = profile['surface_brightness'][argument]
        error_bright = (profile['surface_brightness'][argument+1] - profile['surface_brightness'][argument-1])/2
        error_radius = (profile['radius'][argument+1] - profile['radius'][argument-1])/2
        print(f'{gname}: {sb:.2f} +- {error_bright:.1f}; {breaks["Radius"][i]} +- {np.ceil(error_radius):.0f}')


