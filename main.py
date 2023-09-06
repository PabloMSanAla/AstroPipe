#%%

from AstroPipe.classes import Image, SExtractor,AstroGNU,Directories
from AstroPipe.masking import automatic_mask, ds9_region_masking
from AstroPipe.plotting import plot_ellipses 
import AstroPipe.utilities as ut 
from AstroPipe.profile import isophotal_photometry,elliptical_radial_profile,rectangular_photometry
from AstroPipe.profile import break_estimation


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


path = '/Volumes/G-Drive/PhD/Data/backup/AMIGAS'
objects_file = '/Users/pmsa/Documents/PhD/Projects/AMIGA/Objects.fits'
breakFile = '/Users/pmsa/Documents/PhD/Projects/AMIGA/Breaks_stats_radius.csv'


# path = '/scratch/pmsa/AMIGAS'
# objects_file = '/scratch/pmsa/AMIGAS/Objects.fits'
# breakFile = '/scratch/pmsa/AMIGAS/Breaks_stats_radius.csv'

index =  ut.make_parser().parse_args().index

objects = Table.read(objects_file)
# index = np.where(objects['Galaxy']=='CIG1047')[0][0]

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

# width = 50*radius
# if width > 7000 or width < 4000:
#     width = 7000

# if any(width//2 > np.array(img.pix)):
#     width = np.array(np.min(img.pix))*2

# if not any(width > np.array(img.data.shape)):

#     ut.check_print(f'Croping image {img.name}...')
#     ut.check_print(f'... from {img.data.shape} to {(width,width)} pixels')

#     img.data,img.header = ut.crop(img.data, img.header, img.pix,(width,width),
#             out=os.path.join(folders.out,f'{img.name}_crop.fits'))

if os.path.isfile(os.path.join(folders.out,f'{img.name}_crop.fits')):
    img.data,img.header = fits.getdata(os.path.join(folders.out,f'{img.name}_crop.fits'),header=True)
    img.pix = np.divide(img.data.shape,2)

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


# img.get_morphology(width=ut.kpc_to_arcsec(25,distance)/img.pixel_scale)
img.get_morphology()
img.r_eff = img.reff
ut.check_print(f'Morphology found pa={img.pa} eps={img.eps} r_eff={img.r_eff}...')
ut.check_print(f'Redifinding center from {img.pix}...')
img.pix = ut.find_center(img.data, img.pix )
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

def find_radius(radius, profile, brightness):
    arg = ut.closest(profile,brightness)
    rad, prof = radius[arg-10:arg+10], profile[arg-10:arg+10]
    x = np.linspace(rad[0],rad[-1],100)
    y = np.interp(x,rad,prof)
    return x[ut.closest(y,brightness)]


manual_limits =  {'CIG11':133, 'CIG33': 288,
                  'CIG59':178, 'CIG94': 98,
                  'CIG96': 358, 'CIG100': 73,
                  'CIG154':100, 'CIG279':200,
                  'CIG329':240, 'CIG335':158,
                  'CIG340':179, 'CIG512':188,
                  'CIG568':123, 'CIG613':133,
                  'CIG616':179, 'CIG626':233,
                  'CIG744':163, 'CIG772':153,  
                  'CIG800':218, 'CIG838':104,
                  'CIG947':549, 'CIG971':300,
                  'CIG1002':77, 'CIG1004':233,
                  'CIG1047':88}

run_fabada = '613' in img.name
save_fig = True

extent = np.array([-img.pix[0],img.data.shape[1]-img.pix[0],
        -img.pix[1],img.data.shape[0]-img.pix[1]])
extent *= img.pixel_scale


r25 = find_radius(profile['radius'],profile['surface_brightness'],25)
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

# force_limits = {'x': profile['radius'][index_limit]+np.diff(profile['radius'])[index_limit],
#                 'y': profile['surface_brightness'][index_limit-index_offset]+0.5}

force_limits = {'x': manual_limits[img.name.split('_')[0]],
                'y': 32}

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
        'vmax':np.ceil(img.counts_to_mu(std))+0.4}


ut.check_print('Plotting Profile...')


# trunc_arg = aaron_break_finder(profile['radius'].value,profile['surface_brightness'].value)

for i in range(2):
    trunc_arg = [ut.closest(profile['radius'],radial_break)]
    trunc_aper = patches.Ellipse((0,0),
                2*radial_break,
                2*(radial_break * (1 - profile['ellipticity'][trunc_arg[0]])),
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
        ha = 'right' if r25>radial_break else 'left'
        sign = -1 if r25>radial_break else 1
        ax2.axvline(profile['radius'][trunc_arg],color='black',ls='--')
        ax2.text(profile['radius'][trunc_arg] + sign*0.3,31,'break',
                  rotation=90, fontsize=8,va='bottom', ha=ha)
       
        if np.abs(r25-radial_break) < 20:
            if ha == 'right': ha = 'left'
            else: ha = 'right'
            sign *=1.3
        else: 
            ha = 'right' if r25>radial_break else 'left'
            sign = 1 if r25<radial_break else -1

        ax2.axvline(r25, color='black',ls=':')
            
        ax2.text(r25 - sign*0.3,31,'R$_{25}$',
                  rotation=90, fontsize=9,va='bottom',ha=ha)

    if i==1:
        ax2.legend(ncol=1, fontsize=7, bbox_to_anchor=(0.90, 0.95),frameon=False,)

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
            lim(-alpha25*r25*1.2,alpha25*r25*1.2)


    ax1_2 = plot_ellipses(profile, ax=ax1_2, max_r=force_limits['x'], step=5,lw=0.6,alpha=0.9)
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

    ax2.set_ylim([33,16])
    ax2.set_yticks([17,22,27,32])
    
    ax4.set_ylim(-15,190)
    ax4.set_yticks([0,90,180])

    ax5.set_ylim(-0.1,1.1)
    ax5.set_yticks([0,0.5,1])

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