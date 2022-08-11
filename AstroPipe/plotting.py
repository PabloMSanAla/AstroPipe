import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
import astropy.visualization as vis
import matplotlib.patches as patches
from astropy.table import Table

matplotlib.rcParams['figure.figsize'] = (10,7)
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.major.size']=10
matplotlib.rcParams['xtick.major.size']=10
matplotlib.rcParams['ytick.minor.size']=5
matplotlib.rcParams['xtick.minor.size']=5
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['legend.fancybox'] = True
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['axes.titlesize']=16


def gaussian(x, mu, var, A=1):
    exp = np.exp(-((x - mu) ** 2) / (2 * var))
    norm = A / np.sqrt(2 * np.pi * var)
    return norm * exp
#%%
def noise_hist(result,out=False):
    
    fig = plt.figure(figsize=(9,6))
    ax = result.plot_fit()
    ax.set_title('Noise Distribution '
        r'$\sim \mathcal{N}($'+'$\mu={:.2e}$,$\sigma={:.3e})$'.format(result.values['center'],
                                                                      result.values['sigma']))
    ax.set_xlim([result.values['center']-6*result.values['sigma'],
                result.values['center']+6*result.values['sigma']])
    ax.set_xlabel('Counts values (ADUs)')
    ax.set_ylabel('# of pixels')
    plt.legend()
    plt.tight_layout()
    if out:
        fig.savefig(out,dpi=300)
    
def counts_to_mu(counts,zp, pixel_scale):
    return zp - 2.5*np.log10(counts/pixel_scale**2)

def mu_to_counts(mu, zp, pixel_scale):
    return 10**((zp-mu)/2.5)*pixel_scale**2


def show(image,vmin=None,vmax=None,zp=None, pixel_scale=1,
            cmap='nipy_spectral_r',wcs=None,nan='cyan'):
    
    image = np.array(image)
    if not zp: 
        if not vmin: vmin = np.nanpercentile(image[np.where(image>0)],0.01)
        if not vmax: vmax = np.nanpercentile(image,99)
    else:
        if not vmin: vmin = counts_to_mu(18.5,zp,pixel_scale)
        if not vmax: vmax = counts_to_mu(28.5,zp,pixel_scale)
    
    norm = vis.ImageNormalize(image,vmin=vmin,vmax=vmax,stretch=vis.LogStretch())

    if len(np.shape(image)) == 2:
        fig,ax = plt.subplots(1,1)
        im = ax.imshow(image,norm=norm,
        cmap=cmap,origin='lower',interpolation='none')
        if not zp: fig.colorbar(im)
        if zp:
            mmax,mmin = counts_to_mu(np.array([vmin,vmax]),22.5,0.33)
            bar = fig.colorbar(im,ticks=mu_to_counts(np.arange(mmax,mmin,2.5),zp,pixel_scale))
            ticklabels = 5*np.round((zp-2.5*np.log10(bar.get_ticks()))*2)/10
            bar.set_ticklabels(['{:2.1f}'.format(i) for i in ticklabels])

        
    else:
        fig,ax=plt.subplots(figsize = (6*len(image),6), nrows=1, 
                            ncols=len(image),sharex=True,sharey=True)
        plt.subplots_adjust(top=0.85,bottom=0.12,left=0.10,
                            right=0.978,hspace=0.205,wspace=0.038)
        for i in range(len(image)):
                im = ax[i].imshow(image[i],norm=norm,
                    origin = 'lower', cmap=cmap,interpolation='none')
        pos_bar = [0.1, 0.9, 0.8, 0.03]
        cax = fig.add_axes(pos_bar)
        if not zp:
            fig.colorbar(im, cax=cax,orientation="horizontal", pad=0.2,format='%.0e')
            cax.xaxis.set_ticks_position("top")
        else:
            mmax,mmin = counts_to_mu(np.array([vmin,vmax]),22.5,0.33)
            bar = fig.colorbar(im, cax=cax,orientation="horizontal", 
                pad=0.2,format='%.0e',ticks=mu_to_counts(np.arange(mmax,mmin,2.5),zp,pixel_scale))
            ticklabels = 5*np.round((zp-2.5*np.log10(bar.get_ticks()))*2)/10
            bar.set_ticklabels(['{:2.1f}'.format(i) for i in ticklabels])
            cax.xaxis.set_ticks_position("top")

    # plt.tight_layout()


def displayimage(image, qmin=1, qmax=99, scale='linear',cmap='nipy_spectral_r'):           # with default arg 'title'
    interval = vis.AsymmetricPercentileInterval(qmin, qmax)
    vmin, vmax = interval.get_limits(image)
    if scale == 'linear':
        stretch = vis.LinearStretch(slope=0.5, intercept=0.5)
    if scale == 'sinh':
        stretch = vis.SinhStretch()
    if scale == 'log':
        stretch = vis.LogStretch()
    if scale == 'power':
        stretch = vis.PowerStretch(1.5)
    if scale == 'sqrt':
        stretch = vis.SqrtStretch()
    if scale == 'squared':
        stretch = vis.SquaredStretch()
    if scale == 'hist':
        stretch = vis.HistEqStretch(image)  # Needs argument data and data min, max for vmin, vmax
        vmin = np.nanmin(image); vmax = np.nanmax(image)

    
    if type(image)==list:
        norm = vis.ImageNormalize(image[0], vmin=vmin, vmax=vmax, stretch=stretch)
        fig,ax=plt.subplots(figsize = (6*len(image),6), nrows=1, 
                            ncols=len(image),sharex=True,sharey=True)

        for i in range(len(image)):
            if i==0:
                im = ax[i].imshow(image[i],norm=norm,
                    origin = 'lower', cmap=cmap)
            if i>0:
                ax[i].imshow(image[i],norm=im.norm,
                    origin = 'lower', cmap=cmap)
        pos_bar = [0.1, 0.9, 0.8, 0.03]
        cax = fig.add_axes(pos_bar)
        fig.colorbar(im, cax=cax,orientation="horizontal", pad=0.2,format='%.0e'
                )
        cax.xaxis.set_ticks_position("top")
    else:
        
        fig,ax = plt.subplots(1,1,figsize = (6,6))
        norm = vis.ImageNormalize(image, vmin=vmin, vmax=vmax, stretch=stretch)
        im = ax.imshow(image, interpolation='none', origin='lower', cmap=cmap, norm=norm)
        plt.colorbar(im)

    plt.tight_layout()



def plot_ellipses(profile, step=1, ax=None,max_r=None, center=(0,0),color='black',**kwargs):
    if type(profile) == dict: profile = Table(profile)
    if not ax: ax = plt.subplot(111)
    if not max_r: max_r = np.nanmax(profile['radius'])
    i=0
    for row in profile:
        if i%step==0 and row['radius']<max_r:
            ellipse_patch = patches.Ellipse(center,
                2*row['radius'],
                2*(row['radius'] * (1 - row['ellipticity'])),
                row['pa'],
                color=color, fill=False, **kwargs)

            ax.add_patch(ellipse_patch)
        i+=1
    return ax