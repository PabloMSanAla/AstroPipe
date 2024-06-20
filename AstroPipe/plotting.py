import AstroPipe.utils as ut
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
import astropy.visualization as vis
import matplotlib.patches as patches
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

from screeninfo import get_monitors
from matplotlib.widgets import RadioButtons,TextBox, Button
from matplotlib.backend_bases import MouseButton
from photutils.aperture import CircularAperture
from os.path import join
import cv2
from astropy.io import fits

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
matplotlib.rcParams['axes.linewidth'] = 1.3


def gaussian(x, mu, var, A=1):
    exp = np.exp(-((x - mu) ** 2) / (2 * var))
    norm = A / np.sqrt(2 * np.pi * var)
    return norm * exp

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
        fig.savefig(out,dpi=300,bbox_inches='tight', pad_inches=0.1)
    
def counts_to_mu(counts, zp, pixel_scale):
    return zp - 2.5*np.log10(counts/pixel_scale**2)

def mu_to_counts(mu, zp, pixel_scale):
    return 10**((zp-mu)/2.5)*pixel_scale**2

def mask_cmap(alpha=0.5):
    transparent = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
    gray = matplotlib.colors.colorConverter.to_rgba('black',alpha = alpha)
    cmap = matplotlib.colors.ListedColormap([transparent, gray])
    return cmap


def show_old(image,vmin=None,vmax=None,zp=None, pixel_scale=1,
            cmap='nipy_spectral_r',wcs=None, nan='cyan',maskalpha=0.5):
    
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
            mmax,mmin = counts_to_mu(np.array([vmin,vmax]),zp,pixel_scale)
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
            mmax,mmin = counts_to_mu(np.array([vmin,vmax]),zp,pixel_scale)
            bar = fig.colorbar(im, cax=cax,orientation="horizontal", 
                pad=0.2,format='%.0e',ticks=mu_to_counts(np.arange(mmax,mmin,2.5),zp,pixel_scale))
            ticklabels = 5*np.round((zp-2.5*np.log10(bar.get_ticks()))*2)/10
            bar.set_ticklabels(['{:2.1f}'.format(i) for i in ticklabels])
            cax.xaxis.set_ticks_position("top")

    # plt.tight_layout()

def show(image, ax=None, vmin=None, vmax=None, zp=None, pixel_scale=1, mask=None,
            cmap='nipy_spectral_r', wcs=None, nan='cyan', plotmask=True, maskalpha=0.5,**kwargs):
    ''' 
    Function to visualize astronomical images in logaritmic scales. 
    
    '''
    
    if len(np.shape(image)) == 2:    # Only one image

        # Define vmin and vmax either in units of image or magnitudes
        if vmin is None or vmax is None:
            _,median,std = sigma_clipped_stats(image, sigma=2.5)
            if zp is None:
                if vmin is None: vmin = median - 1.5*std
                if vmax is None: vmax = np.nanmax([np.nanpercentile(image.data,99.9), median + 25*std])
            else:
                if vmin is None: vmin = counts_to_mu(np.nanmax([np.nanpercentile(image.data,99.9), 
                                                        median + 25*std]), zp, pixel_scale)
                if vmax is None: vmax = counts_to_mu(median - 1.5*std, zp, pixel_scale) 
        vmin = 19 if np.isnan(vmin) else vmin
        vmax = ut.mag_limit(std, Zp=zp, scale=pixel_scale) - 1.5 if np.isnan(vmax) else vmax

        # Check if data is a masked array to plot mask
        hasmask = hasattr(image, 'mask')
        if hasmask: 
            data = image.data
            if mask is None: mask = image.mask
        else: data = image

        # Don't plot if mask is None
        if mask is None:
            plotmask = False

        # if vmin is negative, small offset to work with LogNorm
        if vmin < 0: data = data - 2*vmin; vmin = -vmin
        norm = LogNorm(vmin=vmin, vmax=vmax)


        if not ax: fig,ax = plt.subplots(1,1)
        else: fig=ax.get_figure()

        if not zp: # plot using norm if not calibrate
            im = ax.imshow(data, norm=norm, cmap=cmap, origin='lower', interpolation='none',**kwargs)
            fig.colorbar(im, ax=ax, shrink=0.6)
        else: # plot using magnitudes
            im = ax.imshow(counts_to_mu(data, zp, pixel_scale), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', interpolation='none',**kwargs)
            fig.colorbar(im, ax=ax, shrink=0.6)
        if (mask is not None) and plotmask: ax.imshow(mask, origin='lower', cmap=mask_cmap(alpha=maskalpha))
        plt.tight_layout()
        return ax

    else:                           # Multiple images   

        if vmin == None or vmax == None:
            stats = sigma_clipped_stats(image, sigma=2.5)
        if vmin == None: vmin = stats[1] - 1.5*stats[2]
        if vmax == None: vmax = np.nanmax([np.nanpercentile(image[0].data,99.9), stats[1] + 25*stats[2]])
        if vmin < 0: image -= 2*vmin; vmin = -vmin
        norm = LogNorm(vmin=vmin, vmax=vmax)
        if not ax: fig,ax=plt.subplots(figsize = (6*len(image),6), nrows=1, 
                            ncols=len(image),sharex=True,sharey=True)
        else: fig=ax.get_figure()

        plt.subplots_adjust(top=0.85,bottom=0.12,left=0.10,
                            right=0.978,hspace=0.205,wspace=0.038)

        for i,ima in enumerate(image):
            hasmask = hasattr(ima, 'mask')
            if hasmask: data = ima.data
            else: data = ima
            im = ax[i].imshow(data,norm=norm, origin = 'lower', cmap=cmap,interpolation='none',**kwargs)
            if hasmask: ax[i].imshow(image.mask, origin='lower',cmap=mask_cmap(alpha=maskalpha))

        pos_bar = [0.1, 0.9, 0.8, 0.03]
        cax = fig.add_axes(pos_bar)

        if not zp:
            fig.colorbar(im, cax=cax, orientation="horizontal", pad=0.2,format='%.0e')
            cax.xaxis.set_ticks_position("top")
        else:
            mmax,mmin = counts_to_mu(np.array([vmin,vmax]),zp, pixel_scale)
            bar = fig.colorbar(im, cax=cax,orientation="horizontal", 
                pad=0.2,format='%.0e',ticks=mu_to_counts(np.arange(mmax,mmin,2.5),zp,pixel_scale))
            ticklabels = 5*np.round((zp-2.5*np.log10(bar.get_ticks()))*2)/10
            bar.set_ticklabels(['{:2.1f}'.format(i) for i in ticklabels])
            cax.xaxis.set_ticks_position("top")
        plt.tight_layout()
        return ax
    
def histplot(data, vmin=None, vmax=None):
    '''
    Creates histogram of the data given. 

    Parameters
    ----------
        data : array_like
            Data to plot.
        vmin : float, optional
            Minimum value to plot.
        vmax : float, optional
            Maximum value to plot.
    Returns
    -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
    '''
    mean,med,std = sigma_clipped_stats(data, sigma=2.5)
    q99 = np.nanpercentile(data,99.9)
    
    hrange = [mean-10*std, q99*1.1]
    if vmin is not None: hrange[0] = vmin 
    if vmax is not None: hrange[1] = vmax 
    
    fig,ax = plt.subplots(1,1,figsize=(16,4))
    ax.hist(data.flatten(),range=hrange,bins=1000)
    ax.axvline(med,color='red',ls=':')
    ax.axvline(med-std,color='red',ls='--')
    ax.axvline(med+std,color='red',ls='--')
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Pixel count')
    ax.set_yscale('log')
    fig.suptitle(fr'$\mu \pm \sigma$ = {mean:1.3e} $\pm$ {std:1.3e} $\,\,\,$ Median = {med:1.3e}',
                 fontsize=16)
    fig.tight_layout()
    return fig,ax

def surface_figure(image, profile, out=None, mumax=None, radmax=None, **kwargs):
    '''Function that creates a figure with the image, the profile and the mask.
     In the first axes will show the surface brightness image, 
    in the second the image with the mask and the ellipses fit to create the profile
    in the last column, it will show the profile of the image with its 
    morphological parameters. 
    
    Parameters
    ----------
        image : AstroPipe.Image
            Image object to plot.
        profile : AstroPipe.Profile
            Profile object to plot.
        out : str, optional
            Path to save the figure. If None, the figure will not be saved.
        mumax : float, optional
            Maximum surface brightness value to plot.
        radmax : float, optional
            Maximum radius value to plot in pixels.
        **kwargs : dict, optional
        
    Returns
    -------
        fig : matplotlib.figure.Figure
            Figure object.
    '''
    plt.rcParams["text.usetex"]= True
    
    fig = plt.figure(figsize=(11.5,4))
    axim = plt.subplot2grid((5,3),(0,0),rowspan=5)
    axmask = plt.subplot2grid((5,3),(0,1),rowspan=5, sharex=axim)
    axmu = plt.subplot2grid((5,3),(0,2),rowspan=3)
    axpa = plt.subplot2grid((5,3),(3,2),rowspan=1,sharex=axmu)
    axeps = plt.subplot2grid((5,3),(4,2),rowspan=1,sharex=axmu)

    if mumax is None: 
        arglim = np.where(np.isnan(profile.mu) * (profile.rad > profile.rad[ut.closest(profile.mu,24)]))[0]
        arglim = arglim[0]-1 if arglim.any() else len(profile.mu)-1
        mumax = profile.mu[arglim] if not np.isnan(profile.mu[arglim]) else 32
    if radmax is None: radmax = profile.rad[ut.closest(profile.mu, mumax)]

    # Properties
    extent = np.array([-image.x,image.data.shape[1]-image.x,
                    -image.y,image.data.shape[0]-image.y]).astype(float)
    extent *= image.pixel_scale
    vmax = np.ceil(ut.mag_limit(image.bkgstd, Zp=image.zp, omega=image.pixel_scale, scale=image.pixel_scale,n=1))
    if np.isnan(vmax): vmax = mumax - 5
    specs = {'vmin':np.ceil(np.nanmin(profile.mu)), 'vmax':vmax, 'cmap':'nipy_spectral', 
            'origin':'lower', 'interpolation':'none', 'extent':extent}
    
    for key in kwargs: specs[key] = kwargs[key]

    # Showing images and profile
    magnitude = image.zp - 2.5*np.log10(image.data.data - image.bkg) + 5*np.log10(image.pixel_scale) 
    im = axim.imshow(magnitude, **specs)
    axmask.imshow(magnitude, **specs)
    axmask.imshow(image.data.mask, origin='lower',cmap=mask_cmap(alpha=0.5), extent=extent)
    axmask = plot_ellipses(profile.rad*image.pixel_scale, profile.pa, profile.eps, 
                        np.array((profile.x-image.x, profile.y-image.y))*image.pixel_scale, 
                        ax=axmask, step=5, alpha=0.6, max_r=radmax*image.pixel_scale)

    fig = profile.plot(axes=(axmu,axpa,axeps))

    # Properties 
    for ax in [axim, axmask]:
        ax.set_xlim(np.array([-1.1,1.1])*radmax*image.pixel_scale)
        ax.set_ylim(np.array([-1.1,1.1])*radmax*image.pixel_scale)
        ax.set_xlabel('Distance [arcsec]',fontsize=14)
    
    
    for ax in [axmu,axeps,axpa]:
        ax.set_xlim(np.array([-0.08,1.1])*radmax*image.pixel_scale)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(labelsize=8)

    axim.set_ylabel('Distance [arcsec]',fontsize=14)
    axmask.set_yticklabels([])
    axim.text(1.0, 1.02, image.name.replace('_','-'), va='bottom', ha='right',
            transform=axim.transAxes, fontweight='bold', fontsize=16)

    # # Colorbar inside profile axes
    # cbarax = axmu.inset_axes([0.9*radmax*image.pixel_scale, specs['vmin'], 
    #                         0.05*radmax*image.pixel_scale, specs['vmax']-specs['vmin']],
    #             transform=axmu.transData)
    # cbar = fig.colorbar(im, cax=cbarax)
    # cbarax.axis('off')
    # cbarax.invert_yaxis()
    # axmu.set_ylim([mumax*1.15,0.95*profile.mu[np.isfinite(profile.mu)][0]])
    # axmu.set_xlim(np.array([-0.05,1.1])*radmax*image.pixel_scale)

    # Colorbar in top of the mask image
    axins = axmask.inset_axes([0,1.01, 
                           1,0.04])
    cbar = fig.colorbar(im, cax=axins, orientation='horizontal')#, location='top', pad=0.1)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('$\mu$ [mag arcsec$^{-2}$]',fontsize=14,labelpad=10)


    fig.subplots_adjust(top=0.915, bottom=0.09,
                        left=0.045, right=0.95,
                        hspace=0.3, wspace=0.1)
    if out:
        fig.savefig(out, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

    return fig


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

def plot_ellipses(radius, pa, eps, center, ax=None, color='black', step=1, max_r=None, **kwargs):
    if not ax: ax = plt.subplot(111)
    if not max_r: max_r = np.nanmax(radius)
    for i in range(len(radius)):
        if i%step==0 and radius[i]<=max_r:
            ellipse_patch = patches.Ellipse(center[:,i],
                2*radius[i],2*(radius[i] * (1 - eps[i])),
                pa[i], color=color, fill=False, **kwargs)
            ax.add_patch(ellipse_patch)
    return ax


def plot_ellipses_old(profile, step=1, ax=None,max_r=None, center=(0,0),color='black',**kwargs):
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

def plot_ellipses_new(profile, step=1, ax=None, max_r=None, center=(0,0),color='black',**kwargs):
    if type(profile) == dict: profile = Table(profile)
    if not ax: ax = plt.subplot(111)
    if not max_r: max_r = np.nanmax(profile['radius'])
    i=0
    for i,rad in enumerate(profile.rad):
        if i%step==0 and rad<=max_r:
            ellipse_patch = patches.Ellipse(center,
                2*rad,
                2*(rad * (1 - profile.eps[i])),
                profile.pa[i],
                color=color, fill=False, **kwargs)
            ax.add_patch(ellipse_patch)    
    return ax


def make_random_cmap(ncolors=256, seed=None):
    """
    Make a matplotlib colormap consisting of (random) muted colors.

    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.
        Separate function calls with the same ``seed`` will generate the
        same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
        The matplotlib colormap with random colors in RGBA format.
    """
    from matplotlib import colors

    rng = np.random.default_rng(seed)
    hue = rng.uniform(low=0.0, high=1.0, size=ncolors)
    sat = rng.uniform(low=0.2, high=0.7, size=ncolors)
    val = rng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((hue, sat, val))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    return colors.ListedColormap(colors.to_rgba_array(rgb))
    
def make_cmap(max_label, background_color='#000000ff', seed=None):

    from matplotlib import colors

    cmap = make_random_cmap(max_label + 1, seed=seed)

    if background_color is not None:
        cmap.colors[0] = colors.to_rgba(background_color)

    return cmap


def mags(data, zp, scale):
        return zp + 5*np.log10(scale) - 2.5*np.log10(data)


def interactive_mask_modify(Image, out=None, scaling = 0.8, screendpi = 100, **kwargs):

    scaling = 0.8
    for m in get_monitors():
        if m.is_primary:
            figsize = np.nanmin([m.width,m.height])*scaling
            break
    screendpi = 100

    galaxy = Image.name

    specs = {'vmin' : 19, 'vmax':26.5, 'cmap': 'nipy_spectral',
            'origin':'lower', 'interpolation':'none'}
    
    for key in kwargs: specs[key] = kwargs[key]

    # Global variables, not sure if needed
    global original_labels, new_labels
    global size, historylabels, historyapers, newmaskname

    # Create figure and axes
    fig = plt.figure(figsize=(figsize/screendpi,figsize/screendpi))
    ax = fig.add_axes([0.05,0.05,0.75,0.85])
    ax_radio = fig.add_axes([0.81, 0.3, 0.15, 0.15])
    size_ax = fig.add_axes([0.81, 0.25, 0.15, 0.05])
    save_ax = fig.add_axes([0.81, 0.15, 0.15, 0.05])


    # Plot image and mask
    im = ax.imshow(mags(Image.data.data, Image.zp, Image.pixel_scale), **specs)
    fig.colorbar(im, ax=ax, shrink=0.8, location='top')
    immask = ax.imshow(Image.data.mask, origin='lower', cmap=mask_cmap(alpha=0.5))

    # Widgets
    radio_butons = RadioButtons(ax_radio, ('delete', 'create'), active=0)
    size = 5
    size_box = TextBox(size_ax, 'size', initial=str(size))

    newmaskname = join(Image.directory, f'{galaxy}_newmask.fits')
    save_button = Button(save_ax, 'Save')

    new_mask = np.zeros_like(Image.data.mask)
    new_mask[Image.data.mask != 0] = 1

    original_labels = cv2.connectedComponentsWithStats(new_mask.astype(np.uint8), 8, cv2.CV_32S)[1]
    new_labels = original_labels.copy()
    historylabels = [0]*100
    historyapers = []

    def update_size(text):
        global size
        size = float(text)

    
    def save_newmask(hello):
        fits.PrimaryHDU(new_labels.astype(np.int32),
                        Image.header).writeto(
                        newmaskname, overwrite=True)
        print('New mask saved as {}'.format(newmaskname))


    def on_click(event):
        global historylabels 
        if event.inaxes!=ax_radio:
            if radio_butons.value_selected == 'delete':
                if event.button is MouseButton.LEFT:
                    x = np.int32(event.xdata)
                    y = np.int32(event.ydata)
                    label = original_labels[y,x]
                    if label != 0: historylabels.append(label)
                    new_labels[new_labels == historylabels[-1]] = 0
                elif event.button is MouseButton.MIDDLE:
                    new_labels[original_labels == historylabels[-1]] = historylabels[-1]
                    del historylabels[-1]
            elif radio_butons.value_selected == 'create':    
                if event.button is MouseButton.LEFT:
                    x = event.xdata
                    y = event.ydata 
                    historyapers.append(CircularAperture([x, y], size))
                    aper_mask = historyapers[-1].to_mask(method='center').to_image(Image.data.shape)
                    label = np.nanmax(new_labels) + 1
                    new_labels[aper_mask != 0] = label
                    original_labels[(aper_mask != 0) * (original_labels==0)] =label
                    historylabels.append(label)
                elif event.button is MouseButton.MIDDLE:
                    new_labels[original_labels == historylabels[-1]] = 0
                    del historylabels[-1]
        immask.set_data(new_labels)
        plt.draw()

    plt.connect('button_press_event', on_click)
    size_box.on_submit(update_size)
    save_button.on_clicked(save_newmask)

    plt.show()

