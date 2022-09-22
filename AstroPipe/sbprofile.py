

import AstroPipe.utilities as ut

import numpy as np 
import os

import astropy.units as u
from astropy.table import QTable
from astropy.wcs.utils import pixel_to_skycoord
from astropy.stats import sigma_clip
from astropy.visualization import ImageNormalize,LogStretch

from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.aperture import RectangularAperture


import matplotlib.patches as patch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import stats
from scipy.ndimage import median_filter
from scipy.signal import argrelextrema

from lmfit.models import GaussianModel


from scipy.signal import medfilt
from sklearn.cluster import KMeans


def isophotal_photometry(IMG,center=None,pa=None,eps=None,r_eff=None,
                zp=None,max_r=1000,fix_center=False,fix_pa=False,fix_eps=False,
                plot=False,save=None):

    if not pa: pa =  IMG.pa * np.pi/180
    if not eps: eps = IMG.eps 
    if not center: center=IMG.pix
    if not r_eff: r_eff=IMG.r_eff
    if not zp: zp=IMG.zp

    data = IMG.data - IMG.bkg 

    guess_aper = EllipseGeometry(x0=center[0], y0=center[1],
                            sma=r_eff,eps=eps, pa=pa)


    ellipse = Ellipse(data, guess_aper)

    isolist = ellipse.fit_image(r_eff,integrmode='median',sclip=3,nclip=3,maxsma=max_r,
                minsma=1,step=0.05,fix_center=fix_center,fix_pa=fix_pa,fix_eps=fix_eps)


    magnitude = zp - 2.5*np.log10(isolist.intens/IMG.pixel_scale**2) 
    lower_err = zp - 2.5*np.log10((isolist.intens+isolist.int_err)/IMG.pixel_scale**2) 
    upper_err   = zp - 2.5*np.log10((isolist.intens-isolist.int_err)/IMG.pixel_scale**2) 

    
    lower_err = magnitude - lower_err 
    upper_err = upper_err - magnitude  
    
    lower_err[np.isnan(lower_err)] = np.nanmin(lower_err)
    upper_err[np.isnan(upper_err)] = np.nanmax(upper_err)

    if IMG.wcs:
        coords = pixel_to_skycoord(isolist.x0,isolist.y0,IMG.wcs)
        ra = coords.ra.deg*u.deg
        dec = coords.dec.deg*u.deg
    else:
        ra = isolist.x0
        dec = isolist.y0

    meta = {'pixel_scale':IMG.pixel_scale,
                    'zero_point':zp, 'background':IMG.bkg}

    if hasattr(IMG,'std'): 
        meta['STD'] = IMG.std 
        meta['MAG_LIM'] = ut.mag_limit(IMG.std, Zp=zp,scale=IMG.pixel_scale)

    profile = QTable([isolist.sma*IMG.pixel_scale*u.arcsec,magnitude*u.mag/u.arcsec**2,
                    lower_err*u.mag/u.arcsec**2,upper_err*u.mag/u.arcsec**2,
                    isolist.pa*(180/np.pi)*u.deg,isolist.eps,
                    ra,dec],
                    names=['radius','surface_brightness','sb_err_low','sb_err_upper',
                    'pa','ellipticity','ra_center','dec_center'],
                    meta=meta)
    if plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        
        vmin = IMG.mu_to_counts(IMG.maglim)
        vmax = IMG.mu_to_counts(18)

        norm = ImageNormalize(IMG.data,vmin=vmin,vmax=vmax,stretch=LogStretch())
        
        im = ax.imshow(data,norm=norm,interpolation='none', 
                        origin='lower',cmap='nipy_spectral_r')
        
        bar = fig.colorbar(im,ticks=IMG.mu_to_counts(np.arange(18,IMG.maglim,2.5)))
        bar.set_ticklabels(np.arange(18,IMG.maglim,2.5))

        for iso in isolist:
            ellipse_patch = patch.Ellipse((iso.x0,iso.y0),
                    2*iso.sma, 2*iso.sma*(1-iso.eps),
                    iso.pa*180/np.pi,
                    color='black',alpha=0.3,fill=False)
            ax.add_patch(ellipse_patch)
        ax.set_xlim([IMG.pix[0]-max_r,IMG.pix[0]+max_r])
        ax.set_ylim([IMG.pix[1]-max_r,IMG.pix[1]+max_r])
        plt.tight_layout()
        
    if save:
        profile.write(save.profile,overwrite=True)
        if plot:
            plt.savefig(os.path.join(save.temp,
                os.path.basename(save.profile).split('.')[-2]+'_apertures.jpg'),dpi=200)


    return profile


def isophotal_photometry_fix(IMG, center=None, eps=None, pa=None, rad=None,
                       zp=None, growth_rate = 1.03, max_r = 1000, plot=False,save=None):

    if type(pa)==type(None): pa =  IMG.pa * np.pi/180
    if type(eps)==type(None): eps = IMG.eps 
    if type(center)==type(None): center=IMG.pix
    if not zp: zp=IMG.zp
    if type(rad)==type(None): 
        rad = [1]; 
        while rad[-1]<max_r:
            rad.append(rad[-1]*growth_rate)

    
    if type(pa)!=list and type(pa)!=np.ndarray: pa = [pa]*len(rad)
    if type(eps)!=list and type(eps)!=np.ndarray: eps = [eps]*len(rad)
    if np.shape(center) != (2,len(rad)): center = np.transpose(len(rad)*[center])

    # print(np.shape(rad),np.shape(pa),np.shape(eps),np.shape(center))
    data = IMG.data - IMG.bkg 

    intensity = []
    intensity_std = []
    ellip_apertures = []

    previous_mask = np.zeros_like(IMG.data)

    if plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        vmin = IMG.mu_to_counts(IMG.maglim)
        vmax = IMG.mu_to_counts(18)

        norm = ImageNormalize(IMG.data,vmin=vmin,vmax=vmax,stretch=LogStretch())
        
        im = ax.imshow(data,norm=norm,interpolation='none', 
                        origin='lower',cmap='nipy_spectral_r')
        
        bar = fig.colorbar(im,ticks=IMG.mu_to_counts(np.arange(18,IMG.maglim,2.5)))
        bar.set_ticklabels(np.arange(18,IMG.maglim,2.5))

    for i in range(len(rad)):
        if len(ellip_apertures) > 1:
            previous_mask = mask

        ellip_apertures.append(EllipticalAperture((center[0][i],center[1][i]), rad[i], (1-eps[i])*rad[i], pa[i]))
        mask = ellip_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = ut.where([data.mask==False,mask!=0,previous_mask==0])
        clipped = sigma_clip(data.data[index],sigma=3,maxiters=3)
        intensity.append(np.ma.median(clipped))
        intensity_std.append(np.nanstd(clipped)/np.sqrt(np.size(clipped)))
        if plot:
            ellip_apertures[-1].plot(color='black',alpha=0.4,lw=0.7,axes=ax)
    
    if plot: 
        ax.set_xlim([IMG.pix[0]-max_r,IMG.pix[0]+max_r])
        ax.set_ylim([IMG.pix[1]-max_r,IMG.pix[1]+max_r])
        plt.tight_layout()

    sb_profile  = zp - 2.5*np.log10(np.divide(intensity,IMG.pixel_scale**2)) 
    lower_err = zp - 2.5*np.log10(np.add(intensity,intensity_std)/IMG.pixel_scale**2) 
    upper_err   = zp - 2.5*np.log10(np.subtract(intensity,intensity_std)/IMG.pixel_scale**2) 

    lower_err = sb_profile - lower_err 
    upper_err = upper_err - sb_profile  
    
    lower_err[np.isnan(lower_err)] = np.nanmin(lower_err)
    upper_err[np.isnan(upper_err)] = np.nanmax(upper_err)

    rad = np.array(rad)*IMG.pixel_scale

    if IMG.wcs:
        coords = pixel_to_skycoord(center[0,:],center[1,:],IMG.wcs)
        ra = coords.ra.deg*u.deg
        dec = coords.dec.deg*u.deg
    else:
        ra = center[0,:]
        dec = center[1,:]

    profile = QTable([rad*u.arcsec ,sb_profile * (u.mag / u.arcsec**2),
                    lower_err*u.mag/u.arcsec**2,upper_err*u.mag/u.arcsec**2,
                    pa*u.deg,eps,ra,dec],
                    names=['radius','surface_brightness','sb_err_low','sb_err_upper',
                    'pa','ellipticity','ra_center','dec_center'],
                    meta = {'pixel_scale':IMG.pixel_scale,
                    'zero_point':zp,})

    if save:
        profile.write(save.profile.split('.')[-2]+'_fixed.fits',overwrite=True)
        if plot:
            plt.savefig(os.path.join(save.temp,
                os.path.basename(save.profile).split('.')[-2]+'_apertures_fixed.jpg'),dpi=200)
    return profile


def rectangular_photometry(IMG, center=None, pa=None, width=5,
                       zp=None, growth_rate = 1.03, max_r = 1000,plot=False,save=None):
    
    if not pa: pa =  IMG.pa * np.pi/180
    if not center: center = IMG.pix
    if not zp: zp = IMG.zp

    
    data = IMG.data - IMG.bkg 

    rect_apertures = []
    intensity_rec = []
    intensity_rec_std = []
    radii = []
    previous_mask = np.zeros_like(IMG.data)

    if plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        vmin = IMG.mu_to_counts(IMG.maglim)
        vmax = IMG.mu_to_counts(18)

        norm = ImageNormalize(IMG.data,vmin=vmin,vmax=vmax,stretch=LogStretch())
        
        im = ax.imshow(data,norm=norm,interpolation='none', 
                        origin='lower',cmap='nipy_spectral_r')
        
        bar = fig.colorbar(im,ticks=IMG.mu_to_counts(np.arange(18,IMG.maglim,2.5)))
        bar.set_ticklabels(np.arange(18,IMG.maglim,2.5))
    
    sma = 1
    while sma <= 2*max_r:


        if len(rect_apertures) > 1:
            previous_mask = mask

        rect_apertures.append(RectangularAperture((center[0],center[1]), sma, width, pa))
        mask = rect_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = ut.where([data.mask==False,mask!=0,previous_mask==0])
        intensity_rec.append(np.ma.median(data.data[index]))
        intensity_rec_std.append(np.ma.std(data.data[index])/np.sqrt(
                                np.size(data.data[index])))
        radii.append(sma/2)
        sma *= growth_rate

        if plot:
            rect_apertures[-1].plot(color='black',alpha=0.7,lw=0.7,axes=ax)
    
    if plot: 
        ax.set_xlim([IMG.pix[0]-max_r,IMG.pix[0]+max_r])
        ax.set_ylim([IMG.pix[1]-max_r,IMG.pix[1]+max_r])
        plt.tight_layout()


    sb_profile  = zp - 2.5*np.log10(np.divide(intensity_rec,IMG.pixel_scale**2))
    lower_err   = zp - 2.5*np.log10(np.add(intensity_rec,intensity_rec_std)/IMG.pixel_scale**2) 
    upper_err   = zp - 2.5*np.log10(np.subtract(intensity_rec,intensity_rec_std)/IMG.pixel_scale**2) 
    
    lower_err = sb_profile - lower_err 
    upper_err = upper_err - sb_profile  
    
    lower_err[np.isnan(lower_err)] = np.nanmin(lower_err)
    upper_err[np.isnan(upper_err)] = np.nanmax(upper_err) 

    radii = np.array(radii)*IMG.pixel_scale

    profile = QTable([radii*u.arcsec ,sb_profile * (u.mag / u.arcsec**2),
                    lower_err*u.mag/u.arcsec**2,upper_err*u.mag/u.arcsec**2],
                    names=['radius','surface_brightness','sb_err_low','sb_err_upper'],
                    meta = {'pixel_scale':IMG.pixel_scale,
                    'zero_point':zp,
                    'pa':pa,'width':width,
                    'x0':center[0],'y0':center[1]})
    
    if save:
        profile.write(save.profile.split('.')[-2]+'_rect.fits',overwrite=True)
        if plot:
            plt.savefig(os.path.join(save.temp,
                os.path.basename(save.profile).split('.')[-2]+'_rectangles.jpg'),dpi=200)

    return profile


def background_estimation(IMG, width = 5, max_r=None, out=True):

    if not max_r:
        max_r = np.max(np.shape(IMG.data))//2

    width /= IMG.pixel_scale
    if width<10: width=20

    clipped = sigma_clip(IMG.data,sigma=3)
    clipped = clipped[np.where(~clipped.mask)]
    first_mode = find_mode(clipped)[0]
    IMG.set_background(first_mode)

    profile = isophotal_photometry_fix(IMG, growth_rate=1.2, max_r=max_r)
    mu = profile['surface_brightness']
    radius = profile['radius'][np.isfinite(mu)]
    
    mu = mu[np.isfinite(mu)]
    
     
    argsort = np.argsort(mu)
    dradius = np.diff(radius)
    n = np.where(argsort==np.argmax(dradius))[0][0] - len(argsort)
    radius_selected = radius[argsort[n:]]
    dradius_selected = np.diff(radius_selected)
    std_radius = np.max(dradius_selected)

    if np.min(radius_selected) < std_radius:
        bkg_radius = np.min(radius_selected.value) + 0.1*std_radius.value
    else:
        bkg_radius = np.max(radius_selected.value) + 10*IMG.pixel_scale
    
    bkg_aperture = EllipticalAnnulus((IMG.pix[0],IMG.pix[1]),
                    bkg_radius/IMG.pixel_scale , (bkg_radius+width)/IMG.pixel_scale, 
                    (1-IMG.eps)*(bkg_radius+width)/IMG.pixel_scale, None,
                    IMG.pa*np.pi/180)

    mask_aper = bkg_aperture.to_mask(method='center').to_image(IMG.data.shape)
    mask_aper = np.ma.array(mask_aper,mask=1-mask_aper)
    aper_values = IMG.data*mask_aper
    aper_values = aper_values[np.where(~aper_values.mask)].flatten()
    mode,gauss_fit = find_mode(aper_values)

    IMG.set_background(first_mode + mode)

    if out:
        im = IMG.show(width=3*bkg_radius/IMG.pixel_scale)
        bkg_aperture.plot()
        plt.title('Background of {:s} = {:^e}'.format(IMG.name,mode))
        ax2=im.figure.add_subplot(3,3,8)
        ax2.hist(aper_values,bins=500)
        ax2.axvline(mode,c='r',ls='-.')
        ax2.plot(gauss_fit.userkws['x'],gauss_fit.best_fit,'k',ls='-.')
        ax2.axis('off')
        plt.savefig(out,dpi=200)
    
    return mode, bkg_radius/IMG.pixel_scale

       
def find_mode(data):
    hist, bin_edges = np.histogram(data,bins=500)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    model = GaussianModel()
    params = model.guess(hist,x=bin_centers)
    result = model.fit(hist,params,x=bin_centers)    

    return  result.values['center'], result

def derivative(x,y,n=4):
    """
    Computes de slope from the n adjacent points using 
    linear regression.
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



def aaron_break_finder(rad,mu,min=21,max=31,n=4,p=5):
    """
    Finds the disk breaks in the surface brightness profile as 
    seen in Watkins et al. (2019)
    """
    index = ut.where([mu>min,mu<max,np.isfinite(mu)])[0]
    der = derivative(rad[index],mu[index],n=n)
    der = median_filter(der,size=int(p*len(mu)/100))
    cum_sum = np.cumsum(der-np.mean(der))
    maximum = rad[index][argrelextrema(cum_sum, np.greater)[0]]
    minimum = rad[index][argrelextrema(cum_sum, np.less)[0]]
    critic =  np.sort(np.append(maximum,minimum))
    arg = []
    for point in critic:
        arg.append(ut.closest(rad,point)+1)
    return arg



def find_slope(rad, mu, w=2, smFac=0.1):
    '''
    Derive slope of surface brightness profile vs. radius using adjacent w data
    points.  Returns this and a median-smoothed version of it.

    Requires:
     - rad (numpy.array): radius array
     - mu (numpy.array): surface brightness array, in mags/arcsec^2
     - w (int): width of array segments used to calculate local slope
     - smFac (float): fraction of array used as smoothing kernel size

    Returns:
     - inrad (numpy.array): clipped radius array of proper size
     - smooth_h (numpy.array): median filtered local scale lengths
     - h (numpy.array): unfiltered local scale lengths
    '''
    inrad = rad[w:-w]
    m = np.array([])
    for r in range(len(inrad)):
        idx = np.where(rad == inrad[r])[0][0]
        fit_rads = rad[idx-w:idx+w]
        fit_flux = mu[idx-w:idx+w]
        f = np.polyfit(fit_rads, fit_flux, 1)
        m = np.append(m, f[0])
        
    kernel = round(smFac*len(m), 0)
    if kernel % 2 == 0:
        kernel += 1
    smooth_m = medfilt(m, int(kernel))
    h = 1.0857/m  # Scale length
    smooth_h = 1.0857/smooth_m
    
    return inrad, smooth_h, h


def cusum_wat(x):
    '''
    Cumulative sum of the difference from the mean.
    The minimum or maximum is the break point.

    Requires:
     - x (numpy.array): the array for which you want the CUSUM

    Returns:
     - s (numpy.array): the CUSUM of x
    '''
    xbar = np.mean(x)
    s = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0:
            s[i] = 0
        else:
            s[i] = s[i-1] + (x[i]-xbar)
            
    return s

def cusum(x):
    '''
    Cumulative sum of the difference from the mean.
    The minimum or maximum is the break point.

    Requires:
     - x (numpy.array): the array for which you want the CUSUM

    Returns:
     - c (numpy.array): the CUSUM of x
    '''
    c = np.cumsum(x-np.mean(x))
    c[0] = 0
    return c

def change_point(inrad, h, N=1e5):
    '''
    Finding the change point of the radial slope profile using the method illustrated here:
    http://www.variation.com/cpa/tech/changepoint.html

    Requires:
     - inrad (numpy.array): clipped radius array output from find_slope()
     - h (numpy.array): a local slope array output from find_slope()
     - N (int): number of bootstrapping iterations for confidence estimate

    Returns:
     - i1 (int): index in inrad of change point
     - s (numpy.array): cumulative sum profile
     - conf (float): percentage of bootstrapping runs in which a break was found

    NOTE: can use either h or smooth_h; whichever you feel does the better job
    NOTE2: for confidence estimate, consider also applying the
    criterion that the break must occur in the same general place in
    the array, to enhance robustness.  Not currently implemented.
    '''
    s = cusum(h)
    # Amplitude of the break
    sdiff = np.max(s) - np.min(s)
    
    # Confidence estimate by bootstrapping
    # print('Bootstrapping...')
    bs_sdiff = np.array([])
    for i in range(np.int64(N)):
        randi = np.random.randint(0, len(h), len(h))
        randh = h[randi]
        rands = cusum(randh)
        randsdiff = np.max(rands) - np.min(rands)
        bs_sdiff = np.append(bs_sdiff, randsdiff)
    good = bs_sdiff < sdiff
    conf = len(good[good])/len(good)
    
    # Location estimate one: maximum in s
    if abs(np.max(s)) > abs(np.min(s)):
        i1 = np.where(s == np.max(s))[0][0]
    else:
        i1 = np.where(s == np.min(s))[0][0]
    
    return i1, s, conf


def find_all_breaks(inrad, h, rin, rout):
    '''
    Tailored to find a maximum of three breaks across the disk.  Will
    require manual adjustment to find more than this.

    Requires:
     - inrad (numpy.array): clipped radius array output from find_slope()
     - h (numpy.array): a local slope array output from find_slope()
     - rin (float): inner radius boundary for searching for breaks
     - rout (float): outer radius boundary for searching for breaks

    Returns:
     - rbr1, rbr2, rbr3 (float): the three break radii the routine
       found, in the same units as inrad array

    NOTE: if a break wasn't found to be significant, this returns -999
    '''
    want = (inrad >= rin) & (inrad <= rout) & (~np.isinf(h)) & (~np.isnan(h))
    goodrad = inrad[want]
    goodh = h[want]
    
    # First run, find if there's a global break
    ig, sg, confg = change_point(goodrad, goodh)
    # Check confidence; if less than 95% of bootstrap runs found a
    # break, the global break isn't significant, so it doesn't bother to
    # look for any beyond that.
    if confg < 0.95:
        print('No breaks found')
        return -999, -999, -999
    else:
        rbr2 = goodrad[ig]
        want_in = (goodrad < rbr2)
        want_out = (goodrad > rbr2)
        i_in, s_in, conf_in = change_point(goodrad[want_in], goodh[want_in])
        i_out, s_out, conf_out = change_point(goodrad[want_out], goodh[want_out])
        # print('Initial guess: ',goodrad[want_in][i_in], goodrad[ig], goodrad[want_out][i_out])
        # Same here: always using 95% confidence threshold (p-value = 0.05)
        if conf_in < 0.95:
            rbr1 = -999
        else:
            rbr1 = goodrad[want_in][i_in]
        if conf_out < 0.95:
            rbr3 = -999
        else:
            rbr3 = goodrad[want_out][i_out]
            
        # print(conf_in, confg, conf_out)
        return rbr1, rbr2, rbr3

def break_estimation(radius, mu, rms, skyrms=0, 
        rin=7,rout=150, npx = 4, zp=22.5 , pixel_scale=0.333):

    '''
    Estimate three different break in a surface brightness radial profile.
    The consistance of the breaks are also estimated by adding noise to the profile. 
    Requires:
     - radius (numpy.array): radius array in arcsec
     - mu (numpy.array): mu array in counts
     - std (numpy.array): std array in counts
     - skyrms (float): sky rms level in counts
     - npx (int): number of pixels in the image
     - zp (float): zeropoint of the image
     - pixel_scale (float): pixel scale in arcsec/pixel
    Returns:
     - rbr (float): break radius in arcsec
     - rbr_err (float): error on break radius in arcsec
    '''

    sig = np.sqrt((np.sqrt(skyrms**2)/np.sqrt(npx))**2 + rms**2)
    intensity = (pixel_scale**2) * np.power(10,(zp - mu)/2.5)
    inrad, smooth_h, h = find_slope(radius, mu)

    mu_up = -2.5*np.log10(intensity+sig) + zp
    mu_up[np.isnan(mu_up)] = -999
    inrad_up, smooth_h_up, h_up = find_slope(radius, mu_up)

    mu_down = -2.5*np.log10(intensity+sig) + zp
    mu_down[np.isnan(mu_down)] = -999
    inrad_down, smooth_h_down, h_down = find_slope(radius, mu_down)

    
    # Unaltered profile
    rbr1, rbr2, rbr3 = find_all_breaks(inrad, smooth_h, rin, rout)
    # First sky perturbation
    rbr1_up, rbr2_up, rbr3_up = find_all_breaks(inrad_up, smooth_h_up, rin, rout)
    # Second sky perturbation
    rbr1_down, rbr2_down, rbr3_down = find_all_breaks(inrad_down, smooth_h_down, rin, rout)

    rbr = np.array([rbr1,rbr2,rbr3,
                    rbr1_up,rbr2_up,rbr3_up,
                    rbr1_down,rbr2_down,rbr3_down])
    if any(rbr>0):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(rbr[rbr>0].reshape(-1,1))
        centers = kmeans.cluster_centers_
    else:
        centers = [-999,-999,-999]

    return centers



import numpy as np

from photutils.isophote.geometry import EllipseGeometry

__all__ = ['build_ellipse_model']


def build_ellipse_model(shape, isolist,intensity=None, fill=0., high_harmonics=False):
    """
    Build a model elliptical galaxy image from a list of isophotes.

    For each ellipse in the input isophote list the algorithm fills the
    output image array with the corresponding isophotal intensity.
    Pixels in the output array are in general only partially covered by
    the isophote "pixel".  The algorithm takes care of this partial
    pixel coverage by keeping track of how much intensity was added to
    each pixel by storing the partial area information in an auxiliary
    array.  The information in this array is then used to normalize the
    pixel intensities.

    Parameters
    ----------
    shape : 2-tuple
        The (ny, nx) shape of the array used to generate the input
        ``isolist``.

    isolist : `~photutils.isophote.IsophoteList` instance
        The isophote list created by the `~photutils.isophote.Ellipse`
        class.

    fill : float, optional
        The constant value to fill empty pixels. If an output pixel has
        no contribution from any isophote, it will be assigned this
        value.  The default is 0.

    high_harmonics : bool, optional
        Whether to add the higher-order harmonics (i.e., ``a3``, ``b3``,
        ``a4``, and ``b4``; see `~photutils.isophote.Isophote` for
        details) to the result.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image with the model galaxy.
    """
    from scipy.interpolate import LSQUnivariateSpline

    # the target grid is spaced in 0.1 pixel intervals so as
    # to ensure no gaps will result on the output array.
    finely_spaced_sma = np.arange(isolist[0].sma, isolist[-1].sma, 0.1)

    if intensity is None: intensity = isolist.intens 
    # interpolate ellipse parameters

    # End points must be discarded, but how many?
    # This seems to work so far
    nodes = isolist.sma[2:-2]

    intens_array = LSQUnivariateSpline(
        isolist.sma, intensity, nodes)(finely_spaced_sma)
    eps_array = LSQUnivariateSpline(
        isolist.sma, isolist.eps, nodes)(finely_spaced_sma)
    pa_array = LSQUnivariateSpline(
        isolist.sma, isolist.pa, nodes)(finely_spaced_sma)
    x0_array = LSQUnivariateSpline(
        isolist.sma, isolist.x0, nodes)(finely_spaced_sma)
    y0_array = LSQUnivariateSpline(
        isolist.sma, isolist.y0, nodes)(finely_spaced_sma)
    grad_array = LSQUnivariateSpline(
        isolist.sma, isolist.grad, nodes)(finely_spaced_sma)
    a3_array = LSQUnivariateSpline(
        isolist.sma, isolist.a3, nodes)(finely_spaced_sma)
    b3_array = LSQUnivariateSpline(
        isolist.sma, isolist.b3, nodes)(finely_spaced_sma)
    a4_array = LSQUnivariateSpline(
        isolist.sma, isolist.a4, nodes)(finely_spaced_sma)
    b4_array = LSQUnivariateSpline(
        isolist.sma, isolist.b4, nodes)(finely_spaced_sma)

    # Return deviations from ellipticity to their original amplitude meaning
    a3_array = -a3_array * grad_array * finely_spaced_sma
    b3_array = -b3_array * grad_array * finely_spaced_sma
    a4_array = -a4_array * grad_array * finely_spaced_sma
    b4_array = -b4_array * grad_array * finely_spaced_sma

    # correct deviations cased by fluctuations in spline solution
    eps_array[np.where(eps_array < 0.)] = 0.

    result = np.zeros(shape=shape)
    weight = np.zeros(shape=shape)

    eps_array[np.where(eps_array < 0.)] = 0.05

    # for each interpolated isophote, generate intensity values on the
    # output image array
    # for index in range(len(finely_spaced_sma)):
    for index in range(1, len(finely_spaced_sma)):
        sma0 = finely_spaced_sma[index]
        eps = eps_array[index]
        pa = pa_array[index]
        x0 = x0_array[index]
        y0 = y0_array[index]
        geometry = EllipseGeometry(x0, y0, sma0, eps, pa)

        intens = intens_array[index]

        # scan angles. Need to go a bit beyond full circle to ensure
        # full coverage.
        r = sma0
        phi = 0.
        while phi <= 2 * np.pi + geometry._phi_min:
            # we might want to add the third and fourth harmonics
            # to the basic isophotal intensity.
            harm = 0.
            if high_harmonics:
                harm = (a3_array[index] * np.sin(3. * phi)
                        + b3_array[index] * np.cos(3. * phi)
                        + a4_array[index] * np.sin(4. * phi)
                        + b4_array[index] * np.cos(4. * phi)) / 4.

            # get image coordinates of (r, phi) pixel
            x = r * np.cos(phi + pa) + x0
            y = r * np.sin(phi + pa) + y0
            i = int(x)
            j = int(y)

            if (i > 0 and i < shape[1] - 1 and j > 0 and j < shape[0] - 1):
                # get fractional deviations relative to target array
                fx = x - float(i)
                fy = y - float(j)

                # add up the isophote contribution to the overlapping pixels
                result[j, i] += (intens + harm) * (1. - fy) * (1. - fx)
                result[j, i + 1] += (intens + harm) * (1. - fy) * fx
                result[j + 1, i] += (intens + harm) * fy * (1. - fx)
                result[j + 1, i + 1] += (intens + harm) * fy * fx

                # add up the fractional area contribution to the
                # overlapping pixels
                weight[j, i] += (1. - fy) * (1. - fx)
                weight[j, i + 1] += (1. - fy) * fx
                weight[j + 1, i] += fy * (1. - fx)
                weight[j + 1, i + 1] += fy * fx

                # step towards next pixel on ellipse
                phi = max((phi + 0.75 / r), geometry._phi_min)
                r = max(geometry.radius(phi), 0.5)
            # if outside image boundaries, ignore.
            else:
                break

    # zero weight values must be set to 1.
    weight[np.where(weight <= 0.)] = 1.

    # normalize
    result /= weight

    # fill value
    result[np.where(result == 0.)] = fill

    return result