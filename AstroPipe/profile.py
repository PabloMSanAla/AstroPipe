
from AstroPipe import utils as ut
from AstroPipe.plotting import show

import numpy as np 
import os

import astropy.units as u
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.wcs.utils import pixel_to_skycoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.visualization import ImageNormalize,LogStretch

from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.aperture import RectangularAperture


import matplotlib.patches as patch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import stats
from scipy.ndimage import median_filter
from scipy.signal import medfilt, argrelextrema
from scipy.interpolate import interp1d



from lmfit.models import GaussianModel

from sklearn.cluster import KMeans



class Profile:
    '''
    Class to work with surface brightness profiles. 

    Attributes
    ----------
        rad : array
            Radii of the profile in pixels
        int : array
            Intensity of the profile [in same units as image meassured]
        intstd : array
            Standard deviation of the intensity [int units]
        pa : array
            Position angle of the profile [deg]
        eps : array
            Ellipticity of the profile [1-b/a]
        center : 2d-array
            Center of the profile [pixels]
        bkg : float
            Background of the image [int units]
        bkgstd : float
            Standard deviation of the background [int units]
        zp : float
            Zero point of the image [mag]
        pixscale : float
            Pixel scale of the image [arcsec/pixel]
        mu : array
            surface brightness magnitude of the profile [mag*arcsec^-2]
        upperr : array
            Upper limit of the surface brightness magnitude [mag*arcsec^-2]
        lowerr : array
            Lower limit of the surface brightness magnitude [mag*arcsec^-2]
        columns : list
            List of the columns of the profile to be saved in table
        units : list
            List of the units of the columns
        meta : dict
            Dictionary of the metadata of the profile to be saved in table

    Methods
    -------
        set_params(radii=None, intensity=None, instensity_err=None, pa=None, eps=None, center=None, bkg=None, bkgstd=None, zp=None, pixscale=None)
            Set the parameters of the profile
        
        load(filename)
            Load the profile from a file


    '''
    def __init__(self, filename=None,  max_radius=None, init_radius=1, growth_rate=1.01):
        
        if max_radius is not None:
            alpha = np.log10(growth_rate)
            size = np.log10(max_radius)//alpha + 2
            self.rad = init_radius*10**(alpha*np.arange(0,size))

            self.int = np.zeros_like(self.rad)
            self.intstd = np.zeros_like(self.rad)
            self.flux = np.zeros_like(self.rad)
            self.fluxstd = np.zeros_like(self.rad)
            self.npixels = np.zeros_like(self.rad)
        
        self.bkg = 0 
        self.bkgstd = 0 
        self.zp = 0 
        self.pixscale = 1

        if filename is not None: 
            self.load(filename)
        
    def __call__(self, array, hdu=0, plot=None, save=None):
        '''
        Returns the average photometric radial profile in the array
        '''
        if array is str: array = fits.getdata(array, hdu)

        profile = elliptical_radial_profile(array, self.rad, (self.x, self.y), self.pa, self.eps, 
                                    plot=plot, save=save)
        
        profile.set_params(bkg=self.bkg, bkgstd=self.bkgstd, 
                           zp=self.zp, pixscale=self.pixscale)
        profile.brightness()

        return profile
    
    def set_params(self, 
            radii=None, intensity=None, instensity_err=None, 
            flux=None, fluxstd=None, npixels=None,
            pa=None, pastd=None, eps=None, epsstd=None, center=None, 
            bkg=None, bkgstd=None, zp=None, pixscale=None):

        
        if radii is not None: self.rad = radii
        if intensity is not None: self.int = intensity
        
        # This conversion variable is only to create arrays of static pa, eps, and center
        if np.size(pa) == np.size(eps) == 1:
            conversion = np.ones_like(self.rad)
            self.type = 'fixed'
        else: 
            conversion, self.type = 1, 'dynamic'
        
        if instensity_err is not None: self.intstd = instensity_err
        
        if pa is not None: self.pa = pa*conversion
        if pastd is not None: self.pastd = pastd*conversion
        if eps is not None: self.eps = eps*conversion
        if epsstd is not None: self.epsstd = epsstd*conversion
        if center is not None and np.array(center).size==2: 
            self.x = center[0]*conversion
            self.y = center[1]*conversion
        elif center is not None: self.x, self.y = center
        
        if flux is not None: self.flux = flux
        if fluxstd is not None: self.fluxstd = fluxstd
        if npixels is not None: self.npixels = npixels

        if bkg is not None: self.bkg = bkg
        if bkgstd is not None: self.bkgstd = bkgstd 
        if zp is not None: self.zp = zp
        if pixscale is not None: self.pixscale = pixscale

        self.columns = ['radius', 'intensity', 'intensity_err', 'flux','flux_err',
                        'npixels','pa', 'pa_err', 'eps', 'eps_err', 'x', 'y']
        self.units = ['arcsec', 'counts', 'counts','counts', 'counts','na', 
                      'deg', 'deg', 'na', 'na', 'pixel', 'pixel']
        self.meta = {'zp':zp, 'pixscale':pixscale, 'bkg':bkg , 'bkgstd':bkgstd}

        if zp is not None and pixscale is not None and any(self.int>0):
            self.brightness()
        
    
    def brightness(self, zp=None, bkg=None, pixscale=None,  bkgstd=None):

        zp = self.zp if zp is None else zp
        bkg = self.bkg if bkg is None else bkg
        bkgstd = self.bkgstd if bkgstd is None else bkgstd
        pixscale = self.pixscale if pixscale is None else pixscale
        
        self.mu, self.upperr, self.lowerr = get_surface_brightness(
            self.rad, self.int,  self.intstd, bkg, bkgstd, pixscale, zp)
    
    def morphology(self, level):
        '''returns the morphology of the profile
            at a surface brightness level'''
        if not hasattr(self, 'mu'): self.brightness()
        arg = ut.closest(self.mu, level)
        pa = np.median(self.pa[arg-2:arg+8])
        eps = np.median(self.eps[arg-2:arg+8])
        return self.rad[arg], pa, eps
    
    def get_magnitude(self):
        '''TODO: Create function that computes the
        asymptotic magnitute of the profile
        '''
        return False

    def skycenter(self, WCS):
        self.ra, self.dec = pixel_to_skycoord(self.x, self.y, WCS)
    
    def plot(self, axes=None, color='r', label=None, **kwargs):
        label = self.type if label is None else label
        fig = plot_profile(self.rad*self.pixscale, self.mu, self.pa, self.eps, self.upperr, self.lowerr, 
                           axes=axes, color=color, label=label, **kwargs)
        return fig

    def extend(self, array, max_radius, growth_rate=None):
        '''Extends the radial profile to a new maximum radius'''
        if growth_rate is None: growth_rate = np.nanmedian(self.rad[1:]/self.rad[:-1])
        alpha = np.log10(growth_rate)
        n = np.log10(max_radius/self.rad[-1])//alpha + 1
        new_rad = self.rad[-3]*10**(alpha*np.arange(1,n+1))
        new_prof = elliptical_radial_profile(array, new_rad, (self.x[-1], self.y[-1]), self.pa[-1], self.eps[-1])
        self.rad = np.concatenate((self.rad, new_rad[2:]))
        self.int = np.concatenate((self.int, new_prof.int[2:]))
        self.intstd = np.concatenate((self.intstd, new_prof.intstd[2:]))
        self.pa = np.concatenate((self.pa, new_prof.pa[2:]))
        self.eps = np.concatenate((self.eps, new_prof.eps[2:]))
        self.x = np.concatenate((self.x, new_prof.x[2:]))
        self.y = np.concatenate((self.y, new_prof.y[2:]))
        self.brightness()
    
    def remove_nans(self):
        if not hasattr(self, 'mu'): self.brightness()
        index = np.isnan(self.int) + np.isnan(self.rad) + np.isnan(self.mu)
        self.rad = self.rad[~index]
        self.int = self.int[~index]
        self.intstd = self.intstd[~index]
        self.flux = self.flux[~index]
        self.fluxstd = self.fluxstd[~index]
        self.npixels = self.npixels[~index]
        self.pa = self.pa[~index]
        self.eps = self.eps[~index]
        self.x = self.x[~index]
        self.y = self.y[~index]
        self.brightness()
    
    def interpolateCurve(self, var1, var2, nElements=10000, kind='linear'):
        '''
        Enhances resolution of the curve of growth via interpolation.

        Parameters
        ----------
        nElements : int
            Number of resolution elements for interpolated curve
        kind : string
            Interpolation type.  Valid values are: ‘linear’, ‘nearest’,
            ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            or ‘next’.

        Returns
        -------
        high_res_var1 : numpy.ndarray
            Variable 1 with resolution defined by nElements
        high_res_var2 : numpy.ndarray
            Variable 2 with resolution defined by nElements
        '''
        f = interp1d(var1, var2, kind=kind)
        high_res_var1 = np.linspace(np.min(var1), np.max(var1), nElements)
        high_res_var2 = f(high_res_var1)

        return high_res_var1, high_res_var2

    def curveOfGrowth(self, sky=None):
        '''
        Produces a curve of growth in magnitudes

        Parameters
        ----------
        sky : float, optional
            Estimate of the average sky flux local to the galaxy.
            The default is 0.

        Returns
        -------
        sma : numpy.ndarray
            Semi-major-axis array, in pixels
        mags : numpy.ndarray
            Total magnitude enclosed within sma
        '''
        if sky is None: sky = self.bkg

        sma = self.rad
        tflux = self.flux - self.npixels*sky
        mags = -2.5*np.log10(tflux) + self.zp

        return sma, mags

    def totalMagnitude(self, sma, mags, npoints=10):
        '''
        Computes total magnitude of galaxy using curve of growth

        Parameters
        ----------
        sma : numpy.ndarray
            Semi-major-axis array, in pixels
        mags : numpy.ndarray
            Total magnitude enclosed within sma
        npoints : int, optional
            Number of points to use for fitting. The default is 10.

        Returns
        -------
        totalMag : float
            Total magnitude extrapolated to infinity

        NOTE: fitting limits currently hard-coded, but proven effective for
        exponential profiles
        '''
        slope = ut.localSlope(sma, mags)
        # want = (slope >= -0.008) & (slope <= 0)  # Typical useful range
        want = (sma>sma[-npoints])
        fit = np.polyfit(slope[want], mags[want], 1)
        totalMag = fit[1]

        return totalMag

    def fractionalRadius(self, totalMag, fluxFrac=0.5):
        '''
        Computes the radius containing some fraction of the galaxy's total
        light.

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        fluxFrac : float, optional
            The desired fraction of enclosed light. The default is 0.5, i.e.
            by defaul this returns the half-light radius.

        Returns
        -------
        fracRad : float
            The radius containing the desired fraction of the galaxy's total
            light, in pixels
        '''
        totalFlux = 10**(-0.4*(totalMag - self.zp))
        findFlux = totalFlux * fluxFrac

        # A little crude, but we interpolate to improve the radius resolution
        sma, cog = self.interpolateCurve(self.rad,
                                         self.flux-self.npixels*self.bkg)

        idx = ut.closest(cog, findFlux)
        fracRad = sma[idx]

        return fracRad

    def concentration(self, totalMag, f1 = 0.8, f2=0.2):
        '''
        Derives the concentration parameter C discussed by Conselice (2003)
        between fraction f1 and f2. Default values are the concentration index
        originally proposed by Kent (1985), ApJS, 59, 115. [C82]

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        f1 : float, optional
            Outer fraction level. The default is 0.8. [0-1]
        f2 : float, optional
            Inner fraction level. The default is 0.2. [0-1]

        Returns
        -------
        c82 : float
            The concentration parameter C = 5log(R_f1/R_f2)
        '''
        r1 = self.fractionalRadius(totalMag, f1)
        r2 = self.fractionalRadius(totalMag, f2)
        c = 5*np.log10(r1/r2)
        return c

    def isophotalRadius(self, SbMu, sky=None, returnMorph=False):
        '''
        Get radius at a level of surface brightness

        Parameters
        ----------
            SbMu : float
                Surface brightness level
        Returns
        -------
            rad : float
                Radius at the given surface brightness level [arcseconds]
        '''

        if not hasattr(self, 'mu') and not sky: 
            self.brightness()
    
        if sky: 
            mu = self.zp - 2.5*np.log10(self.int - sky) + 5*np.log10(self.pixscale)
        else: mu = np.zeros_like(self.mu) + self.mu

        sma, sb = self.interpolateCurve(self.rad,
                                        mu)
        if returnMorph: 
            _, pa = self.interpolateCurve(self.rad, self.pa)
            _, eps = self.interpolateCurve(self.rad, self.eps)
    
        idx = ut.closest(sb, SbMu)
        output = sma[idx]*self.pixscale
        if returnMorph: output = [output, pa[idx], eps[idx]]

        return output

    def concentrationRe(self, totalMag, alpha=0.7):
        '''
        Derived the concentration parameter proposed by Trujillo et al. (2001)
        MNRAS, 326, 869

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        alpha : float
            Factor by which to define outer isophote level, as alpha*rEff

        Returns
        -------
        cRe : float
            The concentration parameter sum(I[<alpha*Reff])/sum(I[<Reff])
        '''
        rEff = self.fractionalRadius(totalMag, 0.5)

        sma, cog = self.interpolateCurve(self.rad,
                                         self.flux-self.npixels*self.bkg)

        idx = ut.closest(sma, rEff)
        idy = ut.closest(sma, alpha*rEff)

        sum1 = cog[idy]
        sum2 = cog[idx]

        cRe = sum1/sum2

        return cRe

    def petrosianRadius(self, eta=0.2, sky=0):
        '''
        Derives Petrosian radius (Petrosian 1976).

        Parameters
        ----------
        eta : float, optional
            Value of Petrosian index used to define Petrosian radius.
            The default is 0.2 (Bershady et al. 2000)

        Returns
        -------
        radPetro : float
            Petrosian radius in pixels

        NOTE: quick check, for an exponential profile, R_petrosian ~
        2x R_eff.
        '''
        if sky==0: sky = self.bkg

        sma, sb = self.interpolateCurve(self.rad,
                                        self.int-sky)
        __, cog = self.interpolateCurve(self.rad,
                                        self.flux-self.npixels*sky)
        __, area = self.interpolateCurve(self.rad,
                                         self.npixels)
        petrosian = sb * (area/cog)
        idx = ut.closest(petrosian, eta)
        radPetro = sma[idx]

        return radPetro

    def write(self, filename=None, overwrite=True):
        self.meta = {'zp':self.zp, 'pixscale': self.pixscale, 
                     'bkg':self.bkg , 'bkgstd': self.bkgstd}
        
        if filename is None: filename = 'profile.txt'
        
        table = Table([self.rad, self.int, self.intstd, 
                       self.flux, self.fluxstd, self.npixels,
                       self.pa, self.eps, self.x, self.y], 
                        names=self.columns,
                        units=self.units, meta=self.meta)
        table.write(filename, overwrite=overwrite)
        return os.path.isfile(filename)
    
    def load(self, filename):
        table = Table.read(filename)
        self.set_params(np.array(table['radius'].value), 
                        np.array(table['intensity'].value), np.array(table['intensity_err'].value), 
                        np.array(table['pa'].value), np.array(table['eps'].value), 
                        (np.array(table['x'].value), np.array(table['y'].value)), 
                        table.meta['BKG'], table.meta['BKGSTD'], table.meta['ZP'], table.meta['PIXSCALE'])
        self.table = table

def plot_profile(radius, mu, pa, eps, mupper=None, mlower=None, axes=None, color='k', label=None, **kwargs):
    '''
    Function to plot the surface brightness profile of a galaxy
    and the elliptical parameters used in the photometry (eps and pa).
    It also plots the upper and lower limits of the profile.

    Parameters
    ----------
        radius : array
            Radius of the profile [arcsec]
        mu : array
            Surface brightness of the profile [mag*arcsec^-2]
        pa : float
            Position angle of the galaxy [deg]
        eps : float
            Ellipticity of the galaxy 
        mupper : array
            Upper error of the surface brightness [mag*arcsec^-2]
        mlower : array
            Lower error of the surface brightness [mag*arcsec^-2]
        ax : tuple of matplotlib.axes
            Axes to plot the profile, eps and pa
        color : str
            Color of the profile
        label : str
            Label of the profile
        **kwargs : dict
            Additional arguments to pass to the plot function
    
    Returns
    -------
        fig : matplotlib.figure
            Figure with the profile and the parameters
    '''

    if axes is None: 
        fig = plt.figure(figsize=(5,6))
        axeps = plt.subplot2grid((5,1),(4,0))
        axmu = plt.subplot2grid((5,1),(0,0),rowspan=3,sharex=axeps)
        axpa = plt.subplot2grid((5,1),(3,0),sharex=axeps)
    else:
        axmu, axpa, axeps = axes
        fig = axmu.get_figure()

    # Surface Brightness plot
    axmu.plot(radius, mu, color=color, label=label, **kwargs)
    if mupper is not None: axmu.plot(radius, mupper, color=color, ls='--')
    if mlower is not None: axmu.plot(radius, mlower, color=color, ls='--')
    axmu.set_ylabel('$\mu\,[\mathrm{mag\,arcsec}^{-2}]$',fontsize=14,labelpad=3)
    axmu.invert_yaxis()

    # Positional Angle plot
    pa[pa>180] = pa[pa>180] - 180
    pa[pa<0] = pa[pa<0] + 180
    axpa.plot(radius, pa, color=color, **kwargs)
    axpa.set_ylabel('PA (deg)',fontsize=12, labelpad=3)
    axpa.set_ylim([-10,190]); axpa.set_yticks([0,45,90,135,180])
    axpa.locator_params(axis='y',nbins=3, tight=True)
    axpa.tick_params(labelsize=12)
    #axpa.yaxis.set_label_position('right')

    # Epsilon plot
    axeps.plot(radius, eps, color=color, **kwargs)
    axeps.set_ylabel('$\\epsilon$',fontsize=12,labelpad=3)
    axeps.set_xlabel('Radius (arcsec)',fontsize=14,labelpad=3)
    axeps.set_ylim([-0.1,1.1]); axeps.set_yticks([0,0.5,1])
    axeps.tick_params(labelsize=12)

    for ax in [axmu, axpa]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in [axmu,axpa,axeps]:
        ax.grid(ls='--',alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)
    return fig

def get_surface_brightness(rad, intensity,  intensitystd, bkg, bkgstd, pixscale, zp):
    '''
    Function to calculate the surface brightness from the intensity.
    It also computes the lower and upper uncertanties due to the 
    background and intensity std. It interpolates the nan values. 
    '''
    mag = zp -2.5*np.log10(intensity - bkg) + 5*np.log10(pixscale)
    lowerr = zp - 2.5*np.log10(intensity - bkg + intensitystd + bkgstd) + 5*np.log10(pixscale)
    upperr = zp - 2.5*np.log10(intensity - bkg - intensitystd - bkgstd) + 5*np.log10(pixscale)
    if any(np.isnan(lowerr)): lowerr[np.isnan(lowerr)] = np.interp(rad[np.isnan(lowerr)], rad[~np.isnan(lowerr)], lowerr[~np.isnan(lowerr)])
    if any(np.isnan(upperr)): upperr[np.isnan(upperr)] = np.interp(rad[np.isnan(upperr)], rad[~np.isnan(upperr)], upperr[~np.isnan(upperr)])
    maxdiff = np.nanmax([np.abs(mag-lowerr), np.abs(mag-upperr)])
    upperr[upperr<mag] = mag[upperr<mag] + maxdiff
    lowerr[lowerr>mag] = mag[lowerr>mag] - maxdiff
    return mag, upperr, lowerr

def surface_photometry(data, mask, center, growth_rate=1.03,
                        plot=None, verbose=False):
    '''
    Function that analyses the surface photometry of a galaxy image.
    TODO:It will:
        (1) Compute the morphological parameters, ellipticity and positon angle
        (2) Compute the background and radius where it reaches it
        (3) Compute three radial profiles:
            (3.1) Variable ishophotal radial profile
            (3.2) Fixed Elliptical radial profile
            (3.3) Rectangular radial profile
        (4) Small analysis to improve the profiles 
        (5) Returns a table with the results
    
    '''
    table = []
    return table


def elliptical_radial_profile(data, rad, center, pa, eps, growth_rate=1.03,
                        plot=None, save=None):
    '''
    Sigma clipped average elliptical radial profile of a galaxy.
    It uses the elliptical apertures to get the radial profile.
    
    Parameters
    ----------
        data : 2D array
            Image data.
        rad : array or float
            Radius of the apertures or maximum radius of aperture.
        center : tuple
            Center of the galaxy (x,y).
        pa : float
            Position angle of the galaxy.
        eps : float
            Ellipticity of the galaxy [1-b/a]
        max_r : float
            Maximum radius of the profile.
        growth_rate : float
            Growth rate of the apertures.
        plot : str
            If given it will save the plot in the given path.
        save : str
            If given it will save the profile in the given path.
    
    Returns
    -------
        profile : AstroPipe.sbprofile.Profile
            Profile object with the radial profile.
   '''

    if type(rad) not in [np.ndarray, list]:
        profile = Profile(max_radius=rad, growth_rate=growth_rate)
        profile.set_params(pa=pa, eps=eps, center=center)
    else:
        profile = Profile()
        profile.set_params(pa=pa, eps=eps, center=center, radii=rad,
                           intensity=np.zeros_like(rad), instensity_err=np.zeros_like(rad))

    # TODO: If keyword rad is givem maybe extrapolate pa, and eps¿?
    # if rad is not None:
    #     profile.pa = np.interp(rad, profile.rad, profile.pa)
    #     profile.eps = np.interp(rad, profile.eps, profile.eps)
    #     profile.set_params(center=(center[0], center[1]))
    #     profile.center = np.interp(rad, profile.rad, profile.center)

    ellip_apertures = []

    previous_mask = np.zeros_like(data)

    for i,rad in enumerate(profile.rad):
        
        if i>0:
            profile.rad[i] = (profile.rad[i-1] + profile.rad[i])/2
        # generate astropy aperture
        ellip_apertures.append(EllipticalAperture(
            (profile.x[i],profile.y[i]), rad, 
            (1-profile.eps[i])*rad, profile.pa[i]*np.pi/180))
        
        # create mask and index list of the aperture and updates previous one 
        mask = ellip_apertures[-1].to_mask(method='center').to_image(data.shape)
        index = (data.mask==False) * (mask!=0) * (previous_mask==0)
        previous_mask = mask

        # compute sigma clipped median of the aperture 
        clipped = sigma_clip(data.data[index], sigma=2.5, maxiters=3)
        profile.int[i] = np.ma.median(clipped)
        profile.intstd[i] = np.nanstd(clipped)/np.sqrt(np.size(clipped))
        if np.isnan(profile.intstd[i]): profile.intstd[i] = profile.intstd[i-1] 
        
        # integrated photometry 
        profile.npixels[i] =  profile.npixels[i-1] + np.size(data.data[index])
        profile.flux[i] =  profile.flux[i-1] + np.nansum(data.data[index])
        profile.fluxstd[i] = np.sqrt(profile.fluxstd[i-1]**2 +  (np.sqrt(profile.npixels[i-1])*profile.intstd[i-1])**2)

    if plot is not None:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        show(data, ax=ax)
        for ap in ellip_apertures:
            ap.plot(color='black', alpha=0.6,lw=0.5, axes=ax)
        for val,lim in zip([profile.x[0], profile.y[0]],
                            [ax.set_xlim, ax.set_ylim]):
            lim([val-1.1*profile.rad[-1], val+1.1*profile.rad[-1]])
        fig.savefig(plot, dpi=200, bbox_inches='tight', pad_inches=0.1)

    if save is not None:
        profile.write(save.split('.')[-2]+'_static.fits',overwrite=True)

    return profile


def isophotal_photometry(data, center, pa, eps, reff,  max_r=None, growth_rate=1.03,
                        fix_center=False, fix_pa=False,fix_eps=False, plot=None, save=None):
    '''
    Sigma clipped average radial profile of a galaxy.
    It uses the elliptical apertures to get the radial profile.
    
    Parameters
    ----------
        data : 2D array
            Image data.
        center : tuple
            Center of the galaxy.
        pa : float
            Guess of the position angle of the galaxy.
        eps : float
            Guess ellipticity of the galaxy [1-b/a]
        max_r : float
            Maximum radius of the profile.
        growth_rate : float
            Growth rate of the apertures.
        plot : str
            If given it will save the plot in the given path.
        save : str
            If given it will save the profile in the given path.
    
    Returns
    -------
        profile : AstroPipe.sbprofile.Profile
            Profile object with the radial profile.
   '''
    pa = pa * np.pi/180
    step = growth_rate - 1
    
    guess_aper = EllipseGeometry(x0=center[0], y0=center[1],
                            sma=reff, eps=eps, pa=pa)

    ellipse = Ellipse(data, guess_aper)

    isolist = ellipse.fit_image(reff, integrmode='median',sclip=2.5, nclip=3, maxsma=max_r,
                minsma=1,step=step,fix_center=fix_center,fix_pa=fix_pa,fix_eps=fix_eps)
    
    fluxstd = np.sqrt(isolist.npix_e)*isolist.int_err

    profile = Profile()
    profile.set_params(radii=isolist.sma, intensity=isolist.intens, instensity_err=isolist.int_err,
        flux=isolist.tflux_e, fluxstd=fluxstd, npixels=isolist.npix_e,
        pa=isolist.pa*180/np.pi, pastd=isolist.pa_err*180/np.pi,
        eps=isolist.eps, epsstd=isolist.eps_err, center=(isolist.x0, isolist.y0))
    
    if plot is not None:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        show(data, ax=ax)
        for i,rad in enumerate(profile.rad):
            ap = EllipticalAperture(
            (profile.x[i], profile.y[i]), rad, 
            (1-profile.eps[i])*rad, profile.pa[i]*np.pi/180)
            ap.plot(color='black', alpha=0.6,lw=0.5, axes=ax)
        for val,lim in zip([profile.x[0], profile.y[0]], 
                            [ax.set_ylim, ax.set_xlim]):
            lim([val-1.1*max_r,val+1.1*max_r])
        fig.savefig(plot, dpi=200, bbox_inches='tight', pad_inches=0.1)

    if save is not None:
        profile.write(save, overwrite=True)

    return profile


def rectangular_radial_profile(data, rad, center, pa, width=2, growth_rate=1.01, 
                        plot=None, save=None):
    
    '''
    Sigma clipped average rectangular radial profile of a galaxy.
    It uses the rectangular apertures to get the radial profile.
    
    Parameters
    ----------
        data : 2D array
            Image data.
        rad : array or float
            Radius of the apertures or maximum radius of aperture.
        center : tuple
            Center of the galaxy (x,y).
        pa : float
            Position angle of the galaxy.
        widht : float
            Width of the rectangular apertures.
        growth_rate : float
            Growth rate of the apertures.
        plot : str
            If given it will save the plot in the given path.
        save : str
            If given it will save the profile in the given path.
    
    Returns
    -------
        profile : AstroPipe.sbprofile.Profile
            Profile object with the radial profile.
   '''

    if type(rad) not in [np.ndarray, list]:
        profile = Profile(max_radius=rad, growth_rate=growth_rate)
        profile.set_params(pa=pa, eps=0, center=center)
    else:
        profile = Profile()
        profile.set_params(radii=rad, pa=pa, eps=0, center=center,
                           intensity=np.zeros_like(rad), instensity_err=np.zeros_like(rad))

    # TODO: instead of fixed width apertures use increasingly large apertures to improve S/N at large radii


    rect_apertures, std = [], 0 

    previous_mask = np.zeros_like(data)

    for i,rad in enumerate(profile.rad):

        if len(rect_apertures) > 1:
            previous_mask = mask
            std = np.nanmedian(profile.intstd[-1])

        rect_apertures.append(RectangularAperture(
            (profile.x[i],profile.y[i]), 2*rad, width, pa*np.pi/180))
        mask = rect_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = (data.mask==False) * (mask!=0) * (previous_mask==0)
        clipped = sigma_clip(data.data[index], sigma=2.5, maxiters=3)
        profile.int[i] = np.ma.median(clipped)
        profile.intstd[i] = np.nanstd(clipped)/np.sqrt(np.size(clipped))
        
        # if np.nanmedian(profile.intstd[-1]) > 3*std:
        #     width *= 1.5
        #     std = np.nanmedian(profile.intstd[-1])

    if plot is not None:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        show(data, ax=ax)
        for ap in rect_apertures:
            ap.plot(color='black', alpha=0.6,lw=0.5, axes=ax)
        for val,lim in zip([profile.x[0], profile.y[0]],
                            [ax.set_xlim, ax.set_ylim]):
            lim([val-1.1*profile.rad[-1], val+1.1*profile.rad[-1]])
        fig.savefig(plot, dpi=200, bbox_inches='tight', pad_inches=0.1)

    if save is not None:
        profile.write(save.split('.')[-2]+'_rectangular.fits',overwrite=True)

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

def random_rectangular_boxes(center, pa, sma, eps, n=5, wbox=30):
    ecc = np.sqrt(1-(1-eps)**2)
    omegas = np.linspace(0,1.9*np.pi,n)
    r = (sma*(1-eps))/np.sqrt(1-(ecc*np.cos(omegas))**2)
    dx = r*np.cos(omegas)*(1+np.random.randint(2,10,n)/100)
    dy = r*np.sin(omegas)*(1+np.random.randint(2,10,n)/100)
    dx,dy = ut.change_coordinates(np.array([dx,dy]),np.array((0,0)),pa)
    centers = np.array([center[0]+dx,center[1]+dy])
    
    rectangles = RectangularAperture(centers.T,wbox, wbox)
    return rectangles


def background_estimation(data, center, pa, eps, growth_rate = 1.03, out=None, verbose=False):
    '''
    Estimate the local background of a galaxy using elliptical apertures.
    
    It detects the maximum radius of the object containing signal with some 
    statistical tests [bkg_radius]. Then rectangular apertures photometry
    to estimate the background value and its uncertainty. 
    
    The apertures are created using the center, position angle, and ellipticity
    given. Then, the profile is meassure until certain radius. Once we reach 
    intensity values close to the mode + sigma, we increase the resolution
    (growth_rate=1.01), and compute the smooth derivative of the profile. 
    
    We stablish the value of the radius where the background is reach 
    this conditions are met:
        
        - The derivative change its sign for more than five times 
                This is a meassure of the fluctuation around dI = 0 
                when looking for a local assymptote.
        
        - Or when: I(r)/mode - 1 < epsilon [epsilon = 0.1]
                Safe parameter to avoid infinite loops.

    TODO: To faster the process, first compute distances to mode 
    +- sigma, and start in the closest or (q > 25 ¿?), and maybe 
    limit to (q < 75 ¿?).
    
    TODO: first guess background with sigma clipping distribution, 
    then do photometry with elliptical apertures increasing radius 
    and check evidence of convergence (or Kolmogorov–Smirnov test)
    and stop when next aperture is not signficante more similar. 
    Combine with current method.

    TODO: give option of sigma image to do statistical comparison of 
    value optain with this, to see if the are of the same order, if not
    give warning.

    TODO: What is the best estimation of the uncertainty of the background
    meassure here?
        - Is it the std/np.sqrt(N_rect) 
        - Is it the rms or rms normalize by the square root?
    Do tests to see which is the best.

    Parameters
    ----------
        data : 2D numpy array
            Image of the galaxy.
        center : tuple
            Center of the galaxy in pixel coordinates.
        pa : float
            Position angle of the galaxy in degrees.
        eps : float
            Ellipticity of the galaxy. [1-b/a]
        growth_rate : float, optional
            Growth rate of the apertures. The default is 1.03.
        out : str, optional
            Path to save the background profile. The default is None.
        verbose : bool, optional
            Print the results. The default is False.
    
    Returns
    -------

        localsky : float
            Background value.
        localsky_std : float
            Background uncertainty.
        bkg_radius:  float
            Limiting radius where the background is reached.
    '''
    if not hasattr(data,'mask'): data = np.ma.array(data, mask=np.zeros_like(data))
    pa = pa*np.pi/180 # convert to radians
    
    # First guess of the mode value
    flatten = data[data.mask==False].flatten()
    flatten = flatten[np.isfinite(flatten)]
    mode, results = find_mode(flatten)
    std = results.values['sigma']

    rad = 10*np.ones(1)
    intensity = np.zeros(0)
    intensity_std = np.zeros(0)
    ellip_apertures = []
    previous_mask = np.zeros_like(data.mask)
    converge = False
    maxr,epsilon = np.NaN, 1e-1

    while not converge:

        if len(ellip_apertures) > 1:
            previous_mask = mask

        ellip_apertures.append(EllipticalAperture((center[0],center[1]), rad[-1], (1-eps)*rad[-1], pa))
        mask = ellip_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = ut.where([data.mask==False,mask!=0,previous_mask==0])
        clipped = sigma_clip(data.data[index],sigma=3,maxiters=3)
        intensity= np.append(intensity, np.ma.median(clipped))
        intensity_std = np.append(intensity_std, np.nanstd(clipped)/np.sqrt(np.size(clipped)))
    
        if (intensity[-1] < mode + std) and np.isnan(maxr):

            growth_rate = 1.01
            index = intensity < mode + std
            dIdr = ut.derivative(rad[index],intensity[index])
            signs = np.sign(dIdr[1:]/dIdr[:-1]) == -1

            if np.sum(signs) > 5 or (np.abs(intensity[-1]/mode - 1) < epsilon):
                maxr = asymtotic_fit_radius(rad[index],dIdr)
                if verbose: print(f'maxr={maxr:.2f}; rad={rad[-1]:.2f}; intesity={intensity[-1]:.2f}')

        if rad[-1] > 1.3*maxr:
            converge = True
            if verbose: print(maxr,len(rad),len(intensity))
            break

        rad = np.append(rad, rad[-1]*growth_rate)


    dIdr = ut.derivative(rad,intensity)
    ddIdr2 = ut.derivative(rad,dIdr)

    index = intensity < mode + std
    skyradius1 = asymtotic_fit_radius(rad[index],dIdr[index])
    skyradius2 = asymtotic_fit_radius(rad[index],ddIdr2[index])

    skyradii = np.sort([skyradius1,skyradius2])
    aperfactor = np.nanmax([0.01,float(np.round((60 - np.diff(skyradii)) / (np.sum(skyradii)),3))])
    width = float(np.nanmax([np.diff(skyradii),60]))


    bkg_aperture = EllipticalAnnulus((center[0],center[1]),
                     (1-aperfactor)*skyradii[0], (1+aperfactor)*skyradii[1], 
                    (1-0.6*eps)*(1-aperfactor)*skyradii[0], None,
                    pa)

    # Measure the background using the elliptical annulus
    mask_aper = bkg_aperture.to_mask(method='center').to_image(data.shape)
    mask_aper = np.ma.array(mask_aper,mask=1-mask_aper)
    aper_values = data*mask_aper
    aper_values = aper_values[np.where(~aper_values.mask)].flatten()
    localsky, gauss_fit = find_mode(aper_values)  
    aper_bkg = gauss_fit.params['center'].value
    aper_bkgstd = gauss_fit.params['sigma'].value/np.sqrt(aper_values.size)


    # Renctangular Apertures
    n_boxes = np.int64(3*np.mean(skyradii)/(width))
    width_boxes = width*0.8
    rect = random_rectangular_boxes(center,-pa, np.mean(skyradii), 0.6*eps, n=n_boxes, wbox=width_boxes)

    res_stats = []
    for r in rect:
        mask_r =r.to_mask(method='center').to_image(data.shape)
        aper_val = data[(~data.mask) * (mask_r==1)]
        mean,med,std = sigma_clipped_stats(aper_val.flatten())
        res_stats.append(med)

    # Use rectangular localsky estimate
    localsky = np.nanmedian(res_stats)
    localsky_std = np.nanstd(res_stats)/np.sqrt(np.sum(~np.isnan(res_stats)))

    # Plotting
    if out is not None: 
        fig = plt.figure(figsize=(12,8))
        ax1 = plt.subplot2grid((4,3),(1,0))
        ax2 = plt.subplot2grid((4,3),(2,0),sharex=ax1)
        ax3 = plt.subplot2grid((4,3),(3,0),sharex=ax1)
        ax4 = plt.subplot2grid((4,3),(1,1),rowspan=3,colspan=2)
        axtext = plt.subplot2grid((4,3),(0,0),colspan=2)
        axtext.axis('off')
        
        fontsize = 12
        ax1.plot(rad,intensity,'.')
        ax1.set_ylabel('Intensity (ADUs)',fontsize=fontsize)
        ax1.axhline(mode,ls='--',c='k',label='Mode')
        ax1.axhline(mode+std,ls=':',c='k',label='Mode $\pm \sigma$')
        ax1.axhline(aper_bkg, ls='-.', c='magenta', label='Aperture')
        ax1.axhline(localsky, ls='-.', c='green', label='Boxes')
        ax1.set_ylim([mode-std*0.3,mode+std*1.4])
        arglim = np.nanargmin(np.abs(intensity-mode-std))
        ax1.set_xlim([rad[arglim],1.1*np.nanmax(np.append(rad[-1],skyradii))])
        ax1.legend(fontsize=10,loc='upper left',ncol=2, frameon=False)

        ax2.plot(rad,dIdr,'.')
        ax2.set_ylabel('$dI/dr$',fontsize=fontsize)
        ax2.axhline(0,ls='--',c='k')
        ax2.axvline(skyradius2,ls='--',c='r',label='Sign change')
        ax2.set_ylim([dIdr[arglim],-dIdr[arglim]/10])

        ax3.plot(rad,ddIdr2,'.')
        ax3.set_ylabel('$d^2I/dr^2$',fontsize=fontsize)
        ax3.set_xlabel('Radius (pixels)',fontsize=fontsize)
        ax3.axhline(0,ls='--',c='k')
        ax3.axvline(skyradius1,ls='--',c='r',label='Sign change')
        ax3.set_ylim([-ddIdr2[arglim]/10,ddIdr2[arglim]])

        for ax in [ax1,ax2,ax3]:
            ax.axvline(bkg_aperture.a_in,ls='-.',c='magenta',label='Sky annulus')
            ax.axvline(bkg_aperture.a_out,ls='-.',c='magenta')       
            ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0)) 

        show(data-localsky, vmin=gauss_fit.params['sigma'].value, ax=ax4)
        width=1.2*skyradii[1]
        ax4.set_xlim([center[0]-width,center[0]+width])
        ax4.set_ylim([center[1]-width,center[1]+width])
        bkg_aperture.plot(axes=ax4,color='black',lw=1)
        

        text = f'''\
Background estimation using elliptical apertures of \
{os.path.basename(os.path.splitext(out)[0])} 
_____________________________________________________________________________

center = ({center[0]:.2f} , {center[1]:.2f})     \
PA = {pa*180/np.pi:.2f} degrees [x to y]     \
Ellipticity = {eps:.2f}

Background Meassured in an elliptical annulus of width {skyradii[1]-skyradii[0]:.2f} pixels
Mode  = {aper_bkg:.3e} +- {gauss_fit.params['center'].stderr:.3e}      \
Sigma = {gauss_fit.params['sigma'].value:.3e} +- {gauss_fit.params['sigma'].stderr:.3e}
Background = {aper_bkg:.3e} +- {aper_bkgstd:.3e}

Background Meassured in {n_boxes:d} boxes of width {width_boxes:.2f} pixel \
at a distance of {float(np.nanmax([rad[-1],maxr])):.2f} pixels from the center
Background = {localsky:.3e} +- {localsky_std:.3e}
'''

        axtext.text(0.05, 1.05, text,
            ha='left', va='top', transform=axtext.transAxes, fontsize=10)
        rect.plot(axes=ax4,color='black')
        fig.subplots_adjust(hspace=0.4)
        # ax4.text(0.02, 1.05, os.path.basename(os.path.splitext(out)[0]), ha='left',
        #         va='bottom', transform=ax4.transAxes, fontweight='bold',fontsize='large')
        fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)

    return localsky, localsky_std, float(np.nanmax([rad[-1],maxr]))



def res_sum_squares(dmdr, cog, slope, abcissa):

    y2 = abcissa+slope*dmdr    
    rms = np.mean(np.sqrt((cog-y2)**2))

    return rms, y2 

def asymtotic_fit_radius(x,y):
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

# Find radius where an asintote starts in a profile
def find_radius_asintote(x,y):
    # Find the first point where the slope of the line is 0
    # This is the point where the asintote starts
    # x,y are the profile
    # Returns the radius where the asintote starts

    # Find the slope of the line between each point
    slope = ut.derivative(x,y)

    # Find the first point where the slope is 0
    # This is the point where the asintote starts
    # This is the point where the slope changes sign
    sign_change = np.where(np.diff(np.sign(slope)))[0][0]

    # Return the radius where the asintote starts
    return x[sign_change]
    
def find_mode(data):
    mean, median, std = sigma_clipped_stats(data,sigma=3)
    hist, bin_edges = np.histogram(data,bins=1000,
                         range=[median - 5*std, median + 5*std])
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    model = GaussianModel()
    params = model.guess(hist,x=bin_centers)
    result = model.fit(hist,params,x=bin_centers)    

    return  result.values['center'], result



def aaron_break_finder(rad,mu,min=21,max=31,n=4,p=5):
    """
    Finds the disk breaks in the surface brightness profile as 
    seen in Watkins et al. (2019)
    """
    index = ut.where([mu>min,mu<max,np.isfinite(mu)])[0]
    der = ut.derivative(rad[index],mu[index],n=n)
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