import os
import sys
import numpy as np
from termcolor import colored
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import cv2
import fnmatch
from copy import deepcopy
from fabada import fabada
import argparse
from math import gamma

from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from scipy.signal import argrelextrema

from photutils.centroids import centroid_com, centroid_quadratic, centroid_2dg

import datetime



def where(list_conditions):
    index = list_conditions[0]
    for cond in list_conditions[1:]:
        index = np.multiply(index, cond)
    return np.where(index)

def mag_limit(std, Zp=22.5, omega=10, scale=0.33, n=3):
    return -2.5 * np.log10(n * std / (scale * omega)) + Zp

def check_print(message):
    print(colored('AstroPipe', 'green')+': '+ message)

def redshift_to_kpc(redshift,H0=70* u.km / u.s / u.Mpc, 
                             Tcmb0 = 2.725* u.K, Om0=0.3):
    cosmo = FlatLambdaCDM(H0=H0 , Tcmb0=Tcmb0 , Om0=Om0)
    return (cosmo.luminosity_distance(redshift) * 1000 * u.kpc/u.Mpc).value

def kpc_to_arcsec(kpc,distance):
    arcsec_to_rad = np.pi/(180*3600)
    return (kpc)/(arcsec_to_rad*distance)

def arcsec_to_kpc(arcsec,distance):
    arcsec_to_rad = np.pi/(180*3600)
    return arcsec*arcsec_to_rad*distance

def convert_PA(angle):
    if angle <0:
        return 180 + angle
    else:
        return angle

def binarize(image, nsigma=1, mask=None):
    """Binarize an image using a threshold of nsigma*std.
    Statistics are computed with sigma clipped, then the 
    image binarize is dilate to remove noise. 

    Parameters
    ----------
        image : array_like
            Image to binarize.
        nsigma : float, optional
            Number of sigma to use as threshold.
        mask : array_like, optional
            If use, mask is applied.
    
    Returns
    -------
        binary : array_like
            Binarized image.
    """
    mask = None if mask is None else mask
    if hasattr(image, 'mask'): 
        if mask is None: mask = np.ma.getmask(image)
        image = np.ma.getdata(image)

    mean, median, std = sigma_clipped_stats(image, sigma=2.5, mask=mask, maxiters=2)
    index = image > median+nsigma*std

    binarize = np.zeros_like(image)
    binarize[index * ~mask] = 1
    binarize = cv2.erode(binarize, np.ones((5,5)), iterations=1)
  
    return binarize


def morphologhy(binary):
    moments = cv2.moments(binary)
    
    x2 =(moments['m20']/moments['m00'])-(moments['m10']/moments['m00'])**2
    y2 =(moments['m02']/moments['m00'])-(moments['m01']/moments['m00'])**2
    xy = (moments['m11']/moments['m00'])-(
                moments['m10']*moments['m01']/moments['m00']**2)
    
    major = np.sqrt(0.5*(x2+y2) + np.sqrt((0.5*(x2-y2))**2 + xy**2))
    minor = np.sqrt(0.5*(x2+y2) - np.sqrt((0.5*(x2-y2))**2 + xy**2))
    angle = 0.5*np.arctan2(2*xy,x2-y2)*180/np.pi
    eps = 1 - minor/major

    return angle,major,eps

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)
            

def run_fabada(IMG,std=None, verbose=True,max_iter=3000):
    if not std: std=IMG.std
    recover = deepcopy(IMG)
    if hasattr(IMG.data, 'mask'):
        recover.data = fabada(recover.data.data,std**2,max_iter=max_iter,verbose=verbose)
        recover.data = np.ma.masked_array(recover.data,mask=IMG.data.mask)
    else:
        recover.data = fabada(recover.data,std**2,max_iter=max_iter,verbose=verbose)
    
    recover.name = recover.name+'_fabada'
    return recover
            

def make_parser():
    """Create an argument parser taken from MTObjects."""
    parser = argparse.ArgumentParser(description="Index of csv file")

    parser.add_argument("-index", type=int, help="Index", default=0)

    return parser

def closest(data,value):
    return np.nanargmin(np.abs(data-value))


def cutout(file, center, width, hdu=0, mode='image', out=None):
    '''
    Crop a fits file to a given width and center preserving the WCS. 
    
    Parameters
    ----------
        file: str
            Fits file name.
        center: tuple, optional
            Center of the cropped image in pixels (image) or in degrees (wcs).

        width: tuple
            Width of the cropped image in pixels (image) or in degrees (wcs).
        hdu: int, optional
            HDU to crop. Default is 0.
        mode: str, optional
            'image' or 'wcs'.
        out: str, optional
            Output file name. Default is filename_crop.fits.

    Returns
    -------
        out: str
            Cropped fits file name.
    '''
    
    data = fits.getdata(file,hdu)
    header = fits.getheader(file,hdu)

    if mode == 'wcs':
        sky = SkyCoord(center[0], center[1], unit='deg')
        wcs = WCS(header)
        center = np.int64(wcs.world_to_pixel(sky))
        width = np.int64(width / utils.proj_plane_pixel_scales(wcs))
    elif mode == 'image':
        center = np.int64(center)
        width = np.int64(width)
    else:
        raise ValueError('mode must be image or wcs')

    cropped, header = crop(data, header, center, width)

    if out is None: out = file.replace('.fits','_crop.fits')
    
    fits.PrimaryHDU(cropped, header).writeto(out,overwrite=True)

    return os.path.isfile(out)


def crop(data, header, center, width, out=None):
    '''
    Funtion that crops a image given a center and a width updating the WCS
    information in the header.

    Parameters
    ----------
        data: array_like
            Image to crop.
        header: astropy.io.fits.header.Header
            Header of the image.
        center: tuple
            Center of the cropped image in pixels.
        width: tuple
            Width of the cropped image in pixels.
    
    Returns
    -------
        new_data: array_like
            Cropped image.
        new_header: astropy.io.fits.header.Header
            Header of the cropped image.    
    '''
    wcs = WCS(header)
    center = np.int32(center)
    width = np.int32(width)

    new_data = data[center[1]-(width[1]//2 + 1):center[1]+width[1]//2,
                    center[0]-(width[0]//2 + 1):center[0]+width[0]//2]

    new_wcs = wcs[center[1]-(width[1]//2 + 1):center[1]+width[1]//2,
                    center[0]-(width[0]//2 + 1):center[0]+width[0]//2]
    
    header.update(new_wcs.to_header())
    header['COMMENT'] = "= Cropped fits file ({}).".format(datetime.date.today())
    header['ICF1PIX'] = (f'{center[1]-(width[1]//2 + width[1]%2)}:{center[1]+width[1]//2},{center[0]-(width[0]//2 + width[0]%2)}:{center[0]+width[0]//2}',
                          'Range of pixels used for this cutout')
    
    if out is not None: fits.PrimaryHDU(new_data, header).writeto(out, overwrite=True)
    
    return new_data, header 

def flashprint(string):
    print(string, end='\r')
    sys.stdout.flush()



def find_center(data, center, width=30):
    '''
    Find the center of an object in a cropped image.
    It uses centroid_quadratic to find the center of the object.
    It fist a quadractic function to the data and then
    finds the maximum of the function.

    Parameters
    ----------
        data: array
            Image data.
        center: tuple
            Center of the object in the image.
        width: int, optional
            Width of the cropped image.
    
    Returns
    -------
        x,y: tuple
            Center of the object in the image.
    '''    

    crop = data[np.int32(center[1]-width):np.int32(center[1]+width),
                    np.int32(center[0]-width):np.int32(center[0]+width)]

    x,y = centroid_quadratic(crop)

    x += np.int32(center[0]-width)
    y += np.int32(center[1]-width)
    return x,y



def limits(x,y,n=30):
    index = ~np.isnan(y)    
    p = np.poly1d(np.polyfit(x[index],y[index],n))
    mins = argrelextrema(p(x), np.greater)
    return np.nanmin(mins)

def adaptive_histogram(data, bandwidth=None, weights=None):
    """
    Compute histogram based on a top-hat kernel with the specified (adaptive) bandwidth.
    :param data: Collection of data points
    :param bandwidth: width of the top-hat kernel
    """
    if bandwidth is None:
        h1 = 1 + int(np.sqrt(data.size))
        h2 = 1 + int(data.size / 2)
        h = 1 + int(np.sqrt(h1 * h2))
    else:
        h = bandwidth
    if weights is None:
        weights = np.ones_like(data)

    sorted_data = np.argsort(data.flatten())
    x = data.flatten()[sorted_data]
    w = weights.flatten()[sorted_data]
    cumul_mass = np.cumsum(w)

    x_mid = (x[h:] + x[:-h]) / 2
    # x_mean = (x_cumul[h:] - x_cumul[:-h]) / h
    half_h = int(h/2)
    x_median = x[half_h:half_h-h]
    density = (cumul_mass[h:] - cumul_mass[:-h]) / (x[h:] - x[:-h])
    x_bin = np.sqrt(x_mid*x_median)
    return x_bin, density / data.size

def change_coordinates(positions, center, pa):
    rotMatrix = np.array([[np.cos(pa),-np.sin(pa)],
                        [np.sin(pa),np.cos(pa)]])
    newCoordinates = np.matmul(np.transpose(positions-center[:,np.newaxis]),rotMatrix)
    return np.transpose(newCoordinates)

from scipy import ndimage
from matplotlib.colors import LogNorm

# Funtion that given an numerical distribution finds the FWHM of the distribution
def getFWHM(x,y, oversamp=100, height=False):
    '''
    Given a numerical distribution finds the FWHM of the distribution
    interpolation is done to overcome undersampling offsets
    
    Parameters
    ----------
        x : numpy.ndarray
            x-axis of the distribution
        y : numpy.ndarray
            y-axis of the distribution
        oversamp : int
            oversampling factor
        height : bool
            if True returns the height of the distribution at the FWHM

    Returns
    -------
        fwhm : float
            FWHM of the distribution
    '''
    xx = np.linspace(x[0],x[-1],len(x)*oversamp)
    yy = np.interp(xx,x,y)

    max_y = np.max(yy)
    half_max = max_y/2
    idx = np.where(yy >= half_max)[0]
    fwhm = xx[idx[-1]] - xx[idx[0]]
    if np.abs(yy[idx[-1]] - yy[idx[0]]) > max_y/2*len(idx):
        raise Warning('FWHM is not well defined, check your distribution')
    fwhm_y = (yy[idx[-1]] + yy[idx[0]])/2
    
    if not height: return fwhm
    if height: return fwhm, fwhm_y


def find_mode(x, weights=None):
    """
    Locate the mode of a distribution by fitting a polynomial within +-one-sigma interval.
    :param x: Collection of data points
    """
    # Legendre polynomials
    def L0(x): return np.ones_like(x)
    def L1(x): return x
    def L2(x): return (3 * x ** 2 - 1) / 2
    # def L3(x): return (5*x**3 - 3*x) / 2
    # def L4(x): return (35*x**4 - 30*x**2 + 3) / 8
    def norm_L(n): return 2 / (2 * n + 1)

    if weights is None:
        weights = np.ones_like(x)

    total_mass = np.nansum(weights)
    x0 = np.nansum(x*weights) / total_mass
    sigma = np.sqrt(np.nansum((x-x0)**2 * weights)/total_mass)
    delta_peak = 0
    i = 0
    while True:
        i += 1
        x0 += delta_peak
        delta = x - x0
        w = weights[np.abs(delta) < sigma]
        delta = delta[np.abs(delta) < sigma] / sigma
        # scalar product:
        c0 = np.mean(L0(delta)) / norm_L(0) # point density · L0
        c1 = np.mean(L1(delta)) / norm_L(1)  # point density · L1
        c2 = np.mean(L2(delta)) / norm_L(2)  # point density · L2
        # c3 = np.mean(L3(delta)) / norm_L(3)  # point density · L3
        # c4 = np.mean(L4(delta)) / norm_L(4)  # point density · L4
        total_mass = np.nansum(w)
        c0 = np.nansum(w*L0(delta))/total_mass / norm_L(0) # point density · L0
        c1 = np.nansum(w*L1(delta))/total_mass / norm_L(1) # point density · L1
        c2 = np.nansum(w*L2(delta))/total_mass / norm_L(2) # point density · L2
        polynomial_fit = c0*L0(delta) + c1*L1(delta) + c2*L2(delta)  # + c3*L3(delta) + c4*L4(delta)
        delta_peak = delta[np.argmax(polynomial_fit)] * sigma/np.sqrt(i)  # divide by sqrt(i) to prevent oscillations
        # print(x0, delta_peak)
        if np.abs(delta_peak) <= sigma/x.size:
            return x0, x0+delta*sigma, polynomial_fit*delta.size/x.size/sigma


# function that converts aparent magnitude to absolute magnitude given redshift 
def abs_mag(z, apparent_mag):
    distance = redshift_to_kpc(z)
    return apparent_mag - 5*np.log10(distance) + 5

def find(inDIR='',filetype='*.fits'):

    fileList = []
    
    # Walk through directory
    for dName, sdName, fList in os.walk(inDIR):
        for fileName in fList:
            if fnmatch.fnmatch(fileName, filetype): # Match search string
                fileList.append(os.path.join(dName, fileName))

    return fileList

# Computes the power spectrum of an image returns the power spectrum and the corresponding frequencies
# def power_spectrum(image):
#     # Compute the 2D power spectrum
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))
#     # Compute the 1D power spectrum
#     psd2D = np.abs(fshift)**2
#     psd1D = np.mean(psd2D, axis=0)
#     # Compute the corresponding frequencies
#     Lx = image.shape[0]
#     Ly = image.shape[1]
#     fx = np.fft.fftfreq(Lx, d=1)
#     fy = np.fft.fftfreq(Ly, d=1)
#     fX, fY = np.meshgrid(fx, fy)
#     f = np.sqrt(fX**2 + fY**2)
#     ind = np.argsort(f, axis=None)
#     f_sorted = f.flatten()[ind]
#     psd1D_sorted = psd1D.flatten()[ind]
#     return psd1D_sorted, f_sorted


def power_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f) 
    power = np.abs(fshift)
    k_y = np.fft.fftshift(np.fft.fftfreq(fshift.shape[0], d=2))
    k_x = np.fft.fftshift(np.fft.fftfreq(fshift.shape[1], d=2))
    k2 = (k_y**2)[:,np.newaxis] + (k_x**2)[np.newaxis,:]
    return power.flatten(), k2.flatten()


from scipy import ndimage
