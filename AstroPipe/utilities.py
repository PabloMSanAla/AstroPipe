import numpy as np
from termcolor import colored
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import cv2
from copy import deepcopy
from fabada import fabada
import argparse
from math import gamma

from astropy.wcs import WCS
from astropy.io import fits


from photutils.centroids import centroid_com, centroid_quadratic, centroid_2dg




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

def convert_PA(angle):
    if angle <0:
        return 180 + angle
    else:
        return angle

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

    return angle,major/2,eps
            

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


def crop(IMG, width, out=None):

    IMG.crop(IMG.pix, width=width)
    IMG.set_data(IMG.cropped)
    IMG.pix = np.array(width)

    IMG.header['CRVAL1'] = IMG.ra
    IMG.header['CRVAL2'] = IMG.dec

    IMG.header['CRPIX1'] = IMG.pix[1]
    IMG.header['CRPIX2'] = IMG.pix[0]

    IMG.header['NAXIS1'] = 2*width[0]
    IMG.header['NAXIS2'] = 2*width[1]


    IMG.wcs = WCS(IMG.header)
    IMG.hdu = 0

    if out:
        fits.PrimaryHDU(IMG.data,IMG.header).writeto(out,overwrite=True)
        IMG.file = out


def find_center(IMG,r_eff=20):
    
    crop = IMG.data[np.int32(IMG.pix[1]-r_eff):np.int32(IMG.pix[1]+r_eff),
                    np.int32(IMG.pix[0]-r_eff):np.int32(IMG.pix[0]+r_eff)]
    # x1,y1 = centroid_com(crop)
    x,y = centroid_quadratic(crop)
    # x3,y3 = centroid_2dg(crop)
   
    # I_array = np.array([crop[np.int32(y1),np.int32(x1)],crop[np.int32(y2),np.int32(x2)],crop[np.int32(y2),np.int32(x2)]])
    # x = np.sum(np.array([x1,x2,x3])*I_array/np.sum(I_array))
    # y = np.sum(np.array([y1,y2,y3])*I_array/np.sum(I_array))
    x += np.int32(IMG.pix[0]-r_eff)
    y += np.int32(IMG.pix[1]-r_eff)
    return x,y

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


