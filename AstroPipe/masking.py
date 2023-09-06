import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from regions import CirclePixelRegion
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.wcs import WCS, utils
import astropy.units as u
from astropy.coordinates import Angle
from regions.core import PixCoord
import argparse
import sys
import os
from scipy.ndimage import gaussian_filter
from photutils.segmentation import detect_sources
from fabada import fabada

from .classes import SExtractor, AstroGNU, MTObjects
from .utilities import where, change_coordinates

from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter

from matplotlib import pyplot as plt 
from scipy.signal import convolve2d

point_sexcofing = {
                "CHECKIMAGE_TYPE": "SEGMENTATION",
                "DEBLEND_MINCONT": 0.005,
                "DEBLEND_NTHRESH": 32,
                "BACK_SIZE": 120,       # 20 
                'DETECT_THRESH':0.9,
                "SATUR_LEVEL": 50000.0,
                "MAG_ZEROPOINT": 22.5,
                'PHOT_FLUXFRAC': 0.5,
                'MEMORY_OBJSTACK': 10000,           # number of objects in stack
                'MEMORY_PIXSTACK': 1000000,         # number of pixels in stack
                'MEMORY_BUFSIZE': 51200
            }

def sigma_filter(catalog, columns, sigma=5, weights=None, colid = 'NUMBER'):
    if weights is None: weights = np.ones(len(catalog))
    index = np.ones(len(catalog)).astype(bool)
    for col in columns:
        mean, median, std = sigma_clipped_stats(catalog[col], sigma=2.5)
        index &= catalog[col]*weights > mean + sigma*std
    return [catalog[colid][ind] for ind in np.where(index)[0]]

def gaussian1D(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian(sigma, shape, normalize=True):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
    d = np.sqrt(x*x+y*y)
    sigma = sigma
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    if normalize: g /= np.sum(g)
    return g


def gaussian2d(positions, center, sigma):
    return np.exp(-((positions[0] - center[0])**2. / (2. * sigma[0]**2.) + (positions[1] - center[1] )**2. / (2. * sigma[1]**2.)))


def filter_distance2D(positions, center, pa, eps, distance):
    newCoordinates = change_coordinates(positions, center, pa)
    index = (newCoordinates[0] < distance) + (
             newCoordinates[1] < (1-eps)*distance)
    return index

def weight_distance2D(positions, center, pa, eps, sigma):
    newCoordinates = change_coordinates(positions, center, pa)
    return gaussian2d(newCoordinates, np.array([0,0]) , np.array([sigma,(1-eps)*sigma]))


def get_peaks(data, mask=None, verbose=False):
    original_stats = sigma_clipped_stats(data, mask=mask, sigma=2.5)
    recover = fabada(data, 2*(original_stats[2]**2),verbose=verbose)
    peaks = recover - gaussian_filter(data,sigma=5)
    mean, median, std = sigma_clipped_stats(peaks, mask=mask, sigma=2.5)
    peaks = fabada(peaks,3*(std**2),verbose=verbose)
    if mask is not None: peaks[np.where(mask != 0)] = np.nan
    peaks[peaks==0] = np.nan
    return peaks

def sharp_mask(data, C=7, W=5, mask=False, enhace=True):
    '''
    Contrast Enhancement using Digital Unsharp Masking
    https://www.astronomy.ohio-state.edu/pogge.1/Ast350/Unsharp/
    '''
    original_stats = sigma_clipped_stats(data, mask=mask, sigma=2.5)
    if enhace: enhaced = fabada(data, 2*(original_stats[2]**2),verbose=True)
    else: enhaced = data
    smooth = gaussian_filter(data,sigma=W)
    return C*enhaced - (C-1)*smooth

def get_bipeaks(data,n=2.5,sigma=5,verbose=False):

    original_stats = sigma_clipped_stats(data, sigma=2.5)
    recover = fabada(data, 2*(original_stats[2]**2),verbose=verbose)
    peaks = recover - gaussian_filter(data, sigma=sigma)
    residual_stats = sigma_clipped_stats(peaks, sigma=2.5)
    peaks = fabada(peaks,3*(residual_stats[2]**2),verbose=verbose)
    
    peaks[peaks > n*residual_stats[2]] *= 255
    peaks[peaks < n*residual_stats[2]] = 0
    return peaks

def increase_mask(mask, shape=(3,3)):
    import cv2
    if mask.dtype != np.int16: mask = mask.astype(np.int16)
    return cv2.dilate(mask,np.ones(shape)/np.prod(shape), iterations=1)


def sexmask(IMG, folders, plot=False):
    '''
    Use SExtractor to create a mask of the image
    We run FABADA to enhace the object and improve the detection.
    It has three steps:
        1. Run SExtractor with default parameters to detect objects
        2. Run SExtractor with default parameters to detect point sources
        3. Run SExtractor in the residual image to detec point sources inside the galaxy
    The last two steps has a gaussian weight distance and area filter to avoid the masking 
    of extended inner regions of the galaxy.
    TODO: in (3) add filter of detected objects and only apply to nxReff of galaxy


    Parameters
    ----------
        IMG : AstroPipe Image object
            Class object containing the image data

        folders : AstroPipe Directories object
            Class object containing the directories of the project

        plot : bool, optional
            If True, plot the mask and the image and save it in temporary directory

    Returns
    -------
        bool : True if the mask is created
    '''

    if not hasattr(folders,'mask'): folders.set_mask(os.path.join(folders.out,os.path.basename(IMG.name).split(IMG.extension)[0]+'_mask.fits'))
    if not hasattr(IMG,'bkg'): bkg = IMG.bkg
    else: bkg = 0


    if not hasattr(IMG,'std'): 
        mean, bkg, IMG.std = sigma_clipped_stats(IMG.data, sigma=2.5)

    # Step 1: Run FABADA to smooth image an improve detection and initialize mask

    recover = fabada(IMG.data,2*(IMG.std**2),verbose=True)
    recover[np.isnan(IMG.data)] = np.nan
    IMG.set_mask(np.zeros_like(IMG.data))

    # Step 2: Run SExtractor
    #   Step 2.1: Run SExtractor with default param with 1.3 sigma threshold

    params = ['ISOAREA_IMAGE','ELLIPTICITY','FWHM_IMAGE', 'THETA_IMAGE']

    defaultsex_config  = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                                    'CHECKIMAGE_NAME' : os.path.join(folders.temp,
                                                        IMG.name+'_sex.fits'),
                                    'PIXEL_SCALE' : IMG.pixel_scale,
                                    'DETECT_THRESH': 1.0,   # 1.5
                                    "DEBLEND_MINCONT": 0.005,
                                    "DEBLEND_NTHRESH": 32,
                                    'PHOT_FLUXFRAC': 0.9, 
                                    'BACK_SIZE':  120,       # 64
                                    'MEMORY_OBJSTACK': 10000,           
                                    'MEMORY_PIXSTACK': 1000000,     
                                    'MEMORY_BUFSIZE': 51200}

    defaultsex = SExtractor(config= defaultsex_config, params=params)
    defaultsex.run(recover)
    IMG.obj_id(defaultsex)

    defaultsex.objects[np.where(defaultsex.objects == IMG.id)] = 0
    
    if not hasattr(IMG,'r_eff'): IMG.r_eff = float(defaultsex.catalog[IMG.id-1]['FLUX_RADIUS'])
    if not hasattr(IMG,'eps'):   IMG.eps = float(defaultsex.catalog[IMG.id-1]['ELLIPTICITY'])
    if not hasattr(IMG,'pa'):    IMG.pa = float(defaultsex.catalog[IMG.id-1]['THETA_IMAGE'])

    index = filter_distance2D(np.array([defaultsex.catalog['X_IMAGE'],
                                    defaultsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff)
    for ind in sigma_filter(defaultsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE']):
        if defaultsex.catalog['CLASS_STAR'][ind-1] < 0.5:
            defaultsex.objects[np.where(defaultsex.objects == ind)] = 0
    
    defaultsex.objects = increase_mask(defaultsex.objects,shape=(4,4))
    IMG.data.mask[np.where(defaultsex.objects != 0)] = 1

    #   Step 2.2: Run SExtractor for point sources

    midsex_config  = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                                    'CHECKIMAGE_NAME' : os.path.join(folders.temp,
                                    IMG.name+'_sex.fits'),
                                    'PIXEL_SCALE' : IMG.pixel_scale,
                                    "DEBLEND_MINCONT": 0.005,
                                    "DEBLEND_NTHRESH": 32,  
                                    'DETECT_THRESH': 1.2,
                                    'BACK_SIZE': 20,
                                    'PHOT_FLUXFRAC': 0.5,
                                    'MEMORY_OBJSTACK': 10000,           
                                    'MEMORY_PIXSTACK': 1000000,     
                                    'MEMORY_BUFSIZE': 51200}
    
    midsex = SExtractor(config= midsex_config, params=params)
    midsex.run(recover)
    IMG.obj_id(midsex)
    midsex.objects[np.where(midsex.objects == IMG.id)] = 0

    weights = weight_distance2D(np.array([midsex.catalog['X_IMAGE'],
                                          midsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([midsex.catalog['X_IMAGE'],
                                        midsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(midsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE'],
                        sigma=3, weights=weights[index]):
        if midsex.catalog['CLASS_STAR'][ind-1] < 0.5:
            midsex.objects[np.where(midsex.objects == ind)] = 0

    midsex.objects = increase_mask(midsex.objects,shape=(3,3))

    IMG.data.mask[np.where(midsex.objects != 0)] = 1

    #   Step 2.3: Run SExtractor residuals from gaussian smoothing

    pointsex_config = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                        'CHECKIMAGE_NAME' : os.path.join(folders.temp,
                                        IMG.name+'_sex.fits'),
                        'PIXEL_SCALE' : IMG.pixel_scale,
                        "DEBLEND_MINCONT": 0.005,
                        "DEBLEND_NTHRESH": 32,
                        "BACK_SIZE": 64,
                        'DETECT_THRESH': 0.9,
                        'PHOT_FLUXFRAC': 0.5,
                        'MEMORY_OBJSTACK': 10000,           # number of objects in stack
                        'MEMORY_PIXSTACK': 1000000,         # number of pixels in stack
                        'MEMORY_BUFSIZE': 51200}

    pointsex = SExtractor(config = pointsex_config, params=params)
    pointsex.run(get_bipeaks(IMG.data.data,n=2.5,sigma=5))
    IMG.obj_id(pointsex)
    pointsex.objects[np.where(pointsex.objects == IMG.id)] = 0

    weights = 1.5*weight_distance2D(np.array([pointsex.catalog['X_IMAGE'],
                                              pointsex.catalog['Y_IMAGE']]),
                            IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([pointsex.catalog['X_IMAGE'],
                                        pointsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(pointsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE'], 
                            sigma=3, weights=weights[index]):
        pointsex.objects[np.where(pointsex.objects == ind)] = 0

    pointsex.objects = increase_mask(pointsex.objects, shape = (2,2))

    IMG.data.mask[np.where(pointsex.objects != 0)] = 1

    #   Step 3: Save the files 
    
    masked = IMG.data.data.copy()
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header
                    ).writeto(os.path.join(folders.temp,f'{IMG.name}_masked.fits')
                    ,overwrite=True)

    mask_array = np.array(IMG.data.mask, dtype = np.uint8)
    fits.PrimaryHDU(mask_array, header=IMG.header).writeto(folders.mask,overwrite=True)

    if plot: 
        IMG.show(width=300)
        plt.savefig(os.path.join(folders.out,IMG.name+'_mask.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    return os.path.isfile(folders.mask)

def mtomask(IMG, folders, plot=False):
    '''
    Use MTObjects to create a mask of the image
    We run FABADA to enhace the object and improve the detection.
    It has three steps:
        1. Run MTObjects with default parameters to detect objects
        2. Run MTObjects with default parameters to detect point sources
        3. Run MTObjects in the residual image to detec point sources inside the galaxy
    The last two steps has a gaussian weight distance and area filter to avoid the masking 
    of extended inner regions of the galaxy.
    TODO: in (3) add filter of detected objects and only apply to nxReff of galaxy


    Parameters
    ----------
        IMG : AstroPipe Image object
            Class object containing the image data

        folders : AstroPipe Directories object
            Class object containing the directories of the project

        plot : bool, optional
            If True, plot the mask and the image and save it in temporary directory

    Returns
    -------
        bool : True if the mask is created
    '''

    if not hasattr(folders,'mask'): folders.set_mask(os.path.join(folders.out,os.path.basename(IMG.file).split(IMG.extension)[0]+'_mask.fits'))
    if hasattr(IMG,'bkg'): bkg = IMG.bkg
    else: bkg = 0

    if not hasattr(IMG,'std'): 
        mean, bkg, IMG.std = sigma_clipped_stats(IMG.data, sigma=2.5)

    
    # Step 1: Run FABADA to smooth image an improve detection and initialize mask

    recover = fabada(IMG.data,2*(IMG.std**2),verbose=True)
    recover[np.isnan(IMG.data)] = np.nan
    IMG.set_mask(np.zeros_like(IMG.data))

    # Step 2: Run SExtractor
    #   Step 2.1: Run SExtractor with default param with 1.3 sigma threshold

    params = ['ISOAREA_IMAGE','ELLIPTICITY','FWHM_IMAGE', 'THETA_IMAGE']

    defaultsex_config  = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                                    'CHECKIMAGE_NAME' : os.path.join(folders.temp,
                                                        IMG.name+'_sex.fits'),
                                    'PIXEL_SCALE' : IMG.pixel_scale,
                                    'DETECT_THRESH': 1.0,   # 1.5
                                    "DEBLEND_MINCONT": 0.005,
                                    "DEBLEND_NTHRESH": 32,
                                    'PHOT_FLUXFRAC': 0.9, 
                                    'BACK_SIZE':  120,       # 64
                                    'MEMORY_OBJSTACK': 10000,           
                                    'MEMORY_PIXSTACK': 1000000,     
                                    'MEMORY_BUFSIZE': 51200}

    defaultsex = SExtractor(config= defaultsex_config, params=params)
    defaultsex.run(recover)
    IMG.obj_id(defaultsex)

    defaultsex.objects[np.where(defaultsex.objects == IMG.id)] = 0
    
    if not hasattr(IMG,'r_eff'): IMG.r_eff = float(defaultsex.catalog[IMG.id-1]['FLUX_RADIUS'])
    if not hasattr(IMG,'eps'):   IMG.eps = float(defaultsex.catalog[IMG.id-1]['ELLIPTICITY'])
    if not hasattr(IMG,'pa'):    IMG.pa = float(defaultsex.catalog[IMG.id-1]['THETA_IMAGE'])

    index = filter_distance2D(np.array([defaultsex.catalog['X_IMAGE'],
                                    defaultsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff)
    for ind in sigma_filter(defaultsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE']):
        if defaultsex.catalog['CLASS_STAR'][ind-1] < 0.5:
            defaultsex.objects[np.where(defaultsex.objects == ind)] = 0
    
    defaultsex.objects = increase_mask(defaultsex.objects,shape=(4,4))
    IMG.data.mask[np.where(defaultsex.objects != 0)] = 1

    #   Step 2.2: Run MTObjects 

    mto = MTObjects()
    mto.move_factor = 0.01 # To imporve detection of faint outskirts of objects
    
    mto.run(recover)
    IMG.obj_id(mto)
    mto.objects[np.where(mto.objects == IMG.id)] = 0

    weights = weight_distance2D(np.array([mto.catalog['X'],
                                          mto.catalog['Y']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([mto.catalog['X'],
                                        mto.catalog['Y']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(mto.catalog[index], ['area','R_fwhm'],
                        sigma=3, weights=weights[index], colid='ID'):
        mto.objects[np.where(mto.objects == ind)] = 0

    mto.objects = increase_mask(mto.objects,shape=(3,3))

    IMG.data.mask[np.where(mto.objects != 0)] = 1

    #   Step 2.3: Run MTObjects in residuals from gaussian smoothing

    mto.run(get_bipeaks(IMG.data.data,n=2.5,sigma=5))
    IMG.obj_id(mto)
    mto.objects[np.where(mto.objects == IMG.id)] = 0

    weights = 1.5*weight_distance2D(np.array([mto.catalog['X'],
                                              mto.catalog['Y']]),
                            IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([mto.catalog['X'],
                                        mto.catalog['Y']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(mto.catalog[index], ['area','R_fwhm'], 
                            sigma=3, weights=weights[index],colid='ID'):
        mto.objects[np.where(mto.objects == ind)] = 0

    mto.objects = increase_mask(mto.objects, shape = (2,2))

    IMG.data.mask[np.where(mto.objects != 0)] = 1

    #   Step 3: Save the files 
    
    masked = IMG.data.data.copy()
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header
                    ).writeto(os.path.join(folders.temp,f'{IMG.name}_masked.fits')
                    ,overwrite=True)

    mask_array = np.array(IMG.data.mask, dtype = np.uint8)
    fits.PrimaryHDU(mask_array, header=IMG.header).writeto(folders.mask,overwrite=True)

    if plot: 
        IMG.show(width=300)
        plt.savefig(os.path.join(folders.out,IMG.name+'_mask.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    return os.path.isfile(folders.mask)

def automatic_mask(IMG,folders,sex_config=point_sexcofing,nc_config='--numthreads=8',plot=True):


    if not hasattr(folders,'mask'): folders.set_mask(os.path.join(folders.out,os.path.basename(IMG.file).split(IMG.extension)[0]+'_mask.fits'))
    if hasattr(IMG,'bkg'): bkg = IMG.bkg
    else: bkg = 0
    gnu = AstroGNU(IMG.file, hdu=IMG.hdu, dir=folders.temp)
    gnu.noisechisel(config=nc_config)
    gnu.segment(config=nc_config)
    gnu.make_catalog(config=nc_config)

    IMG.noise(gnu.objects,
            plot=os.path.join(folders.temp,IMG.name+"_noise.png"))

    if not "CHECKIMAGE_NAME" in sex_config:
         sex_config["CHECKIMAGE_NAME"] =  os.path.join(folders.temp,
                                    IMG.name+'_sex.fits')
    
    if not 'PIXEL_SCALE' in sex_config:
        sex_config['PIXEL_SCALE'] = IMG.pixel_scale     

    recover = fabada(IMG.data,3*IMG.std,verbose=True)
    ast_obj = detect_sources(recover - bkg, 5*IMG.std, npixels=5)
    
    sex = SExtractor(config = sex_config)
    
    sex.run(recover)

    # Masking Process
    # Step 1: Initialize Image and Masks

    IMG.obj_id(gnu)
    IMG.obj_id(sex)
    ast_id = ast_obj.data[np.int64(IMG.pix[1]),np.int64(IMG.pix[0])]

    # Step 2: Create final Mask (True(1) = masked ; False(0) = Not-masked)

    IMG.set_mask(np.zeros_like(IMG.data))

    #   Step 2.1: Mask all objects of SExtractor outside of Segment source

    IMG.data.mask[where([sex.objects != IMG.id,
               gnu.objects != IMG.id,
               sex.objects != 0])] = 1

    #   Step 2.2: Mask all objects of SExtractor outside of Astropy source

    IMG.data.mask[where([ast_obj.data != ast_id,
                         sex.objects != 0])] = 1

    
    #   Step 2.3: Mask all objects of SExtractor that overlaps one pixel 
    #             with Astropy source

    overlap = np.unique(sex.objects[where([ast_obj.data == ast_id, sex.objects != 0])])
    for id in overlap:
        IMG.data.mask[np.where(sex.objects==id)]=0

    IMG.set_catalog(gnu.catalog)


    # Mask faint extended regions 

    temporary_masked_file = os.path.join(folders.temp,f'{IMG.name}_mask_point.fits')

    masked = IMG.data.data.copy()
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header).writeto(temporary_masked_file,overwrite=True)


    gnu_extended = AstroGNU(temporary_masked_file, hdu=0, dir=folders.temp)

    gnu_extended.noisechisel(config=nc_config+' --tilesize=7,7')
    gnu_extended.segment(config='--gthresh=1 --snminarea=30 ')

    id_extended = gnu_extended.objects[np.int(IMG.pix[1]),np.int(IMG.pix[0])]

    IMG.data.mask[where([gnu_extended.objects != id_extended,
                         gnu_extended.objects != 0])] = 1
    

    # Mask again with SExtractor for point sources using the applied mask so far

    masked[np.where(IMG.data.mask==1)] = np.nan

    sex.run(masked)
    IMG.obj_id(sex)
    sex.objects[sex.objects == IMG.id] = 0
    # sex.objects = cv2.dilate(sex.objects.astype(np.int16), gaussian(3,(10, 10)), iterations=1)
    sex.objects = convolve2d(sex.objects, gaussian(3,(10, 10)), mode='same', boundary='fill', fillvalue=0)
    IMG.data.mask[sex.objects != 0] = 1



    # Mask interior regions of extended sources

    peaks = binarize_peaks(IMG.data.data,IMG.data.mask,theshold=8, sigma=5)
    sex.run(peaks)
    mean, median, std = sigma_clipped_stats(sex.catalog['FLUX_RADIUS'])
    bigregions = sex.catalog['NUMBER'][sex.catalog['FLUX_RADIUS'] > median+5*std]
    IMG.obj_id(sex)
    sex.objects[sex.objects==IMG.sex_id] = 0
    for biggie in bigregions:
        sex.objects[sex.objects == biggie] = 0
    sex.objects = convolve2d(sex.objects, gaussian(2,(6, 6)), mode='same', boundary='fill', fillvalue=0)
    IMG.data.mask[sex.objects != 0] = 1


    # Save the files 

    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header).writeto(temporary_masked_file,overwrite=True)

    mask_array = np.array(1-IMG.data.mask,dtype = np.uint8)
    fits.PrimaryHDU(mask_array,header=IMG.header).writeto(folders.mask,overwrite=True)

    if plot: 
        IMG.show()
        plt.savefig(os.path.join(folders.temp,IMG.name+'_mask.jpg'),dpi=300)


def ds9_region_masking(IMG,folders):

    file = os.path.join(folders.regions,IMG.name+'.reg')

    df_region = pd.read_csv(file,delimiter=' ',header=None,
                        names=['frame', 'region','ra','dec','radius'])
    
    
    if ':' in df_region['ra'][0]:
        units = (u.hourangle,u.deg)
    else:
        units = (u.deg,u.deg)
    
    for index, row in df_region.iterrows():

        pixel_center = skycoord_to_pixel(SkyCoord(row['ra'],row['dec'], unit=units,frame="fk5"), IMG.wcs)
        pixel_center = PixCoord(pixel_center[0],pixel_center[1])
        pixel_radius = np.float64(row['radius'][:-1])/IMG.pixel_scale
        mask_region = CirclePixelRegion(pixel_center,pixel_radius).to_mask(mode='exact')
        IMG.data.mask[np.where(mask_region.to_image(IMG.data.mask.shape)==1)] = True
    
    mask_array = np.array(1-IMG.data.mask,dtype = np.uint8)
    fits.PrimaryHDU(mask_array,header=IMG.header).writeto(folders.mask,overwrite=True)

    return True

