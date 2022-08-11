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

from .classes import SExtractor, AstroGNU
from .utilities import where


default_sex = {
                "CHECKIMAGE_TYPE": "SEGMENTATION",
                "DEBLEND_MINCONT": 0.005,
                "DEBLEND_NTHRESH": 32,
                "BACK_SIZE": 20,
                'DETECT_THRESH':0.9,
                "SATUR_LEVEL": 50000.0,
                "MAG_ZEROPOINT": 22.5,
                'PHOT_FLUXFRAC': 0.5,
                'MEMORY_OBJSTACK': 10000,           # number of objects in stack
                'MEMORY_PIXSTACK': 1000000,         # number of pixels in stack
                'MEMORY_BUFSIZE': 51200
            }




def automatic_mask(IMG,folders,sex_config=default_sex,nc_config='--numthreads=8'):




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
    ast_obj = detect_sources(recover, 5*IMG.std, npixels=5)
    
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

    IMG.data.mask[where([sex.objects != IMG.sex_id,
               gnu.objects != IMG.seg_id,
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

    masked = IMG.data.data
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header).writeto(temporary_masked_file,overwrite=True)


    gnu_extended = AstroGNU(temporary_masked_file, hdu=0, dir=folders.temp)

    gnu_extended.noisechisel()
    gnu_extended.segment(config='--gthresh=1 --snminarea=30')

    id_extended = gnu_extended.objects[np.int(IMG.pix[1]),np.int(IMG.pix[0])]

    IMG.data.mask[where([gnu_extended.objects != id_extended,
                         gnu_extended.objects != 0])] = 1
    
    mask_array = np.array(1-IMG.data.mask,dtype = np.uint8)
    fits.PrimaryHDU(mask_array,header=IMG.header).writeto(folders.mask,overwrite=True)


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



