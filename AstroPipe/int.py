''' 
Modules for process Isaac Newton Telescope images. 
'''

import AstroPipe.calibration as cal
from AstroPipe.classes import AstroGNU

import numpy as np
from astropy.io import fits
import os 


# Read Log and create directories

def int_stack(image_list,out=None):
    if out is None:
        out = os.path.dirname(image_list[0])
        out = os.path.join(out,'stacked.fits')

    h1 = cal.stack(image_list,hdu=1)
    h2 = cal.stack(image_list,hdu=2)
    h3 = cal.stack(image_list,hdu=3)
    h4 = cal.stack(image_list,hdu=4)
    
    
    hdul = [None, h1, h2, h3, h4]
    header = [fits.getheader(image_list[0],i) for i in range(4)]

    int_save(hdul,header,out)

    return hdul

def int_flatstack(image_list, masterbias=[0,0,0,0,0], out=None):
    if out is None:
        out = os.path.dirname(image_list[0])
        out = os.path.join(out,'masterflat.fits')

    h1 = cal.flatstack(image_list,masterbias = masterbias[1], hdu=1)
    h2 = cal.flatstack(image_list,masterbias = masterbias[2],hdu=2)
    h3 = cal.flatstack(image_list,masterbias = masterbias[3],hdu=3)
    h4 = cal.flatstack(image_list,masterbias = masterbias[4],hdu=4)
    
    
    hdul = [None, h1, h2, h3, h4]
    header = [fits.getheader(image_list[0],i) for i in range(4)]
    

    int_save(hdul,header,out)

    return hdul


def int_save(hdul,header,out):
    new_hdul = [fits.PrimaryHDU(hdul[0],header=header[0])]
    for hdu in hdul[1:]:
        new_hdul.append(fits.ImageHDU(hdu,header=header[1]))

    new_hdul = fits.HDUList(new_hdul)
    new_hdul.writeto(out, overwrite=True)
    

    



# Create autoflat for WFC





# Calibrate images



# Create coadds



