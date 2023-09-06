import glob
from re import S 
from astropy.io import fits
import numpy as np
import os
from astropy.stats import SigmaClip
import astroalign

from .classes import AstroGNU

# Make an structure to find out if there are darks
# or bias to make False or True
class structure():

    def __init__(self,night, extension='.fits'):
        self.night = night
        self.directories = os.listdir(night)
        self.directories = [s.lower() for s in self.directories]
        self.extension = extension
        self.check_dark()
        self.check_bias()
        self.check_light()
        self.check_domeflat()
        self.check_skyflat()
        self.make_calibrated()

    def check_dark(self):
        bool_array = ['dark' in f for f in self.directories]
        if any(bool_array):
            self.dark = os.path.join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.darkList = glob.glob(os.path.join(self.dark,'*'+self.extension))
        else:
            self.dark = None
    
    def check_bias(self):
        bool_array = ['bias' in f for f in self.directories]
        if any(bool_array):
            self.bias = os.path.join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.biasList = glob.glob(os.path.join(self.bias,'*'+self.extension))
        else:
            self.bias = None
        
    def check_light(self):
        bool_array = [f.startswith(('li','sc')) for f in self.directories]
        if any(bool_array):
            self.light = os.path.join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.lightList = glob.glob(os.path.join(self.light,'*'+self.extension))
        else:
            self.light = None
    
    def check_domeflat(self):
        bool_array = ['dome' in f for f in self.directories]
        if any(bool_array):
            self.domeflat = os.path.join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.domeflatList = glob.glob(os.path.join(self.domeflat,'*'+self.extension))
        else:
            self.domeflats = None

    def check_skyflat(self):
        bool_array = ['sky' in f for f in self.directories]
        if any(bool_array):
            self.skyflat = os.path.join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.skyflatList = glob.glob(os.path.join(self.skyflat,'*'+self.extension))
        else:
            self.skyflats = None

    def make_calibrated(self):
        self.calibrated = os.path.join(self.night,'calibrated')
        if not os.path.exists(self.calibrated):
            os.mkdir(self.calibrated)
    
    def set_masterdomeflat(self,file):
        self.masterbias = file
    
    def set_masterskyflat(self,file):
        self.masterskyflat = file
    
    def set_masterbias(self,file):
        self.masterbias = file

    def set_masterdark(self,file):
        self.masterbias = file


def stack(image_list, hdu=0, sigma=3, maxiters=3):
    '''
    Given a file list of images stack them using sigma clipping. 

    '''
    shape = fits.getdata(image_list[0],hdu).shape + (len(image_list),)
    stack = np.zeros(shape,dtype=np.float64)
    for i in range(shape[2]):
        stack[:,:,i] = fits.getdata(image_list[i],hdu)
    sigmaclip = SigmaClip(sigma=sigma,maxiters=maxiters)
    master = np.ma.mean(sigmaclip(stack, axis=2),axis=2)
    master[np.where(master.mask)] = np.nan
    return master.data


def darkstack(dark_list, masterbias = 0, hdu=0, sigma=3, maxiters=3):
    masterdark = np.dstack((fits.getdata(dark_list[0],hdu)-masterbias,
                      fits.getdata(dark_list[1],hdu)-masterbias))
    for image in dark_list[2:]:
        masterdark = np.dstack((masterdark,fits.getdata(image,hdu) - masterbias))
    sigmaclip = SigmaClip(sigma=sigma,maxiters=maxiters)
    masterdark = np.ma.mean(sigmaclip(masterdark, axis=2),axis=2)
    masterdark[np.where(masterdark.mask)] = np.nan
    return masterdark.data

def flatstack(flat_list, masterbias=0, masterdark=0, hdu=0):
    sc_norm = SigmaClip(sigma=2,maxiters=3)
    sc_comb = SigmaClip(sigma=3,maxiters=3)
    shape = fits.getdata(flat_list[0],hdu).shape + (len(flat_list),)
    masterflat = np.zeros(shape,dtype=np.float64)
    for i in range(shape[2]):
        masterflat[:,:,i] = fits.getdata(flat_list[i],hdu) - masterbias - masterdark
        norm = np.ma.mean(sc_norm(masterflat[:,:,i]))
        masterflat[:,:,i] /= norm
        
    masterflat = np.ma.mean(sc_comb(masterflat,axis=2),axis=2)
    masterflat[np.where(masterflat.mask)] = np.nan
    return masterflat.data

def calibrate(lights_list, masterdark=0, masterbias=0, masterflat=1, 
                           hdu=0, keytime='EXPTIME',dir=None):
    calibrated_list = []
    if dir is None: iterpath = True

    
        
    for light in lights_list:
        try:
            if iterpath: dir = os.path.dirname(light)
            header = fits.getheader(light,hdu)
            if keytime in header: time = float(header[keytime])
            calibrate = (fits.getdata(light,hdu)-masterdark-masterbias)
            calibrate = correct_flat(calibrate,masterflat) / time
            header = fits.getheader(light,hdu)
            calibrated_list.append(os.path.join(dir,os.path.basename(
                        light).replace('.fits','_calibrated.fits')))
            save_fits(calibrate,header,
                    calibrated_list[-1])
        except Exception as e: 
            print('Error in calibration of',light, ':')
            print(e)
        
    return calibrated_list

def calibrate_night(night):
    if night.bias:
        masterbias = stack(glob.glob(os.path.join(night.bias,'*.fit*')))
        save_fits(masterbias,None,os.path.join(night.bias,'masterbias.fits'))
    else:
        masterbias = 0
    if night.dark:
        masterdark = darkstack(glob.glob(os.path.join(night.dark,'*.fit*')),
                                masterbias)
        save_fits(masterdark,None,os.path.join(night.dark,'masterdark.fits'))
    
    for light in glob.glob(os.path.join(night.light,'*.fit*')):
        calibrate = fits.getdata(light,0)-masterdark-masterbias
        header = fits.getheader(light,0)
        save_fits(calibrate,header,
                os.path.join(night.calibrated,os.path.basename(light).split('.')[-2]+'_calibrated.fits'))

def save_fits(data,header,path):
    hdu = fits.PrimaryHDU(data,header)
    hdu.writeto(path,overwrite=True)
    
def correct_flat(data,flat,mask=True):
    corrected = data/flat 
    if mask:
        corrected[np.where(corrected>np.nanmax(data))] = np.nan
        corrected[np.where(corrected<np.nanpercentile(data[np.where(data<0)],10))] = np.nan
    return corrected

def autoflat(flat_files, masterflat=None, hdu=0,
             config_nc = '-Z15,15 -t0.9 --interpnumngb=9 -d0.95'):

    sc_norm = SigmaClip(sigma=2,maxiters=3)
    sc_comb = SigmaClip(sigma=3,maxiters=3)
    
    if masterflat is None:

        reference = flat_files[np.random.randint(0,len(flat_files),1)[0]]
        reference_data = fits.getdata(reference,hdu)
        gnu = AstroGNU(reference, hdu=hdu, dir=os.path.dirname(reference))
        gnu.noisechisel(config=config_nc)
        reference_data =  (1-gnu.detections) * reference_data
        reference_data[np.where(reference_data==0)] = np.nan
        reference_data /= np.ma.mean(sc_norm(reference_data))

        masterflat = 1
        
    else:
        reference_data = masterflat

    flat = np.zeros(reference_data.shape + (len(flat_files),),dtype=np.float64)

    i=0
    for file in flat_files:
        try:
            data = fits.getdata(file,hdu=hdu) 
            if type(masterflat)==int:
                gnu = AstroGNU(file,hdu=hdu,dir=os.path.dirname(file))
            else:
                gnu = AstroGNU(correct_flat(data,masterflat),hdu=hdu,dir=os.path.dirname(file))
            gnu.noisechisel(config=config_nc)
            flat[:,:,i] = (1-gnu.detections) * data
            flat[:,:,i][np.where(flat[:,:,i]==0)]=np.nan
            norm = np.ma.mean(sc_norm(flat[:,:,i] / reference_data))
            flat[:,:,i] = flat[:,:,i]/norm
        except:
            print(f'Error in {file}')
        i+=1

    masterflat = np.ma.mean(sc_comb(flat, axis=2),axis=2)
    masterflat[np.where(masterflat.mask)]=np.nan

    return masterflat.data
        


def register(list, nref=0,hdu=0,verbose=False):
    """
    Function to register a list of files.
    list: list of files to be registered

    return:
    list_registered: list of images registered
    """
    if verbose: print(f'Start aligment for {len(list)} images')
    if verbose: print(f'Use {list[nref]} for reference image...')
    reference = fits.getdata(list[nref],hdu)
    align_list = [reference]
    del(list[nref])
    for file in list:
        if verbose: print(f'Aligning {file}...')
        target = fits.getdata(file,hdu)
        p, (pos_img, pos_img_rot) = astroalign.find_transform(target, reference)
        aligned = astroalign.register(target, reference)
        align_list.append(aligned)
    if verbose: print(f'Alignment finished.')
    return align_list

def register_arrays(list, nref=0,verbose=False):
    """
    Function to register a list of images.
    list: list of images to be registered

    return:
    list_registered: list of images registered
    """
    if verbose: print(f'Start aligment for {len(list)} images')
    if verbose: print(f'Use {list[nref]} for reference image...')
    reference = list[nref]
    align_list = [reference]
    for target in list:
        p, (pos_img, pos_img_rot) = astroalign.find_transform(target, reference)
        aligned = astroalign.register(target, reference)
        align_list.append(aligned)
    if verbose: print(f'Alignment finished.')
    return align_list
