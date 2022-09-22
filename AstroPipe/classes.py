
from astropy.io import fits
from astropy.wcs import WCS,utils
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, LogStretch

from datetime import datetime
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import os
import subprocess

from lmfit.models import GaussianModel



from .plotting import noise_hist
from .utilities import where, morphologhy, mag_limit
from .sbprofile import background_estimation, isophotal_photometry_fix

import sys
sys.path.append('/Users/pmsa/.local/lib/python3.8/site-packages/sewpy')
import sewpy


class Image:
    '''
    Image class to work with astronomical images.
    To initiate the image class we need:
    - filename: the name of the fits file
    - hdu: (default 0) the hdu of the fits file
    - zp: (default 22.5) the zeropoint of the image    
    '''
    def __init__(self, filename, hdu=0, zp=22.5):

        self.data = fits.getdata(filename, hdu)
        self.header = fits.getheader(filename, hdu)
        self.wcs = WCS(self.header)
        self.pixel_scale = np.mean(utils.proj_plane_pixel_scales(self.wcs)*3600)
        self.file = filename
        self.origin = filename
        self.bkg = 0
        self.name = os.path.basename(filename).split('.')[-2].strip()
        self.zp = zp
        self.hdu = hdu

    def obj(self, ra, dec):
        self.ra = ra
        self.dec = dec
        self.SkyCoord = SkyCoord(ra, dec, frame="fk5", unit="deg")
        self.pix = skycoord_to_pixel(
            self.SkyCoord, self.wcs
        )
        self.pix = (np.float64(self.pix[0]),np.float64(self.pix[1]))

    def obj_id(self, mask):

        if mask.method == 'AstroGNU':
            self.seg_id = mask.objects[int(self.pix[1]), int(self.pix[0])]
        elif mask.method == 'SExtractor':
            self.sex_id = mask.objects[int(self.pix[1]), int(self.pix[0])]

    def sky_to_pixel(self, ra, dec):

        return skycoord_to_pixel(SkyCoord(ra, dec, frame="fk5", unit="deg"), self.wcs)

    def pixel_to_sky(self, xp, yp):

        return pixel_to_skycoord(xp, yp, self.wcs)

    def noise(self,mask,plot=False):
        noise = self.data[where([mask==0, self.data != self.data[0,0]])]
        mu = np.nanmean(noise)
        std = np.nanstd(noise)
        hist = np.histogram(noise,bins=500,range=[mu-6*std,mu+6*std])
        model = GaussianModel()
        params = model.guess(hist[0],x=hist[1][1:],)
        result = model.fit(hist[0],params,x=hist[1][1:]) 
        self.std = result.values['sigma']
        self.header['STD'] = self.std
        self.header['MAG_LIM'] = mag_limit(std, Zp=self.zp, scale=self.pixel_scale)
        if plot:
            noise_hist(result,out=plot)
        
    def crop(self, center, width=(500,500)):
        self.cropped = self.data[np.int64(center[1]-width[1]):np.int64(center[1]+width[1]),
                np.int64(center[0]-width[0]):np.int64(center[0]+width[0])]
        self.crop_param = {'center':np.int64(center),
                            'width':np.int64(width)}
    def set_mask(self,mask):
        self.data = ma.masked_array(ma.getdata(self.data), mask=mask)
    
    def set_data(self, data):
        if hasattr(self,'mask'):
            self.data = ma.masked_array(data, 
                mask=np.ma.getmask(self.data))
        else:
            self.data = data

    def show(self,vmin=None,vmax=None,cmap='nipy_spectral_r',
                    width=400,ax=None):
        mask = hasattr(self.data, 'mask')
        if mask: data = self.data.data
        if not mask: data = self.data
        if not vmin: 
            if hasattr(self,'maglim'): vmin = self.std
            else: vmin = self.mu_to_counts(28.5)
        if not vmax:  vmax = self.mu_to_counts(18.5)

        mmax = self.counts_to_mu(vmax)
        mmin = self.counts_to_mu(vmin)

        norm = ImageNormalize(self.data,vmin=vmin,vmax=vmax,stretch=LogStretch())
        fig = plt.figure()
        im = plt.imshow(data - self.bkg,norm=norm,interpolation='none', origin='lower',cmap=cmap)
        plt.xlim([self.pix[0]-width,self.pix[0]+width])
        plt.ylim([self.pix[1]-width,self.pix[1]+width])
        bar = fig.colorbar(im,ticks=self.mu_to_counts(np.arange(mmax,mmin,2.5)))
        ticklabels = 5*np.round((self.zp-2.5*np.log10(bar.get_ticks()))*2)/10
        bar.set_ticklabels(['{:2.1f}'.format(i) for i in ticklabels])
        plt.tight_layout()
        if mask:
            transparent = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
            gray = matplotlib.colors.colorConverter.to_rgba('black',alpha = 0.3)
            cmap = matplotlib.colors.ListedColormap([transparent, gray])
            plt.imshow(self.data.mask, origin='lower',cmap=cmap)
        return im 
    
    def get_morphology(self,n=4,width=1000) :
        data = self.data
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        binary = np.zeros(data.shape,dtype=np.float32)
        index = np.where(data.mask==False)
        binary[index] = data[index]
        if hasattr(self,'std'): 
            binary[where([data.mask==False,
                    data.data>n*std])] = 1
            binary[where([data.mask==False,
                    data.data<n*std])] = 0
        binary = binary[np.int32(self.pix[1]-width):np.int32(self.pix[1]+width),
                        np.int32(self.pix[0]-width):np.int32(self.pix[0]+width)]
        self.pa,self.r_eff,self.eps = morphologhy(binary)
        if self.pa < 0: self.pa += 180
    
    def set_morphology(self,pa=None,eps=None,r_eff=None):
        if pa: self.pa = pa
        if eps: self.eps = eps
        if r_eff: self.r_eff = r_eff

    def get_background(self,width=5, max_r=None, out=None):
        self.bkg,self.bkg_radius = background_estimation(self,width=width,max_r=max_r,out=out)
    
    def set_background(self,bkg):
        self.bkg = bkg
        self.header['BKG'] = self.bkg
    
    def set_catalog(self, table):
        self.catalog = table
    
    def set_std(self, std):
        self.std = std
        self.maglim = self.counts_to_mu(self.std)

    def set_maglim(self, mag_lim):
        self.maglim = mag_lim
        self.std = self.mu_to_counts(self.maglim)
    
    def set_extinction(self,Av):
        self.Av = Av
    
    def counts_to_mu(self,counts):
        return self.zp -2.5*np.log10(counts/self.pixel_scale**2)
    
    def mu_to_counts(self, mu):
        return 10**((self.zp-mu)/2.5)*self.pixel_scale**2


class SExtractor:
    '''
    Class to run SExtractor on a FITS image.
    
    To run SExtractor: 
        (1st) You need to have it installed on your computer
    
    This class lets you create an instance of SExtractor with 
    the parameters you set.

    Then, you run the class on a FITS image and it returns the 
    segmentation map and catalog from SExtractor.
    '''

    def __init__(self, params=None, config=None):

        self.files_default = {
            "FILTER_NAME": "/Users/pmsa/Documents/PhD/Projects/SExtractor/Param/default.conv",
            "PSF_NAME": "/Users/pmsa/Documents/PhD/Projects/SExtractor/Param/default.psf",
            "STARNNW_NAME": "/Users/pmsa/Documents/PhD/Projects/SExtractor/Param/default.nnw",
            'PHOT_FLUXFRAC': 0.5,
        }

        self.params_default = [
            "NUMBER",
            "ALPHA_J2000",
            "DELTA_J2000",
            "MAG_ISO",
            "MAGERR_ISO",
            'MU_MAX',
            "BACKGROUND",
            "A_WORLD",
            "B_WORLD",
            "THETA_J2000",
            "CLASS_STAR",
            'FLUX_RADIUS',
            'KRON_RADIUS'
        ]

        self.params = self.params_default
        self.config = self.files_default
        self.method = 'SExtractor'

        if params:
            self.add_params(params)
        if config:
            self.add_config(config)

    def add_params(self, p_list):
        
        self.params = np.unique(self.params + p_list).tolist()

    def add_config(self, c_dict):
        for key in c_dict:
            self.config[key] = c_dict[key]

    def run(self, file):
        sew = sewpy.SEW(params=self.params, config=self.config, sexpath="sex")
        if type(file)==str:
            self.file = file
        else:
            self.file = 'sextractor_image.fits'
            fits.PrimaryHDU(file).writeto(self.file,overwrite=True)

        self.out = sew(self.file)
        self.catalog = self.out['table']
        self.seg_file = self.config['CHECKIMAGE_NAME']
        self.objects = fits.getdata(self.seg_file)

        
        
        # if 'CHECKIMAGE_NAME' in self.config:
        #     try:
        #         os.system('mv '+self.out['catfilepath'
        #             ]+ ' ' + os.path.dirname(self.seg_file))
        #     except:
        #         print('CHECKIMAGE not moved')

    def load_catalog(self, file):
        self.cat = Table.read(file, format='ascii.sextractor')

class AstroGNU():
    def __init__(self, data, hdu=0, dir='', loc='/opt/local/bin/'):
        if type(data) == str:
            self.file = data
        else:
            self.file = os.path.join(dir,'save.fits')
            new_hdu = fits.PrimaryHDU(data)
            new_hdu.writeto(self.file,overwrite=True)
            hdu=0

        if not dir: dir = os.path.basename(self.file)
        self.directory = dir
        self.hdu = hdu
        self.extension = self.file.split('.')[-1]
        self.name = os.path.basename(self.file).split('.')[-2]
        self.loc = loc
        self.method = 'AstroGNU'

    def noisechisel(self,config=''):
        
        self.nc_file  = os.path.join(self.directory,self.name+'_nc.fits')
        
        self.nc_config = config
        self.nc_cmd = f'astnoisechisel {self.file} -h{self.hdu} {self.nc_config} -o{self.nc_file} -q'
        self.nc_cmd = os.path.join(self.loc,self.nc_cmd)

        os.system(self.nc_cmd)
        # ncSub = subprocess.Popen(self.nc_cmd)
        # ncSub.wait()
        self.detections = fits.getdata(self.nc_file,'DETECTIONS')
        self.background = np.nanmean(fits.getdata(self.nc_file,'SKY'))
    
    def segment(self, config='',clumps=False):
        self.seg_file  =  os.path.join(
                            self.directory, 
                            self.name+'_seg.fits')

        self.seg_config = config
        self.seg_cmd = f'astsegment {self.nc_file} {self.seg_config} -o{self.seg_file} -q'
        self.seg_cmd = os.path.join(self.loc,self.seg_cmd)
        os.system(self.seg_cmd)

        self.objects = fits.getdata(self.seg_file,'OBJECTS')
        

        if clumps: self.clumps = fits.getdata(self.seg_file,'CLUMPS')

    def make_catalog(self,config='',params='',fracmax=0.05,zp=22.5):
        self.cat_file = os.path.join(self.directory,self.name+'_nc.fits')
        self.mkc_config = config
        self.mkc_cmd = f'astmkcatalog -irdmGnABp --fwhm --fracmaxradius1 --fracmax={fracmax} '
        self.mkc_cmd += f' {self.mkc_config} --zeropoint={zp} {self.seg_file} -o{self.cat_file} -q'
        self.mkc_cmd = os.path.join(self.loc,self.mkc_cmd)
        
        os.system(self.mkc_cmd)
                
        self.catalog = Table.read(self.cat_file)

    def remove(self,nc=True,seg=True,cat=True):
        if nc:  os.system('rm '+self.nc_file)
        if seg: os.system('rm '+self.seg_file)
        if cat: os.system('rm '+self.cat_file)
        


class Directories():
    def __init__(self, name, path=None):
        if not path: path = os.path.dirname(name)
        self.out = os.path.join(path,'AstroPipe_'+name)
        if not os.path.exists(self.out):
             os.mkdir(self.out)
        self.temp = os.path.join(self.out,'temp_'+name)
        if not os.path.exists(self.temp):
             os.mkdir(self.temp)
    def set_regions(self,path):
        self.regions = path
    def set_mask(self,file):
        self.mask = file
    def set_profile(self,file):
        self.profile = file
        
class log_class:
    """Create a log file for the outputs of each execution of the pipeline."""

    def __init__(self):
        """ Initialize log file"""

        self.name = "log_{}.txt".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

        f = open(self.name, "w+")
        f.write(
            75 * "=" + "\n"
            "AstroPipe Log file" + "\n" + 75 * "=" + "\n"
        )
        f.close()

        self.insert_log = ">> " + self.name

    def insert_line(self):
        """Command line to import output in log file"""

        f = open(self.name, "a+")
        f.write(75 * "-" + "\n")
        f.close()



class SourceExtractor():

    def __init__(self, file, hdu=0, config=None):
        self.file = file
        self.hdu = hdu




source_extractor_keys = {
    'DEBLEND_NTHRESH': '--partition-threshold-count',
    'DEBLEND_MINCONT': '--partition-threshold-fraction',
    'DETECT_MINAREA':'--partition-minimum-area',
    'DETECT_THRESH':'--detection-threshold',
    'BACK_SIZE':'--background-cell-size',
    'BACK_FILTERSIZE': '--smoothing-box-size',
    'CHECKIMAGE_NAME':'--check-image-segmentation',
    'CATALOG_NAME':'--output-catalog-filename',
    'MAG_ZEROPOINT':'--magnitude-zero-point',
}