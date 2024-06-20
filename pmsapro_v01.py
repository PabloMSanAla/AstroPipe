#%%
# @pmsastro
from AstroPipe.calibration import structure
from AstroPipe.calibration import *
# calibrate, save_fits, autoflat, calibrate_night, stack, darkstack
from AstroPipe.classes import AstroGNU
from AstroPipe.utils import get_pixel_scale, merge_pdf, mag_limit
from AstroPipe.plotting import show
import subprocess

import os
import pandas as pd
from os.path import join, isfile
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings("ignore")

from PyPDF2 import PdfReader, PdfMerger 

def create_log(path, extension='.fits'):
    table = {'frame':[],'root':[],'type':[],'object':[],'ra':[],'dec':[],'exptime':[],
             'airmass':[],'filter':[],'date':[],'temp':[],'gain':[],'offset':[]}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                header = fits.getheader(join(root,file))
                table['frame'].append(file)
                table['root'].append(root)
                table['object'].append(header['OBJECT'].replace(' ','') if 'OBJECT' in header else 'None')
                table['type'].append(header['IMAGETYP'] if 'IMAGETYP' in header else 'None')
                table['ra'].append(header['RA'] if 'RA' in header else 'None')
                table['dec'].append(header['DEC'] if 'DEC' in header else 'None')
                table['exptime'].append(header['EXPTIME'] if 'EXPTIME' in header else 'None')
                table['airmass'].append(header['AIRMASS'] if 'AIRMASS' in header else 'None')
                table['filter'].append(header['FILTER'] if 'FILTER' in header else 'None')
                table['date'].append(header['DATE-OBS'] if 'DATE-OBS' in header else 'None')
                table['temp'].append(header['CCD-TEMP'] if 'CCD-TEMP' in header else 'None')
                table['gain'].append(header['GAIN'] if 'GAIN' in header else 'None')
                table['offset'].append(header['OFFSET'] if 'OFFSET' in header else 'None')
    return pd.DataFrame(table)



# path = '/Volumes/IACDrive/AstroFotos/AstroPipe_Test'
path = '/Volumes/IACDrive/AstroFotos/2024'
subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]

outPath = '/Volumes/IACDrive/AstroFotos/Procesing/2024/M51-Abril/second'
if not os.path.exists(outPath):
    os.makedirs(outPath)

calPath = join(outPath,'Calibrated')
if not os.path.exists(calPath):
    os.makedirs(calPath)

hdu = 0


band = 'R'
object = 'M51'

createMbias = False
masterbiasFile = join(outPath,f'masterbias.fits')

createMdark = False
masterdarkFile = join(outPath,f'masterdark.fits')

createAutoFlat = False
autoflatFile = join(outPath,f'autoflat_{band}.fits')

mosaicFile = join(calPath,f"{object}_final.fits")


checkplots = ['photometry']
#%% Generate MasterBias, MasterDark and Calibrate each night individially

####################################################
# (1.1) Get files from each night
# 
# TODO: Maybe select it with the log... hard to 
# automate
####################################################

verbose=True

nights = []
for n  in subfolders:
    nights.append(structure(n, band='R'))

biasList = []
darkList = []
lightList = []
for night in nights:
    if night.bias:
        biasList += night.biasList
    if night.dark:
        darkList += night.darkList
    if night.light:
        lightList += night.lightList


if verbose:
    print('Number of files found:')
    print(f'Bias:  {len(biasList):4d}')
    print(f'Dark:  {len(darkList):4d}')
    print(f'Light: {len(lightList):4d}')

#%%

if not os.path.exists(join(path,'log.csv')):
    log = create_log('/Volumes/IACDrive/AstroFotos/2024')
    log.to_csv(join(path,'log.csv'), index=False)
else:
    log = pd.read_csv(join(path,'log.csv'))

darksind = np.where((log['type'] == 'Dark Frame') * (log['exptime']==240))[0]
darkList = []
for i in darksind:
    darkList.append(join(log['root'][i],log['frame'][i]))

lightind = np.where((log['type'] == 'Light Frame') * (log['exptime']==240
                    ) *  (log['filter']=='R'))[0] #* (log['object']=='M51')
lightList = []
for i in lightind:
    lightList.append(join(log['root'][i],log['frame'][i]))

if verbose:
    print('Number of files found:')
    print(f'Bias:  {len(biasList):4d}')
    print(f'Dark:  {len(darkList):4d}')
    print(f'Light: {len(lightList):4d}')

#%%
    
####################################################
#       (1.2) Check stats of files and remove
#             bad frames
#TODO: add elongation of stars, and FWHM. Also check
#      bias and darks to ensure same stats. 
####################################################
    

def stats_table(fileList, outFile, hdu=0):
    columns = ['frame','ra','dec','exptime','airmass',
               'filter','date', 'mean','median','std']
    statsDict = {}
    for c in columns:
        if c == 'frame':
            dtype = f'U{len(fileList[0])+20}'
        elif c == 'filter' or c == 'date':
            dtype = 'U30'
        else:
            dtype = np.float16
        statsDict[c] = np.zeros(len(fileList)).astype(dtype)

    for i,file in enumerate(fileList):
        data = fits.getdata(file,hdu)
        header = fits.getheader(file,0)
        stats = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        statsDict['frame'][i] = file
        statsDict['mean'][i] = np.nanmean(data)
        statsDict['median'][i] = stats[1]
        statsDict['std'][i] = stats[2]
        statsDict['ra'][i] = header['RA'] if 'RA' in header else 0
        statsDict['dec'][i] = header['DEC'] if 'DEC' in header else 0
        statsDict['exptime'][i] = header['EXPTIME'] if 'EXPTIME' in header else 0
        statsDict['airmass'][i] = header['AIRMASS'] if 'AIRMASS' in header else 0
        statsDict['filter'][i] = header['FILTER'] if 'FILTER' in header else 0
        statsDict['date'][i] = header['DATE-OBS'] if 'DATE-OBS' in header else 0
    
    statsTable = pd.DataFrame(statsDict)
    statsTable.to_csv(outFile, index=False)
    return statsTable


statsTable = stats_table(lightList, join(calPath,'stats.csv'))

# Filtering bad frames

nsigma = 3
_,med,_ = sigma_clipped_stats(statsTable['median'])
_,std,_ = sigma_clipped_stats(statsTable['std'])
filterInd = (statsTable['median'] > med-nsigma*std) & (statsTable['median'] < med+nsigma*std)
statsTable['Good'] = filterInd
statsTable.to_csv(join(calPath,'stats.csv'), index=False)
filterind = np.ones_like(statsTable).astype(bool)

if not os.path.exists(join(calPath,'badFrames')):
    os.makedirs(join(calPath,'badFrames'))
for f in np.array(lightList)[~filterInd]:
    os.rename(f, join(calPath,'badFrames', os.path.basename(f)))


for f in lightList[-24:]:
    data = fits.getdata(f,hdu)
    _,med,_ = sigma_clipped_stats(data)
    ax = show(data-med)
    ax.set_title(os.path.basename(f),fontsize=12)
    fig = ax.get_figure()
    figsave = join(calPath,os.path.basename(f).replace('.fits','.pdf'))
    fig.savefig(figsave, format='pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)


inflist = glob.glob(join(calPath,'*.pdf'))
merge_pdf(inflist, join(calPath,'images.pdf'))

for f in inflist:
    os.remove(f)
#%%
####################################################
# (2.) Correct Bias, Dark and Flat  
#       (2.1) Generate or Read MasterBias and 
#              MasterDark
####################################################

if createMbias or not isfile(masterbiasFile):
    masterbias = stack(biasList, hdu=0)
    header = fits.getheader(biasList[0],0)
    header['COMMENT'] = f'Masterbias generated with AstroPipe'
    header['NFRAMES'] = (len(biasList), 'Number of frames stacked')
    save_fits(masterbias, header, masterbiasFile)
else:
    masterbias = fits.getdata(masterbiasFile,0)


# darkList = darkList[:40]
if createMdark or not isfile(masterdarkFile):
    masterdark = darkstack(darkList, masterbias=masterbias, hdu=0)
    header = fits.getheader(darkList[0],0)
    header['COMMENT'] = f'Masterdark generated with AstroPipe'
    header['NFRAMES'] = (len(darkList), 'Number of frames stacked')
    save_fits(masterdark, header, masterdarkFile)
else:
    masterdark = fits.getdata(masterdarkFile,0)

#%% Run astrometry so GNUastro works properly also remove bias and dark

####################################################
#       (2.2) Correct bias and dark from lights
#             and run astrometry
####################################################
    
astrom = astrometry()
for f in lightList:
    calFile = join(calPath, os.path.basename(f))
    tempFile = calFile.replace('.fits','_t.fits')
    data = fits.getdata(f,hdu)
    header = fits.getheader(f,hdu)
    data = data - masterbias - masterdark
    fits.PrimaryHDU(data, header).writeto(tempFile,overwrite=True)
    ra = header['RA'] if 'RA' in header else None
    dec = header['DEC'] if 'DEC' in header else None
    astrom(tempFile, out=calFile, ra=ra,dec=dec)
    os.remove(tempFile)
    os.remove(tempFile.replace('.fits','.wcs'))

calList = glob.glob(join(calPath,'*.fits'))


#%% For all lights calibrated generates autoflat
    
####################################################
#       (2.4) Generate AutoFlat
#             and correct it.
####################################################

hdu=0
config_nc =f'-Z{30},{30} -t{0.6} --interpnumngb={9} -d{0.8} '

if createAutoFlat or not isfile(autoflatFile):
    masterflat = autoflat(calList, config_nc=config_nc, hdu=hdu)
    config_nc =f'-Z{30},{30} -t{0.3} --interpnumngb={9} -d{0.8} '
    masterflat = autoflat(calList, masterflat=masterflat, config_nc=config_nc, hdu=hdu)
    masterflat[np.isnan(masterflat)] = 1
    masterflat[masterflat<0]=1
    save_fits(masterflat, None, autoflatFile)
else:
    masterflat = fits.getdata(autoflatFile)
    masterflat[np.isnan(masterflat)] = 1

# %% Given MasterBias, MasterDark and AutoFlat, calibrate each night
calList = [join(calPath,os.path.basename(f)) for f in lightList if object not in f]


masterflat[np.isnan(masterflat)] = 1
calList = calibrate(calList, masterflat=masterflat, hdu=hdu, dir=night.calibrated, mask=False)

# %% 
####################################################
#   (3) Create final mosaic
#       (3.1) Find center, scale and size of mosaic
####################################################

# Fist found using the wcs of all images the final frame to resample all images.

def deg_to_hms(ra_deg,dec_deg):
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
    return coord.to_string(style="hmsdms", precision=2, pad=True)

def deg_to_dms(dec_deg):
    coord = SkyCoord(ra=0, dec=dec_deg, unit="deg")
    return coord.to_string(style="hmsdms", precision=2, pad=True)


def get_corners(fileList):
    '''
    Get the equatorial coordinates (ra,dec) of the corners of 
    the footprint of a list of images. Also returns the mean pixel scale.

    Parameters
    ----------
        fileList : list
            List of images.
    
    Returns
    -------
        min_ra, max_ra, min_dec, max_dec, scale : float
            Minimum and maximum values of RA and Dec [deg] and 
            mean pixel scale [pixel/arcsec].
    '''
    ras, decs, pixscales = [], [], []
    for file in fileList:
            header = fits.getheader(file)
            wcs = WCS(header)
            height, width = fits.getdata(file).shape
            pixels_corners = np.array([[0, 0], [0, height], [width, height], [width, 0]])
            ra_dec_corners = wcs.pixel_to_world_values(pixels_corners[:, 0], pixels_corners[:, 1])
            ras.extend(ra_dec_corners[0])
            decs.extend(ra_dec_corners[1])
            pixscales.append(get_pixel_scale(header))

    min_ra, max_ra = np.nanmin(ras), np.nanmax(ras)
    min_dec, max_dec = np.nanmin(decs), np.nanmax(decs)

    return min_ra, max_ra, min_dec, max_dec, np.nanmean(pixscales)

def change_config(file, params, length=23):
    with open(file,'r') as f:
        lines = f.readlines()
    for key,value in params.items():
        for i,line in enumerate(lines):
            if line.startswith(key+' '):
                lines[i] = f'{key.ljust(length)} {value}\n'
    with open(file,'w') as f:
        f.writelines(lines)

# Measure corners of the footprint of the images
# and define center, size and scale of mosaic 

calList = glob.glob(join(calPath,'*ted.fits'))
# calList = [l for l in calList if object not in l]
min_ra, max_ra, min_dec, max_dec, scale = get_corners(calList)

scale = np.round(scale, 2)
ra0 = (min_ra+max_ra)/2
dec0 = (min_dec+max_dec)/2
width, height = np.int64((max_ra-min_ra)*3600/scale), np.int64((max_dec-min_dec)*3600/scale)

#%%
####################################################
#       (3.1) Run scamp to refine astrometry
####################################################

## Run SExtractor to use catalogs for Scamp

sexFile = join(calPath, 'sex.config')
sexParamFile = sexFile.replace('.config','.param')
os.system(f'sex -dd >> {sexFile}')

sexParamsScamp = ['NUMBER','ALPHA_J2000','DELTA_J2000',
    'XWIN_IMAGE','YWIN_IMAGE','ERRAWIN_IMAGE',
    'ERRBWIN_IMAGE','ERRTHETAWIN_IMAGE' ,'FLUX_AUTO',
    'FLUXERR_AUTO', 'FLAGS', 'FLAGS_WEIGHT',
    'FLUX_RADIUS', 'ELONGATION', 'SPREAD_MODEL',
    'SPREADERR_MODEL']

with open(sexParamFile,'w') as f:
    f.writelines(f"{item}\n" for item in sexParamsScamp)

sex_param_path = '/Users/pmsa/Documents/AstroPipe/extern/SExtractor'
sexConfigScamp = {'CATALOG_TYPE':       'FITS_LDAC',
                  'PARAMETERS_NAME':    sexParamFile,
                  'DETECT_THRESH':      5,
                  'FILTER':             'N',
                  'BACKPHOTO_TYPE':     'LOCAL',
                  'FILTER_NAME':        join(sex_param_path,'default.conv'),
                  'PSF_NAME':           join(sex_param_path, "default.psf"),
                  'STARNNW_NAME':       join(sex_param_path,"default.nnw"),}

change_config(sexFile, sexConfigScamp)
sexBatch = join(calPath,'sexBatch.sh')
with open(sexBatch,'w') as f:
    f.write('#!/bin/bash\n')
    for file in calList[4:6]:
        f.write(f'sex {file} -c {sexFile} -CATALOG_NAME {file.replace(".fits",".cat")} & \n ')

cmd = ['bash', sexBatch]
result = subprocess.run(cmd, capture_output=True, text=True)

catalogs = glob.glob(join(calPath,'*.cat'))
print('sex done')
scampFile = join(calPath,'scamp.config')
os.system(f'scamp -dd > {scampFile}')

scampParams = {'ASTREF_CATALOG':    'GAIA-EDR3',
               'ASTREF_BAND':       'G',
               'ASTREFMAG_LIMITS':  '5.0,25.0',
               'SOLVE_PHOTOM':      'N',
               'SN_THRESHOLDS':     '3.0,1000.0',
               'CHECKPLOT_DEV':     'NULL'}

change_config(scampFile, scampParams, length=22)

scampListFile = join(calPath,'scampList.txt')
with open(scampListFile,'w') as f:
    for cat in catalogs:
        f.write(f'{cat} \n')

cmd = ['scamp', f'@{scampListFile}', '-c', scampFile]
result = subprocess.run(cmd, capture_output=True, text=True)


#%%
####################################################
#       (3.2) Resample all images to the same grid 
#             using SWarp
####################################################
### TODO: Parallelize this part
###       unsing GNU parallel
###       https://www.gnu.org/software/parallel/
###       by creating a file with the commands
###       and then run parallel 
####################################################


originalPath = os.getcwd()
os.chdir(calPath)

swarpFile = join(calPath,'swarp.config')
os.system(f'swarp -dd >> {swarpFile}')

swarpParams = {'CENTER_TYPE':       'MANUAL',
               'CENTER':            f'{ra0}, {dec0}',
               'PIXELSCALE_TYPE':   'MANUAL',
               'PIXEL_SCALE':       scale,
               'IMAGE_SIZE':        f'{np.int64(width)+2},{np.int64(height)+2}',
               'SUBTRACT_BACK':     'N',
               'MEM_MAX':           '2047',
               'COMBINE_BUFSIZE':   '2047',
               'VMEM_MAX':          '4094'
               }

change_config(swarpFile, swarpParams)

swarpBatch = join(calPath,'swarpBatch.sh')
with open(swarpBatch,'w') as f:
    f.write('#!/bin/bash\n')
    for file in calList[4:6]:
        f.write(f'swarp {file} -c {swarpFile} -IMAGEOUT_NAME {file.replace("_calibrated.fits",".resamp.fits")} & \n')

# os.system(f'bash {swarpBatch}')
cmd = ['bash', swarpBatch]
result = subprocess.run(cmd, capture_output=True, text=True)

#%%

####################################################
#       (3.3) Run SExtractor to refine astrometry
#             and do photometry
####################################################

sexFile = join(calPath, 'sex.config')
sexParamFile = sexFile.replace('.config','.param')
os.system(f'sex -dd >> {sexFile}')

sexParamsScamp = ['NUMBER','ALPHA_J2000','DELTA_J2000',
    'XWIN_IMAGE','YWIN_IMAGE','ERRAWIN_IMAGE',
    'ERRBWIN_IMAGE','ERRTHETAWIN_IMAGE' ,'FLUX_AUTO',
    'FLUXERR_AUTO', 'FLAGS', 'FLAGS_WEIGHT',
    'FLUX_RADIUS', 'ELONGATION', 'SPREAD_MODEL',
    'SPREADERR_MODEL']

with open(sexParamFile,'w') as f:
    f.writelines(f"{item}\n" for item in sexParamsScamp)

sex_param_path = '/Users/pmsa/Documents/AstroPipe/extern/SExtractor'
sexConfigScamp = {'CATALOG_TYPE':       'FITS_LDAC',
                  'PARAMETERS_NAME':    sexParamFile,
                  'DETECT_THRESH':      5,
                  'DETECT_MAXAREA':     400,
                  'DEBLEND_NTHRESH':    1,
                  'PIXEL_SCALE':        scale,
                  'FILTER':             'N',
                  'BACKPHOTO_TYPE':     'GLOBAL',
                  'FILTER_NAME':        join(sex_param_path,'default.conv'),
                  'PSF_NAME':           join(sex_param_path, "default.psf"),
                  'STARNNW_NAME':       join(sex_param_path,"default.nnw"),}

change_config(sexFile, sexConfigScamp)

swarplist = glob.glob(join(calPath,'*resamp.fits'))
sexBatch = join(calPath,'sexBatch.sh')
with open(sexBatch,'w') as f:
    f.write('#!/bin/bash\n')
    for file in swarplist:
        f.write(f'sex {file} -c {sexFile} -CATALOG_NAME {file.replace(".fits",".cat")} & \n ')

cmd = ['bash', sexBatch]
result = subprocess.run(cmd, capture_output=True, text=True)

#%% Photometry calibration

####################################################
#       (3.3) Using Gaia and SDSS, do photometry
#             using flux auto from SExtractor
####################################################

from astropy.coordinates import SkyCoord
from astropy.table import Table

from scipy.optimize import curve_fit

from AstroPipe.query import query_gaia, query_sdss, cross_match


zp = 22.5
refcat = 'SDSS-DR12'

radius = np.nanmax([width,height])*scale/(3600*2)

# Create reference catalog
gaiacat = query_gaia(ra0, dec0, radius)
sdsscat = query_sdss(ra0, dec0, radius)
refcat = cross_match(gaiacat, sdsscat)
#%%
# For each catalog done for scamp, cross match with reference catalog
# and calculate zero point, and flux calibration
targetzp = 22.5

columnid = f'modelMag_{band.lower()}'
line = lambda x, zp: zp + x

catalogs = glob.glob(join(calPath,'*resamp.cat'))
for cat in catalogs:
    table = Table(fits.open(cat)['LDAC_OBJECTS'].data)
    table['ALPHA_J2000'].name,table['DELTA_J2000'].name = 'ra','dec'
    merge = cross_match(table, refcat)
    
    # Calculate zero point
    flux =  merge['FLUX_AUTO']
    flux_err = merge['FLUXERR_AUTO']
    mag = -2.5*np.log10(flux)
    diff = np.zeros_like(mag)
    for i in range(2):   # Three iterations to remove outliers
        _,med,std = sigma_clipped_stats(np.abs(diff))
        index = np.isfinite(mag) * (flux>0) * (np.abs(diff) <= med+3*std)
        mag[index][np.isnan(mag[index])] = 0
        popt, pcov = curve_fit(line, mag[index], merge[columnid][index])    
        diff = merge[columnid] - (zp -2.5*np.log10(flux))
        zp, zperr = popt[0], np.mean(np.sqrt(diff[index]**2))

    # Update image and scale to target zeropoint
    data = fits.getdata(cat.replace('.cat','.fits'))
    header = fits.getheader(cat.replace('.cat','.fits'))
    fluxscale = 10**(0.4*(targetzp - zp))
    data = data * fluxscale
    header['COMMENT'] = f'Photometry calibrated with AstroPipe'
    header['ZP'] = (targetzp, 'Photometric Zero Point')
    header['ZPERR'] = (zperr, 'Photometric Zero Point Error')
    header['FLUXSCALE'] = (fluxscale, 'Flux Scale factor')
    fits.PrimaryHDU(data, header).writeto(cat.replace('.cat','_phot.fits'), overwrite=True)

    # Check plot
    if 'photometry' in checkplots:
        title = f'Photometry Calibration with {np.sum(index)} sources\n'
        title += f'zp = {zp:2.2f} +- {zperr:1.2f} magnitudes'
        x = np.linspace(np.nanmin(flux),np.nanmax(flux),300)
        fig,ax = plt.subplots(1,1)
        ax.plot(flux[index],merge[columnid][index],'g.',label='Used for fit',alpha=0.3)
        ax.plot(flux[~index],merge[columnid][~index],'r.',label='Not used',alpha=0.2)
        ax.plot(x, zp -2.5*np.log10(x),'b-',label='Linear Fit')
        fig.suptitle(title,fontsize=14)
        ax.invert_yaxis()
        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel('FLUX_AUTO$_{TSS}$ [ADUs]')
        ax.set_ylabel(columnid)
        plt.tight_layout()
        fig.savefig(cat.replace('.cat','_phot.pdf'), format='pdf', dpi=100, bbox_inches='tight')
        plt.close(fig)
    

inflist = glob.glob(join(calPath,'*phot.pdf'))
merged = merge_pdf(inflist, join(calPath,'photometry.pdf'))

if merged:
    for f in inflist:
        os.remove(f)


#%% Final mosaic
    
####################################################
#       (3.5) Measure mode of histogram to use it 
#             as background to create seed mosaic
####################################################

finalList = glob.glob(join(calPath,'*sw.head'))

backgrounds = []
swarptxt = join(calPath,'swarpList.txt')
with open(swarptxt, 'w') as f:
    for file in finalList:
        data = fits.getdata(file.replace('.head','.fits'))
        _,med,_ = sigma_clipped_stats(data[data!=0],sigma=2)
        backgrounds.append(med)
        f.write(file.replace('head','fits')+'\n')

backtext = ','.join([f'{b:.3f} ' for b in backgrounds])
swarpParams = { 'SUBTRACT_BACK':     'Y',
                'BACK_TYPE':         'MANUAL',
                'BACK_DEFAULT':      backtext,
                'COMBINE_TYPE':      'WEIGHTED AND CLIPPED',
                'CLIP_SIGMA':        '3.0',}

change_config(swarpFile, swarpParams)

# os.system(f'swarp @{swarptxt} -c {swarpFile} -IMAGEOUT_NAME {mosaicFile}')

#%%
####################################################
#       (3.7) Use seed mosaic to create mask then
#             fit background with Zernike 
#             polynomials and make final mosaic
####################################################

from AstroPipe.classes import AstroGNU, Image
from AstroPipe.plotting import show

image = Image(mosaicFile, zp=22.5, hdu=0)
image.obj(ra0,dec0)

resampList = glob.glob(join(calPath,'*sw.fits'))

maskFile = mosaicFile.replace('.fits','_nc.fits')
if not os.path.isfile(maskFile):
    gnu = AstroGNU(mosaicFile)
    gnu.noisechisel(config=config_nc, keep=True)
else:
    image.set_mask(fits.getdata(maskFile,'DETECTIONS'))

#%%


from astropy.stats import SigmaClip
from scipy.optimize import curve_fit

def tiltedSky(coords,c,dx,dy,x0,y0):
    x,y = coords
    sky = c + dx*(x-x0) + dy*(y-y0)
    return sky


photStats = {'frame':[],'zp':[],'zperr':[],
                'depth':[], 'sky':[]}

checkplots += ['bkg']

for f in resampList[-5:]:
    data = fits.getdata(f)

    framearg = np.argwhere(~np.isnan(data)) 

    x1,x0 = np.max(framearg[:,1]),np.min(framearg[:,1])
    y1,y0 = np.max(framearg[:,0]),np.min(framearg[:,0])

    offset = np.abs(np.array(data.shape) - np.array(image.data.shape))
    offx, offy = offset[1], offset[0]

    cutout = np.zeros((y1-y0,x1-x0)) + data[y0:y1,x0:x1] 
    mask =  image.data.mask[y0:y1,x0:x1] + (cutout==0) + np.isnan(cutout)
    cutout[mask] = 0    # there is some offset between the mosaic and resampled images. wairning?
    cutout = np.ma.masked_array(cutout, mask=mask)
    xx,yy = np.meshgrid(np.arange(x1-x0),np.arange(y1-y0))
    skyparam = curve_fit(tiltedSky, [xx.flatten()[~cutout.mask.flatten()], 
                                    yy.flatten()[~cutout.mask.flatten()]],
                                    cutout.flatten()[~cutout.mask.flatten()],
                        p0=[np.ma.median(cutout),0,0,cutout.shape[1]/2,cutout.shape[0]/2])

    skymodel = tiltedSky([xx,yy],skyparam[0][0],skyparam[0][1],
                        skyparam[0][2],skyparam[0][3],skyparam[0][4])
    
    # Saving stats
    header = fits.getheader(f)
    mean, med, std = sigma_clipped_stats(cutout-skymodel)
    depth = mag_limit(std, Zp=image.zp, scale=image.pixel_scale)
    sky = image.counts_to_mu(np.nanmean(skymodel))

    photStats['frame'].append(f)
    photStats['zp'].append(image.zp)
    photStats['zperr'].append(header['ZPERR'])
    photStats['depth'].append(depth)
    photStats['sky'].append(sky)

    if 'bkg' in checkplots:
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6), sharex=True, sharey=True)
        show(cutout,zp=22.5, ax=ax1)
        show(skymodel,zp=22.5, ax=ax2)
        show(cutout-skymodel,zp=22.5, ax=ax3)
        title=f'z={skyparam[0][0]:.2e} + {skyparam[0][1]:.2e}*(x-{skyparam[0][3]:.2e}) + {skyparam[0][2]:.2e}*(y-{skyparam[0][4]:.2e})'
        fig.suptitle(os.path.basename(f)+'  bkg:'+title,fontsize=12)
        fig.tight_layout()
        fig.savefig(f.replace('.fits','_bkg.jpg'), dpi=100, bbox_inches='tight')
        plt.close(fig)
    bkgmodel = np.zeros_like(data)
    bkgmodel[y0:y1,x0:x1] = skymodel
    fits.PrimaryHDU(data-bkgmodel, image.header).writeto(f.replace('.fits','_bkg.fits'), overwrite=True)

# inflist = glob.glob(join(calPath,'*bkg.jpg'))
# merge_pdf(inflist, join(outPath,'skytilted.pdf'))
# for f in inflist:
#     os.remove(f)

# Careful with header of files
# %%
finalList = glob.glob(join(calPath,'*bkg.fits'))

headers = glob.glob(join(calPath,'*sw.head'))
for h in header:
    os.rename(h, h.replace('sw.head','sw_bkg.head'))


swarptxt = join(calPath,'swarpList.txt')
with open(swarptxt, 'w') as f:
    for file in finalList:
        f.write(file+'\n')

os.system(f'swarp -dd > {swarpFile}')
swarpParams = {'CENTER_TYPE':       'MANUAL',
               'CENTER':            f'{ra0}, {dec0}',
               'PIXELSCALE_TYPE':   'MANUAL',
               'PIXEL_SCALE':       scale,
               'IMAGE_SIZE':        f'{np.int64(width)+2},{np.int64(height)+2}',
               'SUBTRACT_BACK':     'N',
               'MEM_MAX':           '2047',
               'COMBINE_BUFSIZE':   '2047',
               'VMEM_MAX':          '4094'
               }
               
change_config(swarpFile, swarpParams)

os.system(f'swarp @{swarptxt} -c {swarpFile} -IMAGEOUT_NAME {mosaicFile}')

# %%
