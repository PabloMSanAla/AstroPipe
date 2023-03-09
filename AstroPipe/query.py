import wget
import os 

def s4g_images(name, channel=1,mask=False, outdir='',verbose=False):
    """
    Download the S4G images from the irsa website given an object name and channel number.
    """
    url = f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P1/{name}.phot.{channel}.fits'
    if verbose: print(url)

    filename = wget.download(url,out=outdir)
    if verbose: print(filename)

    if mask:
        url = f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P2/{name}.{channel}.final_mask.fits'
        if verbose: print(url)

        filename = wget.download(url,out=outdir)
        if verbose: print(filename)

    return os.path.isfile(filename)

 
def run_bash(bashfile,verbose=False):
    pro = sp.run(bashfile)
    if verbose: print(pro.returncode)
    return pro.returncode


    # Write a function that given right ascension and declination coordinates downloads the image from the SDSS server in a fits format and returns the image as a numpy array
def sdss_mosaic(ra, dec, name, outdir=None, size=0.3, scale=0.396, band='i', verbose=False):
    """
    Download an image from the SDSS server
    """

    url = f'https://dr12.sdss.org/mosaics/script?onlyprimary=True&pixelscale={scale}&ra={ra}&filters={band}&dec={dec}&size={size}'
    if verbose: print(url)

    directory = os.path.join(outdir,name)
    bashfile = os.path.join(directory,name+'.sh')
    
    if not os.path.isdir(directory): os.mkdir(directory)

    script = sdss_script_mosaic(url)
    issaved = save_script(script,bashfile)
    
    if issaved:
        if verbose: print(f'running for {name}')
        originalpath = os.getcwd()
        os.chdir(directory)
        outcmd = run_bash(bashfile,verbose=verbose)
        os.system(f'rm {os.path.join(directory,"frame*.fits*")}')
        os.system(f'rm {os.path.join(directory,"*.swarp")}')
        for f in glob.glob('*.fits'):
            os.system(f'mv {f} {name}-{band}.{f.split(f"-{band}.")[-1]}')
        os.chdir(originalpath)
        return outcmd==0
    else: 
        return False


def sdss_mosaic_list(row, size=0.3, scale=0.396, band='i', verbose=False):
    """
    Download an image from the SDSS server from an input list
    """
    ra = row[0]
    dec = row[1]
    name = row[2]
    outdir = row[3]

    return sdss_mosaic(ra, dec, name, outdir=outdir)
