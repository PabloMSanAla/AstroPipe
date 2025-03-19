import numpy as np
from astropy.stats import SigmaClip


'''Create Star Mask
    1) Create Star Mask
        1.1) Filter Star Mask (Crow) ¿?
        1.2) Filter Bright (tails) and Faint (center)
    2) Compute Normalization value
        2.1) Fit star to Moffat 
        2.2) Fix center
        2.3) Compute normalization term 35% of light
        2.4) Compute sky ring from 80-90% of light
        2.5) Update catalog with these values
    3) Create PSF
'''

def radial_average2D(array,width=1,method='sigma-clipping'):
    ''' Radial average of a numpy array. The center
    of the radial average is assume to be the center of
    the image. 
    Input:
        :array: ndarray to radial average.
        :width: width of the radial bins. Pixels inside this bin
            would be average. 
        :method: ['mean','median','sigma-clipping']
    Output:
        :radial: ndarray of the radial average result 
    '''
    if method=='mean': aggregation = np.mean
    elif method=='median': aggregation = np.median
    elif method=='sigma-clipping': aggregation = SigmaClip(sigma=2., maxiters=None)
    else: raise ValueError('method not recognized')
    
    x = np.arange(0,array.shape[0])
    y = np.arange(0,array.shape[1])
    X,Y = np.meshgrid(x,y)
    Z = np.sqrt((X-array.shape[0]/2)**2 + (Y-array.shape[1]/2)**2)

    radial = np.zeros_like(array)
    i = width
    while i < Z.max():
        index = np.where((Z>i-width) & (Z<i+width))
        radial[index] = aggregation(array[index])
        if method=='sigma-clipping': radial[index] = np.mean(radial[index])
        i += width  
    return radial


# funtion that radially average an image
def radial_average1D(array):
    # create a grid of the same size as the image
    y, x = np.indices(array.shape)
    # compute the center of the image
    center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])
    # compute the radius of each pixel from the center
    r = np.hypot(x - center[0], y - center[1])
    # compute the average value of all pixels with the same radius
    tbin = np.bincount(r.astype(int).ravel(), array.ravel())
    nr = np.bincount(r.astype(int).ravel())
    radialprofile = tbin / nr
    return radialprofile

