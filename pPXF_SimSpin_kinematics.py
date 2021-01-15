# Writing a series of functions that can be used from SimSpin to use pPXF
# Kate Harborne 08/01/21

import glob
from os import path
from astropy.io import fits
from scipy import ndimage
import numpy as np
from ppxf import ppxf
import ppxf.ppxf_util as util

# Function for generating the list of templates at the same resolution as the observation.
def degrade_templates(template_dir, FWHM_templates, FWHM_observation, velscale_obs, vel_ratio=1):
    """
    Function that combines pPXF "log_rebin" and "gaussian_filter1d" convolution 
    to return a list of template spectra that match the resolution and gridding of
    the observed data. 

    :param template_dir: A character string describing the path to the directory
        containing the raw template files. 
    :param FWHM_templates: A float describing the resolution of the templates.
    :param FWHM_observation: A float describing the resolution of the observation.
        This can be different from the instrumental resolution if the galaxy is at
        a signifcant redshift. 
    :param vescale_obs: A float describing the velocity scale in km/s per pixels of
        the observed spectra.
    :param vel_ratio: A float describing if you would like the velocity scale of 
        the templates to be sampled at a higher rate. 

    :return: An array of templates that have been degraded and rebinned to match
        the observations.

    """

    vazdekis = glob.glob(template_dir) # returns list of all template files contained within directory
    FWHM_tem = FWHM_templates
    FWHM_gal = FWHM_observation
    velscale = velscale_obs

    hdu = fits.open(vazdekis[0]) # opening the first template file in the list
    ssp = hdu[0].data            #  and measuring the wavelength range and SSP 
    h2 = hdu[0].header           #  size for initiallising the template array.
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)]) # Gives the range of the spectra wavelengths
    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/vel_ratio)
    templates = np.empty((sspNew.size, len(vazdekis)))
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in resolutions

    for j, file in enumerate(vazdekis):
        hdu = fits.open(file)
        ssp = hdu[0].data
        ssp = ndimage.gaussian_filter1d(ssp, sigma)
        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/vel_ratio)
        if np.median(sspNew) != 0:
            templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates
        else:
            templates[:, j] = 0.

    return templates, logLam2

def apply_pPXF(spectra_file, fwhm_instrument, redshift, noise, vel_ratio=1, template_dir=None, fwhm_templates=None, templates=None, logLam2=None):
    """
    Function to fit galaxy spectra using pPXF.

    If provided with the template directory and resolution, this function will run the degrade_templates()
    function in order to match the resolution of the observation and the templates. Else, if the degraded
    templates are already provided, this function will just use the templates provided to fit the observed 
    galaxy spectra. 

    :param spectra_file: A string describing the path to the SimSpin spectra file.
    :param fwhm_instrument: A float describing the resolution of the observed spectra.
    :param redshift: A float describing the redshift, z, of the observed galaxy.
    :param noise: A float describing the level of noise within the observed spectra.
    :param vel_ratio: A float describing the sampling rate of the templates relative
        to the observed spectra. Default value is 1.
    
    If wishing to degrade templates to the appropriate resolution, 

    :param template_dir: A string describing the path to the directory in which the template files are 
        located. Default is None.
    :param fwhm_templates: A float describing the resolution of the template spectra. Default 
        is None.

    If you have already degraded the templates to the appropriate resolution for a previous 
    fit, you can provide these variables directly to the function to avoid recalculating the
    comupationally expensive degradation and rebinning:

    :param templates: A matrix containing the rebinned and degraded templates. Default is None.
    :param logLam2: An array describing the wavelength labels of the templates. Default is None.

    """
    hdu = fits.open(spectra_file)
    dim = hdu[0].data.shape
    mid = np.array([round(dim[1]/2), round(dim[2]/2)])
    
    gal_lin = hdu[0].data # pulling in the spectra for each pixel
    h1 = hdu[0].header
    lamRange1 = h1['CRVAL3'] + (np.array([-h1['CRPIX3'], h1['CRPIX3']])*h1['CDELT3']) # wavelength range
    FWHM_gal = fwhm_instrument  # SAMI has an instrumental resolution FWHM of 2.65A.
    
    z = redshift      # Initial estimate of the galaxy redshift
    lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin[:, mid[0], mid[1]]) 

    if not templates or not logLam2:
        assert isinstance(template_dir, str) and isinstance(fwhm_templates, float), 'Please provide the path to the template directory and template resolution' 
        templates, logLam2 = degrade_templates(template_dir, fwhm_templates, FWHM_observation=FWHM_gal, velscale_obs=velscale, vel_ratio=vel_ratio)

    velocity = np.empty([dim[1],dim[2]])
    dispersion = np.empty([dim[1],dim[2]])

    for x in range(0,dim[1]):
            for y in range(0,dim[2]):
                
                if all(np.isnan(gal_lin[:,x,y])):
                    velocity[x,y] = None
                    dispersion[x,y] = None
                    
                else:
                    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin[:,x,y])
                    galaxy = galaxy/np.median(galaxy)    # Normalize spectrum to avoid numerical issues
                    noise  = np.full_like(galaxy, noise) # Assume constant noise per pixel here
                    
                    if velscale_ratio > 1:
                        dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c  # km/s
                    else:
                        dv = (logLam2[0] - logLam1[0])*c  # km/s

                    start = [100, 200.]  # (km/s), starting guess for [V, sigma]

                    pp = ppxf.ppxf(templates, galaxy, noise, velscale, start,
                                plot=False, moments=2, quiet=True,
                                degree=4, vsyst=dv, velscale_ratio = vel_ratio)
                                
                    velocity[x,y] = pp.sol[0]
                    dispersion[x,y] = pp.sol[1]

    return velocity, dispersion
