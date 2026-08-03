from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss_fit(x,y, verbose=False, plot=False):
    '''
    given the data in x and y, it try to fit a gaussian and return the parameters of the fit, in order:
    1) amplitude A_fit 
    2) center x0_fit (as in x-x0)
    3) standard deviation sigma_fit
    4) floor of the gaussian B_fit
    5) FWHM = 2.35482 * standard deviation
    '''
    
    def gaussian_with_baseline(x, A, x0, sigma, B):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + B

    #initial guesses 
    initial_B = np.min(y)                      # Baseline noise level
    initial_A = np.max(y) - initial_B          # Height of the peak above baseline
    initial_x0 = x[np.argmax(y)]         # X-coordinate of the max value
    initial_sigma = (np.max(x) - np.min(x)) / 6  # Rough initial guess for width

    x_offset = np.min(x)
    x_centered = x - x_offset
    
    initial_x0_centered = initial_x0 - x_offset
    
    
    guesses = [initial_A, initial_x0_centered, initial_sigma, initial_B]
    
    # Fit the model to telescope data
    popt, pcov = curve_fit(gaussian_with_baseline, x_centered, y, p0=guesses)
    A_fit, x0_fit_centered, sigma_fit, B_fit = popt

    x0_fit = x0_fit_centered + x_offset
    
    # Calculate FWHM in minutes
    fwhm = 2.35482 * abs(sigma_fit)
    
    if verbose:
        print('### Gaussian Fit ###')
        print('Amplitude',A_fit)
        print('Centre [Julian date]',x0_fit)
        print('Sigma [Days]',sigma_fit)
        print('Sigma [min]',sigma_fit*24*60)
        print('Background floor [adu]',B_fit)
        print('FWHM [days]',fwhm)
        print('FWHM [min]',fwhm*24*60)
        print('Estimated antenna beam with 0.25 deg/min',0.25*fwhm*24*60)
                
    if plot:
        plt.scatter(x, y, label='data', color='gray', s=5)
        plt.plot(x, gaussian_with_baseline(x, *[A_fit, x0_fit, sigma_fit, B_fit]), color='red', label=f'Gaussian Fit')
        plt.xlabel(r'JD [day]')
        plt.ylabel('Counts [a.u.]')
        plt.legend(fontsize='small')
        plt.show()
        
    
    return A_fit, x0_fit, sigma_fit, B_fit, fwhm

    
def get_apparent_speed_sun(julian_time, lat=22.3364, long=114.2636, height=150):
    '''compute the apparent speed of the sun in the sky in degrees per minutes.
       input is the julian date which units is days and the decimal are the fraction of the day
       and latitude, longitude (degrees) and height (meters) of the observing place (the default are Hong Kong coordinates)
    
    '''
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time, TimeDelta
    
    # Define telescope location
    loc = EarthLocation(
    lat=lat * u.deg, lon=long * u.deg, height=height * u.m
    )
    
    # computes two times as delta t. The inputted julian time and 1 second after.
    t1 = Time.now()
    t2 = t1 + TimeDelta(1.0, format="sec")
    
    # get the altitude and azimuth of the Sun in the two instants
    f1, f2 = AltAz(obstime=t1, location=loc), AltAz(obstime=t2, location=loc)
    s1, s2 = get_sun(t1).transform_to(f1), get_sun(t2).transform_to(f2)
    
    # given the two position of the sun, compute the angular separation and hence the angular speed dividing by 1 second
    d_alt = (s2.alt.deg - s1.alt.deg) / 1.0  # deg/s
    d_az = ((s2.az.deg - s1.az.deg + 180) % 360 - 180) / 1.0  # deg/s
    speed_deg_per_min = (
    np.sqrt(d_alt**2 + (np.cos(s1.alt.rad) * d_az) ** 2) * 60.0
    )
    
    return speed_deg_per_min


def get_beam_angle(times_jd,counts_adu):
    Amp, centre_jd, std, floor, FWHM_jd = gauss_fit(times_jd,counts_adu)
    FWHM_min = (FWHM_jd*(24*60))
    speed_sun = get_apparent_speed_sun(centre_jd)
    FWHM_deg =  FWHM_min * speed_sun
    return FWHM_deg