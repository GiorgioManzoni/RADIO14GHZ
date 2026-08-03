#!/usr/bin/env python3
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

fits_file = sys.argv[1] 
right_left = sys.argv[2]
right_left = right_left.upper()+'_POL'

# e.g. fits_file='20260622_SUN_VINCENT.fits'
with fits.open(fits_file) as hdul:
    astro_table = Table(hdul[1].data)
    data = astro_table.to_pandas()
#print(data.columns)


# Define the Gaussian with a baseline offset
def gaussian_with_baseline(x, A, x0, sigma, B):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + B


delta_t_days = data['JD']-data['JD'][0]
delta_t_hours = delta_t_days*24
delta_t_min = delta_t_hours*60
delta_t_sec = delta_t_min*60


x_data = delta_t_min
#y_data = data['RIGHT_POL'].values
y_data = data[right_left].values

#initial guesses 
initial_B = np.min(y_data)                      # Baseline noise level
initial_A = np.max(y_data) - initial_B          # Height of the peak above baseline
initial_x0 = x_data[np.argmax(y_data)]         # X-coordinate of the max value
initial_sigma = (np.max(x_data) - np.min(x_data)) / 6  # Rough initial guess for width

guesses = [initial_A, initial_x0, initial_sigma, initial_B]

# Fit the model to telescope data
popt, pcov = curve_fit(gaussian_with_baseline, x_data, y_data, p0=guesses)
A_fit, x0_fit, sigma_fit, B_fit = popt

# Calculate FWHM in minutes
fwhm = 2.35482 * abs(sigma_fit)

# declination of the Sun on 22nd June 2026 in degrees = 23.44
# average speed = 360/(24*60) = 0.25 deg/min
declination_rad = np.radians(23.44)
# drift formula for the sun
fwhm_degrees = fwhm * 0.25 * np.cos(declination_rad)

print(f"Calculated FWHM: {fwhm_degrees:.4f} degrees")

plt.scatter(x_data, y_data, label='RT1 - '+right_left.capitalize()[:-4]+' Polarization', color='gray', s=5)
#plt.plot(x_data, gaussian_with_baseline(x_data, *popt), color='red', label=f'Gaussian Fit (FWHM={fwhm_degrees:.3f}°)')
plt.plot(x_data, gaussian_with_baseline(x_data, *popt), color='red', label=f'Gaussian Fit \nBeam width (FWHM) = %.3f°'%fwhm_degrees)
plt.xlabel(r'$\Delta t$ [min]')
plt.ylabel('Intensity (counts) [a.u.]')
plt.title('Sun 22/06/2026 14.00 pm HKT')
plt.legend(fontsize='small')
plt.savefig('./image.png')
plt.show()
