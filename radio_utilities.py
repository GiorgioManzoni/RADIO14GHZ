from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from zoneinfo import ZoneInfo
from astropy.time import Time
import json
import urllib.request
import time
import io


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
        jd_value = x0_fit
        t_utc = Time(jd_value, format="jd", scale="utc")
        t_hk = t_utc.to_datetime(timezone=ZoneInfo("Asia/Hong_Kong"))
        
        print("###### TIME INFO #######")
        print("UTC Time:      ", t_utc.iso)
        print("Hong Kong Time:", t_hk.strftime("%Y-%m-%d %H:%M:%S %Z"))
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


def print_time_peak(time,flux,timezone="HKT"):
    '''Give the time in Julian date and the flux in any units and 
    you will get back the exact time of the peak. 
    You can choose the format to be timezone="UTC","HKT", "JD" '''
    # Find Julian Date of the peak
    jd_value = time[np.argmax(flux)]
    # Parse the Julian Date into an Astropy Time object (in UTC)
    t_utc = Time(jd_value, format="jd", scale="utc")
    # Convert to a Python datetime object with Hong Kong Time zone
    t_hk = t_utc.to_datetime(timezone=ZoneInfo("Asia/Hong_Kong"))
    if timezone=="UTC":
        print("UTC Time:      ", t_utc.iso)
        return t_utc.iso
    elif timezone=="HKT":
        print("Hong Kong Time:", t_hk.strftime("%Y-%m-%d %H:%M:%S %Z"))
        return t_hk.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif timezone=="JD":
        print("Julian Time:   ",jd_value)
        return jd_value
    else:
        print("Choose a timezone between UTC, HKT or JD.")
        return np.nan
    
def get_solar_flux_on_date(yymmdd,verbose=False):
    import pandas as pd
    import requests

    # 1. Define date (Learmonth files use YYMMDD format, e.g., '240724' for July 24, 2024)
    # Note: Learmonth archives daily files formatted as L<YYMMDD>.SRD

    date_str = str(yymmdd)


    url = f"https://sws.bom.gov.au/Category/World%20Data%20Centre/Data%20Display%20and%20Download/Solar%20Radio/station/learmonth/SRD/2026/L{date_str}.SRD"

    # 2. Fetch file directly from Australian BOM FTP/HTTP mirror
    response = requests.get(url)

    if response.status_code == 200:
        # Column names corresponding to the 8 frequencies in Learmonth SRD format
        col_names = [
            "time_ut",
            "f_245MHz",
            "f_410MHz",
            "f_610MHz",
            "flux_1_4GHz",  # 1415 MHz channel
            "f_2695MHz",
            "f_4975MHz",
            "f_8800MHz",
            "f_15400MHz",
        ]

        # Read space-separated values, treating '//////' or non-numeric entries as NaN
        df = pd.read_csv(
            io.StringIO(response.text),
            sep=r"\s+",
            names=col_names,
            na_values=["//////", "------"],
            on_bad_lines="skip",
        )

        # Convert HHMMSS string column into proper timestamp format
        date_prefix = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        df["time_ut_clean"] = df["time_ut"].astype(str).str.zfill(6)
        df["datetime"] = pd.to_datetime(
            date_prefix + " " + df["time_ut_clean"],
            format="%Y-%m-%d %H%M%S",
            errors="coerce",
        )

        # Filter out any rows with invalid timestamps or missing 1.4GHz flux
        df_clean = df.dropna(subset=["datetime", "flux_1_4GHz"]).copy()
    
        # 1. Daytime Average (Filters out nighttime zero/near-zero values)
        daytime_data = df_clean[df_clean["flux_1_4GHz"] > 10]
        daytime_avg = daytime_data["flux_1_4GHz"].mean()

        # 2. Raw 24-Hour Average (All records)
        raw_avg = df_clean["flux_1_4GHz"].mean()
        
        if verbose:
            print(f"Daytime Solar Average (1.4 GHz): {daytime_avg:.2f} SFU")
            print(f"Raw 24-Hour Average (1.4 GHz):   {raw_avg:.2f} SFU")
            print("\n")

        return daytime_avg
    
def get_conversion_K(julian_dates, counts_adu, amp_adu, solar_flux_sfu, aperture_eff=0.60):
    k_B = 1.38e-23  # Boltzmann constant [J/K]
    dish_diameter = 3.0  # Dish diameter in meters


    # Calculate Effective Aperture
    A_geom = np.pi * (dish_diameter / 2) ** 2
    A_e = aperture_eff * A_geom

    # Convert SFU to W/m^2/Hz
    S_flux = solar_flux_sfu * 1e-22

    # Expected Antenna Temperature of the Sun
    T_sun_expected = (S_flux * A_e) / (2 * k_B)

    # Calculate Calibration Factor K
    K = T_sun_expected / amp_adu

    return K