import os
import requests
import pandas as pd
import datetime
import xarray as xr # x-array
import numpy as np # numpy

def load_tracks():
    path = 'data/NA_data.nc'
    if os.path.exists(path):
        tks = xr.open_dataset(path, engine="netcdf4", decode_times=False)
        return tks

    # IBTrACS.NA.v04r00.nc presents data from 1842-10-25 through 2023-06-07 
    url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.NA.v04r00.nc'

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        with open(path, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    tks = xr.open_dataset('data/NA_data.nc', engine="netcdf4", decode_times=False)
    return tks


def load_clusters():
    data = pd.read_csv('data/storm_landfall_times.csv')
    return data.groupby('spatmoment_label')


def storm_time_to_datetime(storm_time):
    initial_day = '1858-11-17 00:00:00'
    initial_day = datetime.datetime.strptime(initial_day, '%Y-%m-%d %H:%M:%S')
    new_datetime = initial_day + datetime.timedelta(days=storm_time)
    return datetime.datetime(
        year=new_datetime.year,
        month=new_datetime.month,
        day=new_datetime.day,
        hour=new_datetime.hour,
    )

def datetime_to_storm_time(datetime_):
    initial_day = datetime.datetime.strptime('1858-11-17 00:00:00', '%Y-%m-%d %H:%M:%S')

    delta = (datetime_ - initial_day)
    return delta.days + delta.seconds / 3600 / 24

def date_str_to_storm_time(date_str):
    initial_day = datetime.datetime.strptime('1858-11-17 00:00:00', '%Y-%m-%d %H:%M:%S')
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    delta = (date - initial_day)
    return delta.days + delta.seconds / 3600 / 24

def get_intensity(storm):
    #wmo_wind is the max sustained wind speed 
    wind_speed = storm.wmo_wind.values
    #filter for over land
    #landfall = storm.landfall.values
    #wind_speed = wind_speed[~landfall]
    wind_speed = wind_speed[~np.isnan(wind_speed)]
    #wind_spped = np.average(wind_speed)
    #wind_speed = np.max(wind_speed)

    if (len(wind_speed) == 0):
        return -1

    return np.max(wind_speed)
