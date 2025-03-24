import netCDF4 as nc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os
import xarray as xr

raw_paths = [
    'data/raw/ows_papa_2011_2013.nc',
    'data/raw/ows_papa_2013_2015.nc',
    'data/raw/ows_papa_2015_2017.nc',
    'data/raw/ows_papa_2017_2019.nc',
    'data/raw/ows_papa_2019_2021.nc',
    'data/raw/ows_papa_2021_2023.nc',
    'data/raw/ows_papa_2023_2024.nc',
]

def load_data():
    """
    Load raw data from all netCDF files.
    """
    time = []
    density = []
    diffusivity = []
    mld_depth = []

    for path in raw_paths:
        with nc.Dataset(path) as ds:
            num2date = nc.num2date(ds['time'][1:], ds['time'].units)
            time.append(num2date)

            density_key = 'rho'
            density.append(ds[density_key][1:, -300:, 0, 0])

            diffusivity_key = 'nuh'
            diffusivity.append(ds[diffusivity_key][1:, -300:, 0, 0])

            mld_depth.append(ds['mld_surf'][1:, 0, 0])

    # Combine all data into a new nc file.
    data = {
        'time': np.concatenate(time),
        'density': np.concatenate(density),
        'diffusivity': np.concatenate(diffusivity),
        'mld_depth': np.concatenate(mld_depth),
    }
    depth = np.array(list(range(-300, 0)))

    ds = xr.Dataset(
        {
            'density': (['time', 'depth'], data['density']),
            'diffusivity': (['time', 'depth'], data['diffusivity']),
            'mld_depth': (['time'], data['mld_depth']),
        },
        coords={
            'time': data['time'],
            'depth': depth,
        }
    )

    ds.to_netcdf('data/processed/ows_papa.nc')


    return ds