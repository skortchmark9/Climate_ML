import os
import requests

import cartopy.crs as ccrs # used for map projection
import matplotlib.pyplot as plt # matplotlib
import cartopy.feature as cfeature # used for map projection
import xarray as xr # x-array
import numpy as np # numpy
import urllib.request # download request
import warnings # to suppress warnings
import gender_guesser.detector as gender # for analyzing the names of hurricanes
from numpy import linalg as LA # to plot the moments (by calculating the eigenvalues)
from sklearn.cluster import k_means # to perform k-means
from collections import Counter # set operations
import urllib

def map_background(label=False, extent=[-100, 0, 0, 60]):
  # A helper function for creating the map background.
  # INPUT:
  # "extent": corresponds to the location information of the showed map.
  # "label": boolean

  # OUTPUT:
  # Matplotlib AXES object

  plt.figure(figsize = (20, 10))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines()
  ax.set_extent(extent)
  ax.gridlines(draw_labels=label) # show labels or not
  LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                      edgecolor='face',
                                      facecolor=cfeature.COLORS['land'],
                                          linewidth=.1)
  OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                       edgecolor='face',
                                       facecolor=cfeature.COLORS['water'], linewidth=.1)
  ax.add_feature(LAND, zorder=0)
  ax.add_feature(OCEAN)
  return ax

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

def filter_tracks_by_landfall_and_date(tks):
    #Filter storm dataset to include only storms which make landfall at some point
    landfall_mask = tks.groupby('storm').map(lambda x: (x.landfall == 0).any())

    # Extract the storm IDs where landfall occurs
    storms_with_landfall = landfall_mask.storm[landfall_mask]

    # Step 2: Filter the dataset to include only these storms
    filtered_tks0 = tks.sel(storm=storms_with_landfall)
        
    #Filter storms to only include storms in the seasons for which we have blackout data
    lower_year = 2014 
    upper_year = 2023 

    def passes_year(storm):
        """Check if the storm occurred in the period of interest."""
        return ((storm.season <= upper_year) & (storm.season >= lower_year)).any()

    # Apply the function to each storm and get a boolean array
    year_mask = filtered_tks0.groupby('storm').map(passes_year)

    # Extract the storm IDs for storms in those seassons
    storms_within_year = year_mask.storm[year_mask]

    # Step 2: Filter the dataset to include only these storms
    filtered_tks = filtered_tks0.sel(storm=storms_within_year)

    return filtered_tks

def filter_tracks_by_us_boundary(tks):
    #Filter all storms to include only those which pass through the CONUS box
    longitude_boundary = -66.9503  #Easternmost point in CONUS (Maine)
    latitude_boundary = 24.5465 #Southernmost point in CONUS (Key West)

    def passes_boundary(storm):
        """Check if the storm passes through the specified geographical boundary at any point."""
        return ((storm.lon <= longitude_boundary) & (storm.lat >= latitude_boundary)).any()

    # Apply the function to each storm and get a boolean array
    boundary_mask = tks.groupby('storm').map(passes_boundary)

    # Extract the storm IDs that pass through the boundary
    storms_within_boundary = boundary_mask.storm[boundary_mask]

    # Step 2: Filter the dataset to include only these storms
    filtered_tks = tks.sel(storm=storms_within_boundary)
    return filtered_tks

#Define a function to return the track for which a storm is making landfall
def get_landfall_lon_lat(storm):
    """Returns the longitude and latitude values where the landfall variable is equal to zero."""
    #storm['landfall'] = storm['landfall'].astype(int)
    # Filter the storm data where landfall is equal to zero
    filtered_storm = storm.where(storm.landfall == 0, drop=True)
    
    # Extract longitude and latitude values
    lon_lst = filtered_storm.lon.values
    lat_lst = filtered_storm.lat.values
    
    # Remove NaN values
    lon_lst = lon_lst[~np.isnan(lon_lst)]
    lat_lst = lat_lst[~np.isnan(lat_lst)]
    
    return lon_lst, lat_lst

#Define a function to get the first and second moments for the landfall portion of a storm's track
def get_landfall_moments(storm):
  # A function to calculate the track moments given a storm
  # OUTPUT:
  # X-centroid, Y-centroid, X_var, Y_var, XY_var

  # Note that:
  # In this case, no weights are set. In other words, all weights are 1.
  # A weight variable would need to be added in order to explore other weights

  lon_lst, lat_lst = get_landfall_lon_lat(storm)
  # If the track only has one point, there is no point in calculating the moments
  if len(lon_lst)<= 1: return None
      
  # M1 (first moment = mean). 
  # No weights applied
  lon_weighted, lat_weighted = np.mean(lon_lst), np.mean(lat_lst)
    
  # M2 (second moment = variance of lat and of lon / covariance of lat to lon
  # No weights applied
  cv = np.ma.cov([lon_lst, lat_lst])
    
  return [lon_weighted, lat_weighted, cv[0, 0], cv[1, 1], cv[0, 1]]

def load_clusters():
    path = './data/storm_landfall_times.csv'
    with open(path) as f:
        data = f.read()

    return data
