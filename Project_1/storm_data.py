import os
import requests
import pandas as pd
import datetime
from datetime import datetime, timedelta
import xarray as xr # x-array
import numpy as np # numpy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cftime import num2date

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
    initial_day = datetime.strptime(initial_day, '%Y-%m-%d %H:%M:%S')
    new_datetime = initial_day + timedelta(days=storm_time)
    return datetime(
        year=new_datetime.year,
        month=new_datetime.month,
        day=new_datetime.day,
        hour=new_datetime.hour,
    )

def datetime_to_storm_time(datetime_):
    initial_day = datetime.strptime('1858-11-17 00:00:00', '%Y-%m-%d %H:%M:%S')

    delta = (datetime_ - initial_day)
    return delta.days + delta.seconds / 3600 / 24

def date_str_to_storm_time(date_str):
    initial_day = datetime.strptime('1858-11-17 00:00:00', '%Y-%m-%d %H:%M:%S')
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
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

state_dict = {'Alabama':'AL', 'Arkansas': 'AR', 'Connecticut': 'CT',
              'Delaware': 'DE', 'Florida': 'FL', 'Kentucky': 'KY',
              'Louisiana':'LA', 'Maine':'ME', 'Maryland':'MD',
              'Massachusetts':'MA', 'Mississippi':'MS',
              'Missouri':'MO', 'New Hampshire':'NH', 'New Jersey': 'NJ',
              'New York':'NY', 'North Carolina':'NC', 'Oklahoma':'OK',
              'Rhode Island':'RI', 'South Carolina':'SC', 'Tennessee':'TN',
              'Texas': 'TX', 'Virginia':'VA', 'West Virginia':'WV',
              'Georgia':'GA'}

def get_intensity_time(storm): # return intensity at every timestep rather than max per storm
    wind_speed = storm.wmo_wind
    if (len(wind_speed) == 0):
        return -1
    return wind_speed

def get_storm_data(storms_xr, storm_name):
    storms_xr['name'] = storms_xr['name'].astype(str)
    storm = storms_xr.where(storms_xr['name'] == storm_name.upper(), drop=True)
    storm = storm.to_dataframe().reset_index()

    return storm

def get_timestamps(storms_xr, storm_name):
    initial_day = datetime(1858, 11, 17)
    storm = get_storm_data(storms_xr, storm_name)
    storm_times = storm.time

    timestamps, dates, hours = [], [], []

    for time in storm_times:
        if np.isnan(time):
            timestamps.append(np.nan)
            dates.append(np.nan)
            hours.append(np.nan)
        else:
            x = initial_day + timedelta(days=time)
            timestamps.append(pd.Timestamp(x))
            dates.append(x.date())
            hours.append(x.hour)

    return timestamps, dates, hours

def outages_bystate(df, year):
    data = df[year].drop(columns='hour')
    data = data.groupby(by=['date', 'fips_code', 'county', 'state']).max().reset_index()
    data = data.drop(columns=['fips_code', 'county'])
    data = data.groupby(by=['state', 'date']).sum().reset_index()
    data['state_abbr'] = data['state'].map(state_dict)
    return data

def storm_df(storms_xr, storm_name):

    storm_choice = get_storm_data(storms_xr, storm_name)
    
    storm_dict = {'lat': storm_choice.lat, 
                  'lon':storm_choice.lon,
                  'date': get_timestamps(storms_xr, storm_name)[1], 
                  'intensity': get_intensity_time(storm_choice)}

    storm_df = pd.DataFrame(storm_dict)
    storm_df = storm_df.dropna()

    # max lat lon and intensity each date
    pivoted_storm = storm_df.groupby(by= 'date').max().reset_index()

    return pivoted_storm


def filter_data(data, lower_year, upper_year, longitude_boundary, latitude_boundary):
    # Filter for storms that actually make landfall
    landfall_mask = data.groupby('storm').map(lambda x: (x.landfall == 0).any())
    storms_with_landfall = landfall_mask.storm[landfall_mask]
    data_landfall = data.sel(storm=storms_with_landfall)
    
    # Filter for years that have blackout data
    def passes_year(storm):
        return ((storm.season <= upper_year) & (storm.season >= lower_year)).any()
    
    year_mask = data_landfall.groupby('storm').map(passes_year)
    storms_within_year = year_mask.storm[year_mask]
    data_landfall_year = data_landfall.sel(storm=storms_within_year)
    
    # Filter for storms within the geographical boundaries of the continental US
    def passes_boundary(storm):
        return ((storm.lon <= longitude_boundary) & (storm.lat >= latitude_boundary)).any()
    
    boundary_mask = data_landfall_year.groupby('storm').map(passes_boundary)
    storms_within_boundary = boundary_mask.storm[boundary_mask]
    data_landfall_year_US = data_landfall_year.sel(storm=storms_within_boundary)
    
    return data_landfall_year_US

def get_landfall_lon_lat(storm):
    """Returns the longitude and latitude values where the landfall variable is equal to zero."""
    # Filter the storm data where landfall is equal to zero
    filtered_storm = storm.where(storm.landfall == 0, drop=True)
    
    lon_lst = filtered_storm.lon.values
    lat_lst = filtered_storm.lat.values
    
    lon_lst = lon_lst[~np.isnan(lon_lst)]
    lat_lst = lat_lst[~np.isnan(lat_lst)]
    
    return lon_lst, lat_lst

def get_landfall_moments(storm):
  lon_lst, lat_lst = get_landfall_lon_lat(storm)
  # If the track only has one point, there is no point in calculating the moments
  if len(lon_lst)<= 1: return None
      
  # M1 (first moment = mean). 
  lon_weighted, lat_weighted = np.mean(lon_lst), np.mean(lat_lst)
    
  # M2 (second moment = variance of lat and of lon / covariance of lat to lon
  cv = np.ma.cov([lon_lst, lat_lst])
    
  return [lon_weighted, lat_weighted, cv[0, 0], cv[1, 1], cv[0, 1]]

def process_moment_landfall(data, longitude_boundary, latitude_boundary):
    moment_landfall_array = np.array(data)
    filtered_moment_landfall = moment_landfall_array[
        (moment_landfall_array[:, 0] <= longitude_boundary) & 
        (moment_landfall_array[:, 1] >= latitude_boundary)
    ]
    return filtered_moment_landfall[:, :2]

def map_background(label=False, extent=[-100, 0, 0, 60]):

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

def plot_landfall_clusters(moments, clusters, colors=None):
    if colors is None:
        colors = ['black', 'red', 'blue', 'yellow', 'green', 'magenta', 'orange']
    
    labels = clusters[1]
    ax = map_background()

    for k in range(len(moments)):
        ax.plot(moments[k][0], moments[k][1], c=colors[labels[k]], marker='*')

    # Create custom legend handles
    legend_handles = [mpatches.Patch(color=colors[i], label=f'Cluster {i}') for i in range(len(colors))]

    # Add the legend to the plot
    ax.legend(handles=legend_handles, title='Clusters')

    plt.title('K-means clustering result, 7 clusters')
    plt.show()

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

def is_valid_landfall_storm(storm):
    lon_lst, lat_lst = get_landfall_lon_lat(storm)
    return len(lon_lst) > 1   # Keep only storms with more than one point

def plot_bar_with_error(data_to_plot, errors, title, colors = None):
    if colors is None:
        colors = ['black', 'red', 'blue', 'yellow', 'green', 'magenta', 'orange']
    plt.figure(figsize=(10, 6))
    num_bars = len(data_to_plot)
    bar_colors = [colors[i % len(colors)] for i in range(num_bars)]
    plt.bar(range(num_bars), data_to_plot, yerr=errors, capsize=5, color=bar_colors, edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Storm Speed (Kts)')
    plt.title(title)
    plt.xticks(range(num_bars), labels=range(num_bars))
    plt.show()

def get_landfall_times(storm):
    landfall_times = storm.time.where(storm.landfall == 0, drop=True)
    if landfall_times.size > 0:
        first_landfall_time = landfall_times[0].item()
        final_landfall_time = landfall_times[-1].item()
        return first_landfall_time, final_landfall_time
    else:
        return None, None

def convert_time(numeric_time, units, calendar):
    return num2date(numeric_time, units, calendar).strftime('%Y-%m-%d %H:%M:%S')

def combine_data_and_kmeanslabels(data_filtered, moment_landfall, km_landfall, longitude_boundary, latitude_boundary):
    valid_storms = [i for i in range(data_filtered.dims['storm']) if is_valid_landfall_storm(data_filtered.sel(storm=i))]
    data_filtered_new = data_filtered.sel(storm=xr.DataArray(valid_storms, dims="storm"))
    moment_landfall_array = np.array(moment_landfall)
    moment_lon = moment_landfall_array[:, 0]
    moment_lat = moment_landfall_array[:, 1]
    temp_array  = data_filtered_new.assign_coords(moment_lat=('storm', moment_lat))
    full_array  = temp_array.assign_coords(moment_lon=('storm', moment_lon))
    conus_mask = (full_array.moment_lon <= longitude_boundary) & (full_array.moment_lat >= latitude_boundary)
    storms_within_boundary = full_array.storm[conus_mask]
    full_array_filtered = full_array.sel(storm=storms_within_boundary)
    full_array_labeled = full_array_filtered.assign_coords(spatmoment_label=('storm', km_landfall[1]))
    grouped_spat = full_array_labeled.groupby('spatmoment_label')
    summary_spat = grouped_spat.mean(dim='storm')
    time_units = full_array_labeled.time.attrs['units']
    time_calendar = full_array_labeled.time.attrs.get('calendar', 'standard')
    results = []
    for storm_id in full_array_labeled.storm.values:
        storm = full_array_labeled.sel(storm=storm_id)
        first_landfall_time, final_landfall_time = get_landfall_times(storm)
        if first_landfall_time is not None:
            results.append({
                'storm': storm_id,
                'sid': storm.sid.item().decode('utf-8'),  # Decode the byte string to a regular string
                'name': storm.name.item().decode('utf-8'),  # Decode the byte string to a regular string
                'spatmoment_label': storm.spatmoment_label.item(),
                'first_landfall_time': convert_time(first_landfall_time, time_units, time_calendar),
                'final_landfall_time': convert_time(final_landfall_time, time_units, time_calendar)
            })
    results_df = pd.DataFrame(results)
    # results_df.to_csv('data/storm_landfall_times.csv', index=False)
    return summary_spat
