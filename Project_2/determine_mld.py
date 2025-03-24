import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import num2date


profile_types = {
    'density_papa': {
        'key': 'STH_71',  # kg/m^3
        'quality_key': 'QST_5071',
        'path': 'Papa station/density_profile_papa.cdf',
        'threshold': 0.03,
        'name': 'Density (kg/m^3)'
    },
    'temperature_papa': {
        'key': 'T_20',
        'quality_key': 'QT_5020',
        'path': 'Papa station/temperature_profile_papa.cdf',
        'threshold': 0.2,
        'name': 'Temperature (C)'
    },
    'density_keo': {
        'key': 'STH_71',  # kg/m^3
        'quality_key': 'QST_5071',
        'path': 'KEO station/density_profile_KEO.cdf',
        'threshold': 0.03,
        'name': 'Density (kg/m^3)'
    },
    'temperature_keo': {
        'key': 'T_20',
        'quality_key': 'QT_5020',
        'path': 'KEO station/temperature_profile_KEO.cdf',
        'threshold': 0.2,
        'name': 'Temperature (C)'
    },
}


acceptable_quality = [1, 2, 3]  # Good quality codes

def interpolate_bad_values(temp, valid_mask):
    """
    Linearly interpolate over bad (False) values in a 1D temperature profile.
    """
    if np.all(~valid_mask):  # all values bad — return NaNs
        return np.full_like(temp, np.nan)
    
    # If more than 3/4 of the values are missing, return NaNs
    if np.sum(~valid_mask) > (len(temp) * 3 / 4):
        return np.full_like(temp, np.nan)

    x = np.arange(temp.shape[0])
    good_x = x[valid_mask]
    good_y = temp[valid_mask]

    return np.interp(x, good_x, good_y)

def stepwise_interpolate_profile_mld_safe(profile, valid_mask):
    """
    Stepwise (nearest-neighbor) interpolation for missing values in a profile.
    Skips profile if surface value is missing.

    Parameters:
        profile (np.ndarray): 1D profile (e.g., temperature or density)
        valid_mask (np.ndarray): Boolean mask of valid values

    Returns:
        np.ndarray: Stepwise-filled profile, or all NaN if surface is missing
    """
    if not valid_mask[0]:
        return np.full_like(profile, np.nan)
    
    # If more than 3/4 of the values are missing, return NaNs
    if np.sum(~valid_mask) > (len(profile) * 3 / 4):
        return np.full_like(profile, np.nan)


    filled = profile.copy()
    n = len(filled)

    for i in range(n):
        if not valid_mask[i]:
            # Search downward first
            down = np.flatnonzero(valid_mask[i+1:])
            up = np.flatnonzero(valid_mask[:i][::-1])

            down_val = filled[i + 1 + down[0]] if down.size > 0 else np.nan
            up_val = filled[i - 1 - up[0]] if up.size > 0 else np.nan

            # Prefer the closer one (or either if only one exists)
            if not np.isnan(up_val) and not np.isnan(down_val):
                filled[i] = up_val if up[0] <= down[0] else down_val
            elif not np.isnan(up_val):
                filled[i] = up_val
            elif not np.isnan(down_val):
                filled[i] = down_val
            else:
                filled[i] = np.nan

    return filled



def filter_profiles(ds, profile_type):
    key = profile_type['key']
    quality_key = profile_type['quality_key']

    # Extract data at (time, depth, lat=0, lon=0)
    profiles = ds[key][:, :, 0, 0]       # (time, depth)
    quality_code = ds[quality_key][:, :, 0, 0]                   # (time, depth)

    valid_mask = np.isin(quality_code, acceptable_quality)

    # Keep only depth levels where at least one valid value exists
    good_columns_mask = np.any(valid_mask, axis=0)
    profiles = profiles[:, good_columns_mask]
    valid_mask = valid_mask[:, good_columns_mask]

    # Interpolate per profile
    interpolated_profiles = []

    for i in range(profiles.shape[0]):
        profile = profiles[i]
        mask = valid_mask[i]

        cleaned_profile = None

        if np.any(mask):  # at least one good value in this profile
            cleaned_profile = stepwise_interpolate_profile_mld_safe(profile, mask)
        else:
            cleaned_profile = np.full_like(profile, np.nan)    
        interpolated_profiles.append(cleaned_profile)


    return np.array(interpolated_profiles)

def plot_profiles(times, profiles, depths, mld_depths, time_units, use_dates=True):
    # Convert times to datetime
    time_calendar = 'standard'
    times_dt = num2date(times, units=time_units, calendar=time_calendar)
    times_num = mdates.date2num(times_dt)

    # Create the heatmap
    plt.figure(figsize=(12, 6))

    times_to_plot = times_num if use_dates else times
    vmin = None
    vmax = None
    mesh = plt.pcolormesh(times_to_plot, depths, profiles.T, shading='nearest', cmap='plasma', vmin=vmin, vmax=vmax)
    line = plt.plot(times_num, mld_depths, 'g-', label='MLD', alpha=0.5)
    plt.gca().invert_yaxis()  # depth increases downward
    plt.colorbar(mesh, label='Value')

    # get axis and format as dates
    ax = plt.gca()
    if use_dates:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.xlabel('Time')
    plt.ylabel('Depth (m)')
    plt.title('Profile Over Time')
    plt.tight_layout()
    plt.show()

def load(path):
    ds = nc.Dataset(path)
    return ds


def calculate_mld_idx(density_profile, threshold):
    """
    Calculate mixed layer depth for one profile.
    density_profile: 1D array of density values for one time step
    depths: 1D array of corresponding depth values
    Returns: MLD (float) or np.nan if not found
    """
    if np.all(np.isnan(density_profile)):
        return np.nan

    surface_density = density_profile[0]  # assuming sorted top to bottom
    delta_rho = np.abs(density_profile - surface_density)

    idx = np.where(delta_rho > threshold)[0]
    if len(idx) == 0:
        return np.nan
    
    return idx[0]


def get_mld(profiles, depths, threshold):
    mld_indices = np.array([
        calculate_mld_idx(density, threshold)
        for density in profiles  # shape: (n_time, n_depth)
    ])

    mld_values = np.array([
        depths[int(idx)] if not np.isnan(idx) else np.nan
        for idx in mld_indices
    ])

    return mld_values, mld_indices

def plot_mld(profile_type_name):
    profile_type = profile_types[profile_type_name]
    ds = load(profile_type['path'])

    depths = ds['depth'][:]
    times = ds['time'][:]

    profiles = filter_profiles(ds, profile_type)
    mld_depths, _ = get_mld(profiles, depths, profile_type['threshold'])

    assert len(times) == len(profiles)
    assert len(times) == len(mld_depths)

    plot_profiles(
        times,
        profiles,
        depths,
        mld_depths,
        ds['time'].units,
        use_dates=True
    )

    return profiles, mld_depths


def plot_density_shape_function(profile_index):
    n_layers = 16

    profile_type = profile_types['density']
    ds = load(profile_type['path'])

    depths = ds['depth'][:]
    times = ds['time'][:]

    profiles = filter_profiles(ds, profile_type)
    mld_depth, mld_indices = get_mld(profiles, depths, profile_type['threshold'])

    profile = profiles[profile_index]
    mld_idx = int(mld_indices[profile_index])

    print(f'MLD Depth at idx {profile_index}: {mld_depth[profile_index]}m')

    profile = profile[:int(mld_idx)]

    # Print the stdev of the profile
    print(f'Standard Deviation: {np.std(profile)}')
    return profile


def mean_stdev_by_season(season):
    profile_type = profile_types['density']
    ds = load(profile_type['path'])

    depths = ds['depth'][:]

    stddevs = []
    values = []

    profiles = filter_profiles(ds, profile_type)
    times = ds['time'][:]
    times_dt = num2date(times, units=ds['time'].units, calendar='standard')

    if season == 'winter':
        months = (12, 1, 2)
    elif season == 'spring':
        months = (3, 4, 5)
    elif season == 'summer':
        months = (6, 7, 8)
    elif season == 'fall':
        months = (9, 10, 11)
    else:
        months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


    mld_depth, mld_indices = get_mld(profiles, depths, profile_type['threshold'])
    for i in range(len(profiles)):
        time = times_dt[i]
        profile = profiles[i]

        if time.month not in months:
            continue

        if np.any(np.isnan(profile)):
            continue

        mld_idx = int(mld_indices[i])
        # profile = profile[:int(mld_idx)]
        values = values + list(profile)
        stddevs.append(np.std(profile))

    return np.mean(values), np.mean(stddevs)




if __name__ == '__main__':
    plot_mld('density')