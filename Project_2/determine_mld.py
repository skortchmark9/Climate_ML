import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import num2date


temperature_key = 'T_20'
temperature_quality_key = 'QT_5020'
temperature_path = 'temperature_profile_papa.cdf'

density_key = 'STH_71' # kg/m^3
density_quality_key = 'QST_5071'
density_path = 'density_profile_papa.cdf'
time_key = 'time'


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





def filter_profiles(ds, key, quality_key):
    # Extract data at (time, depth, lat=0, lon=0)
    profiles = ds[key][:, :, 0, 0]       # (time, depth)
    quality_code = ds[quality_key][:, :, 0, 0]                   # (time, depth)
    time_vector = ds[time_key][:]                                # (time,)

    valid_mask = np.isin(quality_code, acceptable_quality)

    # Keep only depth levels where at least one valid value exists
    good_columns_mask = np.any(valid_mask, axis=0)
    profiles = profiles[:, good_columns_mask]
    valid_mask = valid_mask[:, good_columns_mask]

    # Interpolate per profile
    interpolated_profiles = []
    valid_times = []

    for i in range(profiles.shape[0]):
        profile = profiles[i]
        mask = valid_mask[i]

        if np.any(mask):  # at least one good value in this profile
            interp = stepwise_interpolate_profile_mld_safe(profile, mask)
            interpolated_profiles.append(interp)
            valid_times.append(time_vector[i])

    interpolated_profiles = np.array(interpolated_profiles)
    valid_times = np.array(valid_times)
    return interpolated_profiles, valid_times

def plot_profiles(profiles, times, depths, time_units, use_dates=True):
    # Convert times to datetime
    time_calendar = 'standard'
    times_dt = num2date(times, units=time_units, calendar=time_calendar)
    times_num = mdates.date2num(times_dt)

    mld_depths = get_mld(profiles, depths)

    # Create the heatmap
    plt.figure(figsize=(12, 6))

    times_to_plot = times_num if use_dates else times

    mesh = plt.pcolormesh(times_to_plot, depths, profiles.T, shading='nearest', cmap='plasma')
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


def calculate_mld(density_profile, depths, threshold):
    """
    Calculate mixed layer depth for one profile.
    density_profile: 1D array of density values for one time step
    depths: 1D array of corresponding depth values
    Returns: MLD (float) or np.nan if not found
    """
    if np.all(np.isnan(density_profile)):
        return np.nan

    surface_density = density_profile[0]  # assuming sorted top to bottom
    delta_rho = density_profile - surface_density

    idx = np.where(delta_rho > threshold)[0]
    if len(idx) == 0:
        return np.nan

    return depths[idx[0]]


def get_mld(profiles, depths, threshold=0.03):
    mld_values = np.array([
        calculate_mld(density, depths, threshold)
        for density in profiles  # shape: (n_time, n_depth)
    ])
    return mld_values


def determine_temp():
    ds = load(temperature_path)
    times, profiles = determine(ds, temperature_key, temperature_quality_key)
    return times, profiles

def determine_density():
    ds = load(density_path)
    times, profiles = determine(ds, density_key, density_quality_key)
    return times, profiles


def determine(ds, key, quality_key):
    depths = ds['depth'][:]
    profiles, times = filter_profiles(ds, key, quality_key)

    plot_profiles(
        profiles,
        times,
        depths,
        ds['time'].units,
        use_dates=True
    )

    return times, profiles

if __name__ == '__main__':
    determine()