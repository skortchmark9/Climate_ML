import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import num2date


temperature_key = 'T_20'
quality_key = 'QT_5020'
time_key = 'time'
path = 'temperature_profile_papa.cdf'

acceptable_quality = [1, 2, 3]  # Good quality codes

def interpolate_bad_values(temp, valid_mask):
    """
    Linearly interpolate over bad (False) values in a 1D temperature profile.
    """
    if np.all(~valid_mask):  # all values bad — return NaNs
        return np.full_like(temp, np.nan)

    x = np.arange(temp.shape[0])
    good_x = x[valid_mask]
    good_y = temp[valid_mask]

    return np.interp(x, good_x, good_y)

def filter_temperatures(file_path):
    ds = nc.Dataset(file_path)
    
    # Extract data at (time, depth, lat=0, lon=0)
    temperature_profiles = ds[temperature_key][:, :, 0, 0]       # (time, depth)
    quality_code = ds[quality_key][:, :, 0, 0]                   # (time, depth)
    time_vector = ds[time_key][:]                                # (time,)

    valid_mask = np.isin(quality_code, acceptable_quality)

    # Keep only depth levels where at least one valid value exists
    good_columns_mask = np.any(valid_mask, axis=0)
    temperature_profiles = temperature_profiles[:, good_columns_mask]
    valid_mask = valid_mask[:, good_columns_mask]

    # Interpolate per profile
    interpolated_profiles = []
    valid_times = []

    for i in range(temperature_profiles.shape[0]):
        profile = temperature_profiles[i]
        mask = valid_mask[i]

        if np.any(mask):  # at least one good value in this profile
            interp = interpolate_bad_values(profile, mask)
            interpolated_profiles.append(interp)
            valid_times.append(time_vector[i])

    interpolated_profiles = np.array(interpolated_profiles)
    valid_times = np.array(valid_times)
    return interpolated_profiles, valid_times

def make_edges_from_centers(centers):
    """Compute bin edges from center points (e.g., for pcolormesh)."""
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges

def plot_temperature_heatmap(temperatures, times, depths, time_units, time_calendar='standard'):
    # Convert times to datetime
    times_dt = num2date(times, units=time_units, calendar=time_calendar)
    times_num = mdates.date2num(times_dt)

    # time_edges = make_edges_from_centers(times)
    # depth_edges = make_edges_from_centers(depths)    

    # Create the heatmap
    plt.figure(figsize=(12, 6))

    # invert temperatures
    # temperatures = np.flip(temperatures, axis=1)

    mesh = plt.pcolormesh(times_num, depths, temperatures.T, shading='nearest', cmap='plasma')
    plt.gca().invert_yaxis()  # depth increases downward
    plt.colorbar(mesh, label='Temperature (°C)')

    # get axis and format as dates
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.xlabel('Time')
    plt.ylabel('Depth (m)')
    plt.title('Temperature Profile Over Time')
    plt.tight_layout()
    plt.show()

def load():
    ds = nc.Dataset(path)
    return ds


def determine():
    ds = load()
    depths = ds['depth'][:]
    temperatures, times = filter_temperatures(path)

    plot_temperature_heatmap(
        temperatures,
        times, 
        depths,
        ds['time'].units,
        ds['time'].calendar if 'calendar' in ds['time'].ncattrs() else 'standard'
    )

if __name__ == '__main__':
    determine()