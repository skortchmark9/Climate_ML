import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import num2date


profile_types = {
    "density_papa": {
        "key": "STH_71",  # kg/m^3
        "quality_key": "QST_5071",
        "path": "data/papa_observational_data/density_profile_papa.cdf",
        "threshold": 0.03,
        "name": "Density (kg/m^3)",
    },
    "temperature_papa": {
        "key": "T_20",
        "quality_key": "QT_5020",
        "path": "data/papa_observational_data/temperature_profile_papa.cdf",
        "threshold": 0.2,
        "name": "Temperature (C)",
    },
}


acceptable_quality = [1, 2, 3]  # Good quality codes


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
            down = np.flatnonzero(valid_mask[i + 1 :])
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
    key = profile_type["key"]
    quality_key = profile_type["quality_key"]

    # Extract data at (time, depth, lat=0, lon=0)
    profiles = ds[key][:, :, 0, 0]  # (time, depth)
    quality_code = ds[quality_key][:, :, 0, 0]  # (time, depth)

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
    time_calendar = "standard"
    times_dt = num2date(times, units=time_units, calendar=time_calendar)
    times_num = mdates.date2num(times_dt)

    # Create the heatmap
    plt.figure(figsize=(12, 6))

    times_to_plot = times_num if use_dates else times
    vmin = None
    vmax = None
    mesh = plt.pcolormesh(
        times_to_plot,
        depths,
        profiles.T,
        shading="nearest",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
    )
    line = plt.plot(times_num, mld_depths, "g-", label="MLD", alpha=0.5)
    plt.gca().invert_yaxis()  # depth increases downward
    plt.colorbar(mesh, label="Value")

    # get axis and format as dates
    ax = plt.gca()
    if use_dates:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
    plt.xlabel("Time")
    plt.ylabel("Depth (m)")
    plt.title("Profile Over Time")
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


def get_mld_from_threshold(profiles, depths, threshold):
    mld_indices = np.array(
        [
            calculate_mld_idx(density, threshold)
            for density in profiles  # shape: (n_time, n_depth)
        ]
    )

    mld_values = np.array(
        [depths[int(idx)] if not np.isnan(idx) else np.nan for idx in mld_indices]
    )

    return mld_values, mld_indices


def plot_mld(profile_type_name):
    profile_type = profile_types[profile_type_name]
    ds = load(profile_type["path"])

    depths = ds["depth"][:]
    times = ds["time"][:]

    profiles = filter_profiles(ds, profile_type)
    mld_depths, _ = get_mld_from_threshold(profiles, depths, profile_type["threshold"])

    assert len(times) == len(profiles)
    assert len(times) == len(mld_depths)

    plot_profiles(times, profiles, depths, mld_depths, ds["time"].units, use_dates=True)

    return profiles, mld_depths


def plot_mld_comparison(mld_density, mld_temperature):
    """Make a line plot showing the density-based MLD and the temperature-based MLD as a function of time"""
    time_calendar = "standard"

    # Choose station (e.g., 'papa') and get both profile types
    density_type = profile_types["density_papa"]

    ds_sim = load("./data/processed/ows_papa.nc")
    sim_mld = ds_sim["mld_depth"][:]
    sim_time = ds_sim["time"][:]
    sim_time_units = ds_sim["time"].units
    sim_times_dt = num2date(sim_time, units=sim_time_units, calendar=time_calendar)
    sim_times_num = mdates.date2num(sim_times_dt)

    # Load datasets
    ds_density = load(density_type["path"])

    # Extract times and depths
    time_slice = min(len(mld_density), len(mld_temperature))
    mld_density = mld_density[:time_slice]
    mld_temperature = mld_temperature[:time_slice]

    times = ds_density["time"][:time_slice]  # assuming time arrays are the same
    time_units = ds_density["time"].units

    # Convert time to datetime
    times_dt = num2date(times, units=time_units, calendar=time_calendar)
    times_num = mdates.date2num(times_dt)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        sim_times_num, sim_mld, label="GOTM-reported MLD", color="green", alpha=0.2
    )
    plt.plot(times_num, mld_density, label="Density-based MLD", color="blue", alpha=0.7)
    plt.plot(
        times_num,
        mld_temperature,
        label="Temperature-based MLD",
        color="red",
        alpha=0.7,
    )

    plt.gca().invert_yaxis()  # because depth increases downward
    # get axis and format as dates
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))

    plt.xlabel("Time")
    plt.ylabel("Mixed Layer Depth (m)")
    plt.title("MLD Comparison (Papa Station)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mld_comparison_by_year(mld_density, mld_temperature):
    """Plot MLD comparison in subplots by year"""

    # Choose station
    density_type = profile_types["density_papa"]
    ds_density = load(density_type["path"])

    # Truncate MLD arrays to same length
    time_slice = min(len(mld_density), len(mld_temperature))
    mld_density = mld_density[:time_slice]
    mld_temperature = mld_temperature[:time_slice]

    # Load time info
    times = ds_density["time"][:time_slice]
    time_units = ds_density["time"].units
    times_dt = num2date(times, units=time_units, calendar="standard")

    # Group by year
    years = np.array([t.year for t in times_dt])
    unique_years = sorted(set(years))

    incomplete_years = (2007, 2008, 2009, 2024, 2023)
    unique_years = [year for year in unique_years if year not in incomplete_years]

    n_years = len(unique_years)
    fig, axes = plt.subplots(n_years, 1, figsize=(12, 3.5 * n_years), sharey=True)

    if n_years == 1:
        axes = [axes]  # ensure iterable

    for i, year in enumerate(unique_years):
        ax = axes[i]
        year_mask = years == year
        year_times = mdates.date2num(np.array(times_dt)[year_mask])
        ax.plot(
            year_times,
            np.array(mld_density)[year_mask],
            label="Density-based MLD",
            color="blue",
            alpha=0.7,
        )
        ax.plot(
            year_times,
            np.array(mld_temperature)[year_mask],
            label="Temperature-based MLD",
            color="red",
            alpha=0.7,
        )
        ax.invert_yaxis()
        ax.set_title(f"MLD Comparison ({year})")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # e.g., Jan, Feb
        ax.grid(True)
        if i == n_years - 1:
            ax.set_xlabel("Date")
        ax.set_ylabel("MLD (m)")
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()


def mean_stdev_by_season(season):
    profile_type = profile_types["density"]
    ds = load(profile_type["path"])

    depths = ds["depth"][:]

    stddevs = []
    values = []

    profiles = filter_profiles(ds, profile_type)
    times = ds["time"][:]
    times_dt = num2date(times, units=ds["time"].units, calendar="standard")

    if season == "winter":
        months = (12, 1, 2)
    elif season == "spring":
        months = (3, 4, 5)
    elif season == "summer":
        months = (6, 7, 8)
    elif season == "fall":
        months = (9, 10, 11)
    else:
        months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    mld_depth, mld_indices = get_mld_from_threshold(profiles, depths, profile_type["threshold"])
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


if __name__ == "__main__":
    plot_mld("density")
