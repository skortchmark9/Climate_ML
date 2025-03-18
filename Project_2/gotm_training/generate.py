import os
import jinja2
import shutil
import glob
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import netCDF4 as nc
import numpy as np
import xarray as xr
import numpy.ma as ma


training_path = '/Users/samuelkortchmar/Documents/Columbia/ClimateML/gotm_training/cases/case_25_-600_1/output.nc'
forcings = {
    'lats': [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    'heat_fluxes': [
        -1800, -1600, -1400, -1200, -1000, -800, -600, -400, -300, -200, -150, -100, -90, -80, -75, -70, -60, -50, -40, -30, -25, -20, -10, 0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800
    ],
    'tx_stresses': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
}

forcings = {
    'lats': [10, 15, 20, 25],
    'heat_fluxes': [-1000, -800, -600, -400, -300, -200, -150, -100, -90, -80, -75, -70, -60, -50, -40, -30],
    'tx_stresses': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}


def generate(lat, heat_flux, tx_stress):
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "base/base.yaml.j2"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render({
        'latitude': lat,
        'heat_flux': heat_flux,
        'tx_stress': tx_stress,
    })
    return outputText

def make_case_path(lat, heat_flux, tx_stress):
    return f'cases/case_{lat}_{heat_flux}_{tx_stress}'

def sweep():
    for lat in forcings['lats']:
        for heat_flux in forcings['heat_fluxes']:
            for tx_stress in forcings['tx_stresses']:
                yield lat, heat_flux, tx_stress

def copy_dir(lat, heat_flux, tx_stress):
    os.makedirs('cases', exist_ok=True)
    case_path = make_case_path(lat, heat_flux, tx_stress)
    base_dir = 'base'
    shutil.copytree(base_dir, case_path)

    with open(case_path + '/case.yaml', 'w') as f:
        f.write(generate(lat, heat_flux, tx_stress))


def generate_case_dirs():
    print('generating cases')
    n = 0
    lats = forcings['lats']
    heat_fluxes = forcings['heat_fluxes']
    tx_stresses = forcings['tx_stresses']

    max = len(lats) * len(heat_fluxes) * len(tx_stresses)
    for lat, heat_flux, tx_stress in sweep():
        n += 1
        copy_dir(lat, heat_flux, tx_stress)
        if (n % 100) == 0:
            print(f'{n}/{max} cases generated')


def run_gotm_case(case):
    if os.path.exists('cases/' + case + '/output.nc'):
        return
    gotm_cmd = '../../gotm case.yaml > log.txt 2>&1'
    os.chdir('cases' + '/' + case)
    # Check if the command was successful
    result = os.system(gotm_cmd)

    os.chdir('../..')

def run_gotm():
    print('running cases...')
    cases = os.listdir('cases')
    with Pool() as pool:
        for _ in tqdm(pool.imap(run_gotm_case, cases), total=len(cases)):
            pass

def do_preprocess():
    """Find all output.nc files, extract data, and create a merged xarray dataset."""
    
    all_data = []

    for lat, heat_flux, tx_stress in sweep():
        case_path = make_case_path(lat, heat_flux, tx_stress) + "/output.nc"
        
        if not os.path.exists(case_path):
            continue  # Skip if the file doesn't exist
        
        ds = nc.Dataset(case_path)
        data = extract_info(ds)
        
        num_times = len(data['time'])
        num_layers = len(data['sf'][0])  # Assuming all cases have the same layers
        num_depths = len(data['rho'][0])  # Assuming rho has a consistent depth dimension

        dataset = xr.Dataset({
            'max_nuh': (['time'], data['max_nuh']),
            'h': (['time'], data['h']),
            'sf': (['time', 'layer'], np.stack(data['sf'])),  # Convert list of arrays to 2D array
            # 'rho': (['time', 'depth'], np.stack(data['rho']))  # Convert list of arrays to 2D array
        }, coords={
            'time': data['time'],
            'layer': np.arange(num_layers),
            'depth': np.arange(num_depths),
            'lat': lat,
            'heat_flux': heat_flux,
            'tx_stress': tx_stress,
        })

        all_data.append(dataset)

    # Merge all datasets along the new dimension ('cases' removed, since forcings are unique)
    merged_dataset = xr.concat(all_data, dim='sample')

    return merged_dataset

def extract_info(ds):
    t_initial = 50
    t_max = 50 + 100
    n_layers = 16

    nuh = ds['nuh'][t_initial:t_max, :, 0, 0]
    mld_surf = ds['mld_surf'][t_initial:t_max, 0, 0]
    rho = ma.getdata(ds['rho'][t_initial:t_max, :, 0, 0])
    z = ds['z'][0, :, 0, 0]

    sfs = {
        'max_nuh': [],
        'h': [],
        'time': [],
        'sf': [],
        'rho': [],
    }
    for t in range(nuh.shape[0]):
        h = mld_surf[t]
        mld_surf_idx_t = np.argmin(np.abs(z + mld_surf[t]))
        nuh_t = nuh[t]
        surface_to_mld_by_surf = nuh_t[mld_surf_idx_t:]
        max_val = np.max(surface_to_mld_by_surf)
        normalized_surf = surface_to_mld_by_surf / max_val
        sf = np.array([np.mean(chunk) for chunk in
            np.array_split(normalized_surf, n_layers)
        ])
        sfs['max_nuh'].append(max_val)
        sfs['time'].append(t)
        sfs['h'].append(h)
        sfs['sf'].append(sf)
        sfs['rho'].append(rho)

    return sfs


def main():
    parser = argparse.ArgumentParser(description='Generate and run GOTM cases.')
    parser.add_argument('--generate', action='store_true', help='Generate case directories')
    parser.add_argument('--run', action='store_true', help='Run GOTM cases')

    args = parser.parse_args()

    if args.generate:
        generate_case_dirs()
    elif args.run:
        run_gotm()
    else:
        print("Please specify --generate or --run")


if __name__ == '__main__':
    main()
