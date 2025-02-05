"""Helper script to download and shrink power outage data for easy processing."""
import os
import pandas as pd
import numpy as np
import requests


DATA_DIR = 'data/'

# List of states on eastern seaboard / south coast
relevant_states = [
    'Texas',
    'Florida',
    'Louisiana',
    'Mississippi',
    'Alabama',
    'Georga',
    'South Carolina',
    'North Carolina',
    'Virginia',
    'Maryland',
    'Delaware',
    'New Jersey',
    'New York',
    'Connecticut',
    'Rhode Island',
    'Massachusetts',
    'New Hampshire',
    'Maine'
]

def download_source_data():
    """Download files from figshare article:

    https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376

    """
    source_url = 'https://api.figshare.com/v2/articles/24237376'
    response = requests.get(source_url)
    data = response.json()
    for file in data['files']:
        output_path = DATA_DIR + file['name']
        if os.path.exists(output_path):
            continue
        if 'eaglei_outages' in file['name'] or 'MCC.csv' in file['name']:
            print('Downloading', file['name'])
            file_url = file['download_url']
            response = requests.get(file_url)
            with open(output_path, 'wb') as f:
                f.write(response.content)



def only_relevant_states(row):
    return row['state'].strip() in relevant_states

def shrink_csv(path):
    print('Shrinking', path)
    df = pd.read_csv(path)
    df = df[df.apply(only_relevant_states, axis=1)]

    df['fips_code'] = ['0' + str(code) if len(str(code)) < 5 else str(code) for code in df['fips_code']]
    df['run_start_time'] = pd.to_datetime(df['run_start_time'], format = '%Y-%m-%d %H:%M:%S')
    df['day'] = df['run_start_time'].dt.date
    df['hour'] = df['run_start_time'].dt.hour
    df = df.groupby(by = ['fips_code', 'county', 'state', 'day']).max()
    df = df.drop(columns = 'run_start_time')
    df = df.reset_index()
    return df

def shrink_csvs():
    did_shrink = False
    for year in range(2014, 2023):
        input_path = f'eaglei_outages_{year}.csv'
        output_path = f"eaglei_outages_{year}_filtered.npz"
        if os.path.exists(DATA_DIR + output_path):
            continue

        did_shrink = True
        df = shrink_csv(DATA_DIR + input_path)
        # Include column names in the saved npz
        np.savez(DATA_DIR + output_path, **df)

    if not did_shrink:
        print('Compressed data already present.')

def load_yearly_data():
    dfs_by_year = {}
    for year in range(2014, 2023):
        output_path = f"eaglei_outages_{year}_filtered.npz"
        npz = np.load(DATA_DIR + output_path, allow_pickle=True)
        df = pd.DataFrame({key: npz[key] for key in npz.keys()})
        dfs_by_year[year] = df
    
    return dfs_by_year


def main():
    download_source_data()
    shrink_csvs()

if __name__ == '__main__':
    main()