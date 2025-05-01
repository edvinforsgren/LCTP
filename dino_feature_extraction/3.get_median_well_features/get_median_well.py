import pandas as pd
import pyarrow.parquet as pq
import os
import argparse
import yaml
## This import works since the script is called with PYTHONPATH=. in running script which is ran from the root of the repo
# If using a different structure, you may need to adjust the import statement
from dino_feature_extraction import utils as ut
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path to config.yaml file")
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    config_file_path = args.config_file

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    save_path = config['save_path']
    # Define the path to the Parquet files
    base_dir = config['base_dir']
    channels = config['channels']

    # To keep
    metadata_cols = [
        'Metadata_Plate',
        'Metadata_hours',
        'Metadata_Well',
        'Metadata_cmpd_cmpdname',
         'Metadata_cmpd_oldcmpdname',
        'Metadata_cmpd_moa_group',
        'Metadata_Date',
        'Metadata_time',
        'Metadata_min',
        'Metadata_Cells',
        'Metadata_Vessel'
    ]

    result_df = pd.DataFrame()
    # Iterate through each partition
    for channel in channels:
        print('This is for channel:', channel)
        channel_path = base_dir + "_" + channel
        for plate in os.listdir(channel_path):
            plate_path = os.path.join(channel_path, plate)
            plate = plate.split('=')[-1]
            if os.path.isdir(plate_path):
                for hour in os.listdir(plate_path):
                    hour_path = os.path.join(plate_path, hour)
                    if os.path.isdir(hour_path):
                        # Read the Parquet file
                        hour = re.findall(r'\d+', hour)[0]
                        print('Plate:', plate, 'Hour:', hour)
                        sel_data = [('Metadata_Plate', '==', plate), ('Metadata_hours', '==', int(hour))]
                        df = pd.read_parquet(channel_path, filters=sel_data, engine='pyarrow')
                        print(df.shape)

                        # Filter for 'Live' cells
                        live_cells = df[df['Metadata_cell_health'] == 'Live']

                        # Select feature columns (excluding unwanted metadata)
                        feature_cols = ut.get_featurecols(df) #[col for col in df.columns if col not in metadata_cols and 'Metadata_' not in col]

                        # Group by 'Metadata_Well' and calculate median for features
                        median_features = live_cells.groupby('Metadata_Well')[feature_cols].median().reset_index()

                        # Merge with desired metadata
                        metadata = live_cells[metadata_cols].drop_duplicates(subset='Metadata_Well')
                        result = pd.merge(metadata, median_features, on='Metadata_Well')
                        
                        # Concatenate the result to the main DataFrame
                        result_df = pd.concat([result_df, result], ignore_index=True)
                        
        # convert Metadata_hours to int
        result_df['Metadata_hours'] = result_df['Metadata_hours'].astype(int)
        # Add 4 hours to P106073 since the first hours are missing due to imaging issues
        result_df.loc[result_df['Metadata_Plate'] == "P106073", 'Metadata_hours'] = result_df[result_df['Metadata_Plate'] == "P106073"]['Metadata_hours'] + 4
        result_df.to_parquet(save_path, engine='pyarrow', compression='gzip')