## This import works since the script is called with PYTHONPATH=. in running script which is ran from the root of the repo
# If using a different structure, you may need to adjust the import statement
from dino_feature_extraction import utils as ut
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path to config.yaml file")
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    config_file_path = args.config_file

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    ## Parse the config data
    plate = config.get('plate')
    path_incl = config['path_incl']
    base_dir = config['base_dir']
    save_dir = config['save_dir']
    channels = config['channels']
    channel_intensities = config['save_channel_intensities']
    batch_size = config['batch_size']
    model_info = config['model']
    # Define base dir, load the metadata and split paths into individual columns
    with open(channel_intensities, 'r') as file:
        channel_intensities = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Load model:
    model = torch.hub.load(model_info[0], model_info[1])
    model.to(device)

    print('Succesfully loaded model:', model_info[1], 'from:', model_info[0])
    metadata_incl = pd.read_parquet(path_incl)
    
    for channel_i, channel_name in enumerate(channels):
        for n_timepoint, timepoint in enumerate(metadata_incl.Metadata_time.unique().tolist()):
            print("Extracting from: ", timepoint)
            print(n_timepoint, "of", len(metadata_incl.Metadata_time.unique().tolist()))
            time_metadata = metadata_incl.query(f'Metadata_time == "{timepoint}"')
            # create a dataset and load data
            
            # dataset = fesu.CellDataset_zip(time_metadata, base_dir, channel_index=channel_i, channel_mean=channel_intensities['channel_means'][channel_i], channel_std=channel_intensities['channel_stds'][channel_i], channel_max=channel_intensities['channel_maxs'][channel_i])
            dataset = ut.CellDataset_zip(time_metadata, base_dir, channel_index=channel_i, channel_mean=channel_intensities['channel_means'][channel_i], channel_std=channel_intensities['channel_stds'][channel_i], channel_max=channel_intensities['channel_maxs'][channel_i])
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

            # features_all_cell_health_Ph = fesu.get_all_features(time_metadata, dataloader, model)
            features_all_cell_health_Ph = ut.get_all_features(time_metadata, dataloader, model)
            full_save_dir = save_dir + "_" + channel_name
            print(f"Directory to save the metadata: {full_save_dir}")
            features_all_cell_health_Ph.to_parquet(save_dir + "_" + channel_name, engine='pyarrow', partition_cols=['Metadata_Plate', 'Metadata_hours', 'Metadata_min'], existing_data_behavior='delete_matching')