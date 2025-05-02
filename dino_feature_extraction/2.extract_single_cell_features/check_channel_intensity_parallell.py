## This import works since the script is called with PYTHONPATH=. in running script which is ran from the root of the repo
# If using a different structure, you may need to adjust the import statement
from dino_feature_extraction import utils as ut
import pandas as pd
import numpy as np
import argparse
import yaml
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import timeit

def process_timepoint(timepoint, metadata_incl, base_dir, channel_index):
    time_metadata = metadata_incl.query(f'Metadata_time == "{timepoint}"')
    dataset = ut.CellDataset_zip(time_metadata, base_dir, channel_index=channel_index)
    
    means, stds, maxes, mins = [], [], [], []
    for i in range(len(dataset)):
        mean, std, max_val, min_val, _ = dataset.get_mean_and_std(i)
        means.append(mean)
        stds.append(std)
        maxes.append(max_val)
        mins.append(min_val)
    
    return np.mean(means), np.mean(stds), np.max(maxes), np.min(mins)

def process_channel(channel_index, metadata_incl, base_dir):
    timepoints = metadata_incl.Metadata_time.unique().tolist()
    
    with mp.Pool() as pool:
        results = list(pool.map(
            partial(process_timepoint, 
                   metadata_incl=metadata_incl, 
                   base_dir=base_dir, 
                   channel_index=channel_index),
            timepoints
        ))
    
    means, stds, maxes, mins = zip(*results)
    return np.mean(means), np.mean(stds), np.max(maxes), np.min(mins)

if __name__ == "__main__":
    # Start timer
    timeit_start = timeit.default_timer()
    parser = argparse.ArgumentParser(description="Path to config.yaml file")
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    
    print('Opening config file')
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Parse the config data
    plate = config.get('plate')  # Using get() with default None
    path_incl = config['path_incl']
    base_dir = config['base_dir']
    channels = config['channels']
    save_channel_intensities = config['save_channel_intensities']

    # Load metadata once
    metadata_incl = pd.read_parquet(path_incl)
    
    # Process each channel
    channel_stats = []
    for channel_i, channel_name in enumerate(channels):
        print(f'Processing channel: {channel_name}')
        mean, std, max_val, min_val = process_channel(channel_i, metadata_incl, base_dir)
        
        print(f'Channel {channel_name} stats:')
        print(f'Mean: {mean:.2f}, Std: {std:.2f}')
        print(f'Max: {max_val:.2f}, Min: {min_val:.2f}')
        
        channel_stats.append((mean, std, max_val, min_val))
    
    # Unzip results
    channel_means, channel_stds, channel_max, channel_min = zip(*channel_stats)
    
    # # Save results
    data = {
    'channel_means': [float(x) for x in channel_means],
    'channel_stds': [float(x) for x in channel_stds],
    'channel_maxs': [float(x) for x in channel_max],
    'channel_mins': [float(x) for x in channel_min],
}

    with open(save_channel_intensities, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print("Elapsed time:", timeit.default_timer() - timeit_start)
