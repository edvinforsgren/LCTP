import feature_extraction_sc_utils as fesu
import pandas as pd
import numpy as np
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path to config.yaml file")
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    config_file_path = args.config_file
    print('Opening config file')
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    ## Parse the config data
    try:
        plate = config['plate']
    except KeyError:
        plate = None
    path_incl = config['path_incl']
    base_dir = config['base_dir']
    channels = config['channels']
    save_channel_intensities = config['save_channel_intensities']


    metadata_incl = pd.read_parquet(path_incl)
    channel_means, channel_stds, channel_max, channel_min = [], [], [], []
    for channel_i, channel_name in enumerate(channels):
        print('channel_name', channel_name)
        all_mean, all_std, all_max, all_min = [], [], [], []
        for timepoint in metadata_incl.Metadata_time.unique().tolist():
            time_metadata = metadata_incl.query(f'Metadata_time == "{timepoint}"')
            # create a dataset and load data
            dataset = fesu.CellDataset_zip(time_metadata, base_dir, channel_index=channel_i)
            for i in range(len(dataset)):
                mean, std, maxes, mins, shape = dataset.get_mean_and_std(i)
                all_mean.append(mean)
                all_std.append(std)
                all_max.append(maxes)
                all_min.append(mins)
        print('means:', np.mean(all_mean))
        print('std:', np.mean(all_std))
        print('max:', np.mean(all_max), 'max max:', np.max(all_max))
        print('min:', np.mean(all_min), 'min min:', np.min(all_min))
        print('zip_count: ', dataset.get_zip_count())
        channel_means.append(np.mean(all_mean))
        channel_stds.append(np.mean(all_std))
        channel_max.append(np.max(all_max))
        channel_min.append(np.min(all_min))
        
    data = {
    'channel_means': channel_means,
    'channel_stds': channel_stds,
    'channel_maxs': channel_max,
    'channel_mins': channel_min
    }
    # Write to a YAML file
    with open(save_channel_intensities, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

