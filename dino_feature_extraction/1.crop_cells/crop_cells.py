import pandas as pd
import os
## This import works since the script is called with PYTHONPATH=. in running script which is ran from the root of the repo
# If using a different structure, you may need to adjust the import statement
from dino_feature_extraction import utils as ut
from datetime import datetime
import yaml
import argparse


def parse_time_point(time_point):
    return datetime.strptime(time_point, "%yy%mm%dd%Hh%M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path to config.yaml file")
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    config_file_path = args.config_file

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    ## Parse the config data
    plate = config.get('plate')  # Using get() with default None
    df_path = config['dataframe_path']
    base_dir = config['base_dir']
    save_dir = config['save_dir']
    sub_dir = config['sub_dir']
    channels = config['channels']
    cell_healths = config['cell_healths']
    
    df = pd.read_parquet(df_path)
    df = ut.get_metadata(df)
    if plate is not None:
        print('Only analyzing plate: ', plate)
        df = df[df['Metadata_Plate'] == plate]
        print(len(df))
    else:
        print('Running on all plates')
    scan_dirs = []
    mask_dirs = []
    
    i=0
    for job, metadata in df.groupby('Metadata_AIch_job'):
        print("job:", job)
        scans, masks = ut.find_matching_mask_and_image_directories(base_dir, job=str(job), plate=str(metadata.Metadata_VesselID.unique().item()))
        if scans and masks:
            scan_dirs.append(scans)
            mask_dirs.append(masks)
        else:
            print("no scans for", job) 
    length = len(mask_dirs)
    for i, mask_dir_scan_dir in enumerate(zip(mask_dirs, scan_dirs)):
        mask_dir, scan_dir = mask_dir_scan_dir
        vessel_id = os.path.basename(scan_dir[0])
        print("vessel_id:", vessel_id)
        print(i, " of ", length)
        time_points = []
        for dir, sc_dir in zip(mask_dir, scan_dir):
            time = ut.extract_time_from_mask_directory(dir)
            time_points.append(time)
        parsed_time_points = [parse_time_point(time_point=tp) for tp in time_points]
        first_timepoint = min(parsed_time_points)
        index = parsed_time_points.index(first_timepoint)
        first_scan_time = ut.convert_time_point_to_scan_format(time_point=time_points[index])
        print(first_scan_time, "==", time_points[index])
        new_dir = os.path.join(save_dir, sub_dir)
        df_incl_list_path, df_excl_list_path, time_crop = ut.crop_save_all_cells_and_channel_zipped(df, save_dir, mask_dir, scan_dir, channels, cell_healths, first_scan_time, ut.single_cell_crop, sub_dir, vessel_id)
    print(time_crop)
        