import os
import re
import pandas as pd
from PIL import Image
import numpy as np
import time
import zipfile
import io
from datetime import datetime
from scipy.ndimage import binary_dilation
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """return a list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return a dataframe of just metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]

def extract_time_from_mask_directory(mask_dir):
    """
    Extract the time point from the mask directory structure.
    """
    match = re.search(r'(\d{2}y\d{2}m\d{2}d\d{2}h\d{2})', mask_dir)
    if match:
        return match.group(0)
    return None

def convert_time_point_to_scan_format(time_point):
    """
    Convert the time point format from mask to scan directory format.
    """
    return time_point.replace('y', '').replace('m', '/').replace('d', '/').replace('h', '')

def find_matching_mask_and_image_directories(root_path, job, plate):
    """
    Find matching scan and mask directories for the given well across all time points.
    """
    ## This is the Job with AIch 
    mask_base_dir = os.path.join(root_path, "Jobs", job,)
    scan_base_dir = os.path.join(root_path, "ScanData")
    matching_mask_dirs = []
    matching_scan_dirs = []
    for root, dirs, _ in os.walk(mask_base_dir):
        for dir_name in dirs:
            mask_dir = os.path.join(root, dir_name)
            time_point = extract_time_from_mask_directory(mask_dir)
            if time_point:
                scan_dir_format = convert_time_point_to_scan_format(time_point)
                scan_dir = os.path.join(scan_base_dir, scan_dir_format, plate)
                if os.path.exists(scan_dir):
                    matching_mask_dirs.append(mask_dir)
                    matching_scan_dirs.append(scan_dir)
                else:
                    print('No matching scans for' , mask_dir)

    return matching_scan_dirs, matching_mask_dirs

def crop_save_all_cells_and_channel_zipped(df, base_dir, mask_dirs, scan_dirs, channels, cell_healths, start_time, crop_func, crop_dir, vessel_id, date_format='%y%m/%d/%H%M'):
    """ 
    A function that crops single-cells and saves them in zip-directories, this to avoid numerous files of single cell crops. The function creates two subdirectories
    /incl and /excl which contains the images and its metadata. 
    param df: dataframe containing treatment info. 
    param base_dir: The directory of where to save the cropped images and the metafiles
    param mask_dirs: directory of masks.
    param scan_dirs: directories of scan images
    param channels: The channels of you images as a list. 
    param cell_healths: The cell healths of your masks as list ex=['Live', 'Dead']
    param start_time: The date and time cultures was started.
    return: dataframe with metadata, original scan path and new directory for cropped and zero padded images.
    return: time it took to crop all images
    """
    df = df.fillna('undefined')  # Fills NaN with 'undefined'
    df = df[df['Metadata_VesselID'] == vessel_id]
    all_wells = df['Metadata_Well'].unique().tolist()
    sites = [1]
    metadata_list_incl = []
    metadata_list_excl = []
    img_paths_incl = []
    img_paths_excl = []
    filenames_incl = []
    filenames_excl = []
    time_crop = 0
    sub_dir = os.path.join(base_dir, crop_dir)
    incl_dir = 'incl' # subdirectory for cells 224x224
    excl_dir = 'excl' # subdirectory for cells >224x224 or <224x224

    for i in range(len(scan_dirs)):
        print(i+1, 'of', len(scan_dirs), ' directories')
        for well in all_wells:
            sub_df = df[df['Metadata_Well'] == well]
            metadata = sub_df.iloc[0].tolist()  # Extract metadata as a list for the current well
            for site in sites: 
                tensors_incl = []
                tensors_excl = []
                all_filenames_incl = []
                all_filenames_excl = []
                zip_path_incl = None
                zip_path_excl = None
                matching_files = get_matching_files(mask_dir=mask_dirs[i], scan_dir=scan_dirs[i], well=well, site=site)
                for cell_health in cell_healths:
                    pairs = [(mask_path, scan_path) for mask_path, scan_path in matching_files 
                        if f'Aich{cell_health}Labels' in mask_path and any(scan_path.endswith(f'{channel}.tif') for channel in channels)]
                    mask_path, scan_path = pairs[0] # Since they all have the same mask
                    with Image.open(mask_path) as mask_img:
                        mask_array = np.array(mask_img)
                    if np.max(mask_array) == 0:
                        continue
                    date_time = scan_dirs[i][-16:-4]
                    hh, mm = extract_hours_minutes(date_time, start_time, date_format)
                    plate = re.search(r"/(\d+)/([^/]+)\.tif$", scan_path).group(1)  # Extract plate from first scan path of the list.
                    scan_arrays = []
                    scan_paths = []
                    scan_channels = []
                    for _, scan_path in pairs:
                        with Image.open(scan_path) as scan_img:
                            scan_array = np.array(scan_img)
                        scan_arrays.append(scan_array)
                        scan_paths.append(scan_path)
                        channel = next((channel for channel in channels if scan_path.endswith(f'{channel}.tif')), None)
                        scan_channels.append(channel)
                    stacked_array = np.stack(scan_arrays)

                    for cell in range(1, np.max(mask_array)+1):
                        metadata_copy = metadata.copy()  # Create a copy of metadata
                        metadata_copy.extend([date_time, plate, site, cell, hh, mm, f'{int(hh)}h{int(mm):02}m', cell_health])
                        filenames_incl = []
                        filenames_excl = []
                        scan_paths_incl = []
                        scan_paths_excl = []
                        if not np.any(mask_array == cell):  # Check if the cell actually exists in the mask
                            continue
                        start = time.perf_counter()
                        padded_imgs, ok_size = crop_func(stacked_array, mask_array, cell)
                        end = time.perf_counter()
                        time_crop += (end-start) # extract time it took to crop
                        if ok_size: # cells <224
                            dir1 = os.path.join(sub_dir, incl_dir) # Where to save the crops (absolute path)
                            os.makedirs(dir1, exist_ok=True)
                            for channel_index, padded_img in enumerate(padded_imgs):
                                tensor, filename, dir_path = crops_plate_channel_torch(dir1, hh, mm, plate, well, site, cell, padded_img, scan_channels[channel_index], cell_health)
                                tensors_incl.append(tensor)

                                filenames_incl.append(filename)
                                scan_paths_incl.extend(scan_path for scan_path in scan_paths if scan_channels[channel_index] in scan_path[-7:])
                            zip_path_incl = os.path.join(dir_path, f"{well}.zip")
                            all_filenames_incl.extend(filenames_incl)
                        else: # cells >224
                            dir2 = os.path.join(sub_dir, excl_dir)
                            os.makedirs(dir2, exist_ok=True)
                            for channel_index, padded_img in enumerate(padded_imgs):
                                tensor, filename, dir_path = crops_plate_channel_torch(dir2, hh, mm, plate, well, site, cell, 
                                                                           padded_img, scan_channels[channel_index], cell_health)
                                tensors_excl.append(tensor)
                                filenames_excl.append(filename)
                                scan_paths_excl.extend(scan_path for scan_path in scan_paths if scan_channels[channel_index] in scan_path[-7:])
                            zip_path_excl = os.path.join(dir_path, f"{well}.zip")
                            all_filenames_excl.extend(filenames_excl)
                        if filenames_incl:
                            metadata_list_incl.append(metadata_copy)
                            incl_paths = [custom_sort(scan_paths_incl), zip_path_incl, custom_sort(filenames_incl)]
                            img_paths_incl.append(incl_paths)
                        if filenames_excl:
                            metadata_list_excl.append(metadata_copy)
                            excl_paths = [custom_sort(scan_paths_excl), zip_path_excl, custom_sort(filenames_excl)]
                            img_paths_excl.append(excl_paths)
                            
            if zip_path_incl:
                with zipfile.ZipFile(zip_path_incl, 'w', zipfile.ZIP_STORED) as zipf:
                    for fn, tensor_incl in zip(all_filenames_incl, tensors_incl):
                        buffer = io.BytesIO()
                        torch.save(tensor_incl, buffer)
                        zipf.writestr(fn, buffer.getvalue())
            if zip_path_excl:
                with zipfile.ZipFile(zip_path_excl, 'w', zipfile.ZIP_STORED) as zipf:
                    for fn, tensor_excl in zip(all_filenames_excl, tensors_excl):
                        buffer = io.BytesIO()
                        torch.save(tensor_excl, buffer)
                        zipf.writestr(fn, buffer.getvalue())


                        
    path_columns = ['Scan_Paths', 'Zip_Paths', 'Filenames']
    df_incl = pd.DataFrame(metadata_list_incl, columns=df.columns.tolist() + ['Metadata_Date', 'Metadata_Vessel', 'Metadata_Site', 'Metadata_Cell_ID', 'Metadata_hours', 'Metadata_min', 'Metadata_time', 'Metadata_cell_health'])
    df_paths_incl = pd.DataFrame(img_paths_incl, columns=path_columns)

    df_excl = pd.DataFrame(metadata_list_excl, columns=df.columns.tolist() + ['Metadata_Date', 'Metadata_Vessel', 'Metadata_Site', 'Metadata_Cell_ID', 'Metadata_hours', 'Metadata_min', 'Metadata_time', 'Metadata_cell_health'])
    df_paths_excl = pd.DataFrame(img_paths_excl, columns=path_columns)

    df_incl = pd.concat([df_incl, df_paths_incl], axis=1)
    df_excl = pd.concat([df_excl, df_paths_excl], axis=1)
    
    print(f"Directory to save the metadata: {os.path.join(sub_dir, incl_dir, 'metadata_incl')}")
    df_incl.to_parquet(os.path.join(sub_dir, incl_dir, 'metadata_incl'), engine='pyarrow', partition_cols=['Metadata_Plate', 'Metadata_hours', 'Metadata_min'], existing_data_behavior='delete_matching') # partition cols decides how the data should be partitioned. 
    df_excl.to_parquet(os.path.join(sub_dir, excl_dir, 'metadata_excl'), engine='pyarrow', partition_cols=['Metadata_Plate', 'Metadata_hours', 'Metadata_min'], existing_data_behavior='delete_matching')

    return df_incl, df_excl, time_crop

def custom_sort(file_list):
    """Custom sort function to place 'Ph' file first, then sort the remaining files."""
    ph_file = [f for f in file_list if 'Ph' in f]
    other_files = sorted([f for f in file_list if 'Ph' not in f])
    return ph_file + other_files

def crops_plate_channel_torch(output_dir, hh, mm, plate, well, site, cell_id, img, channel, cell_health):
    """ 
    A function that creates a subdirectory for the image, save the image, and returns the path
    date must be on the format 'yymm/dd/hhhh'
    Files will be saved into directories of the format output_dir/plate/hour/min
    Observe that the path returned is the absolute path
    """
    hhmm = f'{int(hh)}h{int(mm):02}m' # create correct format for time
    dir_path = os.path.join(output_dir, plate, hhmm) 
    os.makedirs(dir_path, exist_ok=True)
    file_name = f'{well}-{site}-{cell_id}-{cell_health}-{channel}.pt'
    if img.dtype == 'uint16':
        img = (img / 2).astype('uint8')
    tensor = torch.tensor(img)
    return tensor, file_name, dir_path

def extract_hours_minutes(date_str, start_time_str, date_format='%y%m/%d/%H%M'):
    """
    A function that returns hours and minutes passed since start_time using the date
    """
    date_time = datetime.strptime(date_str, date_format)
    start_time = datetime.strptime(start_time_str, date_format)
    time_difference = date_time - start_time
    hours = time_difference.total_seconds() // 3600
    minutes = (time_difference.total_seconds() % 3600) // 60
    return hours, minutes

def get_matching_files(mask_dir, scan_dir, well, site):
    """
    Get matching mask and scan files for the given well and site in the specified directories.
    """
    prefix = f"{well}-{site}"
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif') and f.startswith(prefix)]
    mask_files = [os.path.join(mask_dir, f) for f in mask_files]  # Convert to full paths

    scan_files = get_files_in_subdirectories(scan_dir, '.tif')
    scan_files = [f for f in scan_files if os.path.basename(f).startswith(prefix)]
    
    # Match files based on 'Well-Site'
    matching_files = []
    for mask_file in mask_files:
        mask_file_prefix = '-'.join(os.path.basename(mask_file).split('-')[:2])  # Extract 'Well-Site'
        for scan_file in scan_files:
            scan_file_prefix = '-'.join(os.path.basename(scan_file).split('-')[:2])
            if mask_file_prefix == scan_file_prefix:
                matching_files.append((mask_file, scan_file))

    return matching_files

def get_files_in_subdirectories(directory, extension):
    """
    Get a list of files with a specific extension in the given directory and its subdirectories.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def single_cell_crop(scan_arrays, mask_array, cell, size = 224, mask_padding=5): # this is recently renamed so if something gives error regarding a function called ss_crop_cell this is that function
    """ 
    A function that returns a tuple containing the segmented and zero padded image and a bool
    The bool is True if image size is smaller than 224 and else False. 
    """
    # Find the coordinates of the cell in the mask
    y, x = np.where(mask_array == cell)
    
    # Crop the scan arrays and mask
    y_min, y_max = y.min(), y.max() + 1
    x_min, x_max = x.min(), x.max() + 1
    cropped = scan_arrays[:, y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask_array[y_min:y_max, x_min:x_max]
    
    # Create a binary mask for the cell and expand it
    cell_mask = (cropped_mask == cell)
    expanded_mask = binary_dilation(cell_mask, iterations=mask_padding)
    
    # Zero out non-cell areas using the expanded mask
    cropped[:, ~expanded_mask] = 0
    
    # Check if the cropped image is larger than the desired size or touches the border
    if (cropped.shape[1] > size or cropped.shape[2] > size or y_min == 0 or y_max == mask_array.shape[0] or x_min == 0 or x_max == mask_array.shape[1]):
        return cropped, False
    
    return cropped, True

class CellDataset_zip(Dataset):
    """ 
    A class that creates a detaset from a dataframe containing metadata and image paths. 
    Param df: Dataframe containing metadata and image paths. Metadata columns must start with 'Metadata'
    Param dir: Since the paths in the dataframe is relative, the absolute path to where the images are must be provided.
    If the images as not the correct size for the model used, they have to be resized and centercropped. 
    """
    def __init__(self, df, dir, channel_index=0, channel_mean=0.5, channel_std=0.1, channel_max=1):
        self.df = df
        self.dir = dir
        self.metadata_columns = [col for col in self.df.columns if col.startswith('Metadata')]
        self.zip_path = None
        self.zip_cache = None
        self.channel_index = channel_index
        self.zip_count = 0
        self.channel_max = channel_max
        self.channel_mean = channel_mean / channel_max
        self.channel_std = channel_std / channel_max
        self.transform = transforms.Compose([
            transforms.Lambda(self.__scale_and_stack),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[self.channel_mean, self.channel_mean, self.channel_mean], std=[self.channel_std, self.channel_std, self.channel_std]),
        ])
    
    def __scale_and_stack(self, image):
        image = image.float() / self.channel_max
        image = torch.stack([image, image, image], dim=0)
        return image
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # transforms and returns metadata and image. 
        row = self.df.iloc[idx]
        metadata = row[self.metadata_columns].tolist()
        
        zip_path = row['Zip_Paths']
        file_name = row['Filenames'][self.channel_index] # index 0 points to Ph

        # Open the ZIP file and read specific tensors
        if zip_path == self.zip_path:
            with self.zip_cache.open(file_name) as file:
                buffer = io.BytesIO(file.read())
                image = torch.load(buffer, weights_only=True)
        else:
            if self.zip_cache:
                self.zip_cache.close()
            self.zip_cache = zipfile.ZipFile(zip_path, mode='r')
            self.zip_path = zip_path
            with self.zip_cache.open(file_name) as file:
                buffer = io.BytesIO(file.read())
                image = torch.load(buffer, weights_only=True)
                
        image_transformed = self.transform(image)
        return metadata, image_transformed
        
    def get_zip_count(self):
        return self.zip_count  
        
    def get_mean_and_std(self, idx):
        # transforms and returns metadata and image. 
        row = self.df.iloc[idx]
        metadata = row[self.metadata_columns].tolist()
        
        zip_path = row['Zip_Paths']
        file_name = row['Filenames'][self.channel_index] # index 0 points to Ph

        # Open the ZIP file and read specific tensors
        if zip_path == self.zip_path:
            with self.zip_cache.open(file_name) as file:
                buffer = io.BytesIO(file.read())
                image = torch.load(buffer, weights_only=True)
        else:
            self.zip_count += 1
            if self.zip_cache:
                self.zip_cache.close()
            self.zip_cache = zipfile.ZipFile(zip_path, mode='r')
            self.zip_path = zip_path
            with self.zip_cache.open(file_name) as file:
                buffer = io.BytesIO(file.read())
                image = torch.load(buffer, weights_only=True)
                
        image = image.float()
        return image.mean(), image.std(), image.max(), image.min(), image.shape
    
    
def get_all_features(df, dataloader: DataLoader, model):
    """ 
    df: original dataframe to get the column names.
    dataloader: Dataloader with you data
    model: model of choice. Observe that the model of choice must be compatible with the nr of channels in your images. 
    The funciton will print the first two rows so the user knows if the output is correct. 
    Return: a dataframe of metadata and features extracted from each single cell.
    """
    df_all = pd.DataFrame()
    metadata_list = []
    feature_array = []
    
    for metadata_batch, image in dataloader:
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            outputs = model(image)
        outputs = outputs.cpu().numpy()
        
        batch_size = image.size(0) # Dynamically updated incase sample size is not divisible by batch size. 
        # This below is a bit of a hack, but it works. It should be possible to get the metadata from the dataloader with the whole batch instead of iterating over the batch size... 
        if batch_size == 1:
            metadata = [item.item() if isinstance(item, torch.Tensor) else item[0] for item in metadata_batch]
            output_values = outputs[0]
            metadata_list.append(metadata)
            feature_array.append(output_values)
        else:
            for i in range(batch_size):  # Iterate up to the length
                sample_metadata = []
                for metadata in metadata_batch:
                    if isinstance(metadata, torch.Tensor):
                        sample_metadata.append(metadata[i].item())
                    else:
                        sample_metadata.append(metadata[i])
                metadata_list.append(sample_metadata)
                feature_array.append(outputs[i]) 
    
    metadata_columns = df.columns[df.columns.str.startswith('Metadata')].tolist() # Get the metadata columns
    df_metadata = pd.DataFrame(metadata_list, columns=metadata_columns)
    df_features = pd.DataFrame(feature_array)
    
    # Concatenate metadata and features DataFrames
    df_all = pd.concat([df_metadata, df_features], axis=1)

    return df_all
