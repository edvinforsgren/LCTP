from analysis import utils as ut
import torch
import pandas as pd
import numpy as np
import os
import yaml
import torch.multiprocessing as mp
import argparse
import torch.nn as nn
import time
import random


# Function to process a single compound index
def process_compound(cmp_idx, excl_plate, global_seed, df, moas, compound_list,
                     hours1, hours2, epochs, layers, batch_size, learning_rate,
                     activation_function, loss_function, drop_out):
     # Set a unique seed for each process
    unique_seed = global_seed + cmp_idx + int(excl_plate[-3:])
    # Set deterministic and other configurations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the random seed for PyTorch, NumPy, and Python's random module
    torch.manual_seed(unique_seed)
    np.random.seed(unique_seed)
    random.seed(unique_seed)

    # If using CUDA, set the seed for each GPU
    torch.cuda.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    excl_cmp = [cl[cmp_idx] for cl in compound_list]
    y_preds = []
    pred_labels = []
    excl_plate = [excl_plate]
    for hours, time in zip([hours1, hours2], ['early', 'late']):
        dmso_hours = hours1 + hours2
        # Prepare training data
        X_train, y_encoded, _ = ut.prepare_data_mod(df, [excl_plate, excl_cmp], hours, moas, train=True, 
                                                excl_metadatas=['Metadata_Plate', 'Metadata_cmpd_cmpdname'], dmso_hours=dmso_hours)
        # Prepare test data
        X_test, y_encoded_test, test_df = ut.prepare_data_mod(df, [excl_plate, excl_cmp + ['dmso']], hours, moas, train=False, 
                                                        excl_metadatas=['Metadata_Plate', 'Metadata_cmpd_cmpdname'])
        # Train model
        mlp = ut.GeneralizedNeuralNetwork(
            hidden_layers=layers, 
            input_dim=X_train.shape[1], 
            output_dim=y_encoded.shape[1], 
            dropout_rate=drop_out,
            activation_fn=activation_function,
            criterion=loss_function
        )
        y_encoded = y_encoded.astype(np.float32)
        mlp.fit(X_train, y_encoded, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, validation_split=0, print_freq=198)

        # Get predictions
        y_preds.append(mlp.predict(torch.from_numpy(X_test).cuda()))
        pred_labels.append([f'{c}_pred_{time}' for c in moas])
    # Create prediction DataFrame
    true_labels = [f'{c}_true' for c in moas]
    pred_df = ut.get_metadata(test_df.reset_index().copy())
    for y_pred, pred_label in zip(y_preds, pred_labels):
        pred_df = pd.concat([pred_df, pd.DataFrame(data=y_pred.cpu().numpy(), columns=pred_label)], axis=1)

    pred_df = pd.concat([pred_df, pd.DataFrame(data=y_encoded_test, columns=true_labels)], axis=1)
    return pred_df

def run_all_data(df, moas, compound_list, hours1, hours2, epochs, layers, batch_size, learning_rate, activation_function, loss_function, drop_out, global_seed):

    # Precompute unique values
    unique_plates = df.Metadata_Plate.unique()
    n_plates = len(unique_plates)
    n_compounds = len(compound_list[0])

    pred_dfs = []

    # Outer loop over plates
    for plate_idx, excl_plate in enumerate(unique_plates):
        print(f"Round {plate_idx} of {n_plates - 1}")
        
        # Initialize multiprocessing pool with a closure for worker_init_fn
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(process_compound,
                [(cmp_idx, excl_plate, global_seed, df, moas, compound_list, hours1, hours2, epochs, layers, batch_size, learning_rate, activation_function, loss_function, drop_out)
                 for cmp_idx in range(n_compounds)]
            )
            
        # Add the results to the prediction dataframe list
        pred_dfs.extend(results)

    dmso_dfs = []
    pred_dfs_new = []
    for pred_df in pred_dfs:
        dmso_df = pred_df[pred_df['Metadata_cmpd_cmpdname'] == 'dmso']
        dmso_dfs.append(dmso_df)
        pred_df = pred_df[pred_df['Metadata_cmpd_cmpdname'] != 'dmso']
        pred_dfs_new.append(pred_df)
    pred_dfs = pred_dfs_new
    full_df = pd.concat(pred_dfs, axis=0).reset_index(drop=True)
    full_dmsos = pd.concat(dmso_dfs, axis=0).reset_index(drop=True)
    
    return full_df, full_dmsos

def smooth_data(save_dir, file_name, full_df, full_dmsos, moas):
    # For faster processing    
    full_df['Metadata_Plate'] = full_df['Metadata_Plate'].astype(object)
    full_dmsos['Metadata_Plate'] = full_dmsos['Metadata_Plate'].astype(object)
    # Define MOAs and compound lists
    moas = [moa for moa in full_df.Metadata_cmpd_moa_group.unique().tolist() 
            if (moa and 'dmso' not in str(moa))]
    moas.sort()
    # Concatenate all prediction dataframes
    full_df.sort_values('Metadata_cmpd_cmpdname', inplace=True)
    # Flatten prediction labels from nested list
    flatten_pred_labels = [f'{c}_pred_{t}' for t in ['early', 'late'] for c in moas]
    
    smooth_df = ut.process_predictions(df=full_df, pred_labels=flatten_pred_labels)
    smooth_cmpd = ut.process_smoothed_predictions(smooth_df, moas)
    
    smooth_cmpd.to_parquet(f'{save_dir}/{file_name}_cmpd_nn_eq_pred_smooth.parquet.gzip', compression='gzip')
    
    smooth_df = ut.process_predictions(df=full_dmsos, pred_labels=flatten_pred_labels)
    smooth_dmso = ut.process_smoothed_predictions(smooth_df, moas)
    
    smooth_dmso.to_parquet(f'{save_dir}/{file_name}_dmso_nn_eq_pred_smooth.parquet.gzip', compression='gzip')
    return smooth_cmpd, smooth_dmso

if __name__  == "__main__":
    # Start times to measure runtime
    start = time.time()
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    config_file_path = args.config_file

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters from config    
    cell = config['data']['cell']
    seed = config['seed']
    activation_str = config['model']['activation']
    activation = getattr(nn, activation_str)
    normalize_str = config['model']['normalize']
    normalize = config['model']['normalize'].lower() == "true"
    epochs = config['model']['epochs']
    layers = config['model']['layers']
    hours1 = config['time']['hours1']
    hours2 = config['time']['hours2']
    data_path = config['data']['path']
    save_path = config['data']['save_path']
    batch_size = config['model']['batch_size']
    learning_rate = config['model']['learning_rate']
    loss_function_str = config['model']['loss_function']
    loss_function = getattr(nn, loss_function_str)
    drop_out = config['model']['dropout_rate']
    compare_config_file_path = config['data']['compare_config_file_path']

    df = pd.read_parquet(data_path)
    if normalize:
        df = ut.normalize_plate_data(df)
    
    # Filter out positive controls
    df = df[df['Metadata_cmpd_moa_group'] != 'undefined'].copy()
    df = df[df['Metadata_cmpd_moa_group'].notnull()].copy()

    # Define MOAs and compound lists
    moas = [moa for moa in df.Metadata_cmpd_moa_group.unique().tolist() 
            if (moa and 'dmso' not in str(moa))]
    moas.sort()

    compound_list = [df[df['Metadata_cmpd_moa_group'] == moa]['Metadata_cmpd_cmpdname'].unique().tolist() 
                    for moa in moas if moa != 'dmso']
    compound_list.sort()

    full_df, full_dmsos = run_all_data(df, moas, compound_list, hours1=hours1, hours2=hours2, epochs=epochs, layers=layers, batch_size=batch_size, learning_rate=learning_rate, activation_function=activation, loss_function=loss_function, drop_out=drop_out, global_seed=seed)

    save_dir = f'{save_path}epochs{epochs}_bs{batch_size}_lr{learning_rate}_loss{loss_function_str}_norm{normalize_str}'
    file_name = f'{cell}_dino_ts'
    os.makedirs(save_dir, exist_ok=True)
    full_df.to_parquet(f'{save_dir}/{file_name}_cmpd_nn_eq_pred.parquet.gzip', compression='gzip')
    full_dmsos.to_parquet(f'{save_dir}/{file_name}_dmso_nn_eq_pred.parquet.gzip', compression='gzip')
    
    smooth_cmpd, smooth_dmso  = smooth_data(save_dir, file_name, full_df, full_dmsos, moas)

    # Load existing config if it exists, otherwise start fresh
    if os.path.exists(compare_config_file_path):
        with open(compare_config_file_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    # Make sure 'single_timepoint' section exists
    if 'time_series' not in config:
        config['time_series'] = {}

    # Add your new key-value pairs under 'single_timepoint'
    config['time_series']['time_series_analysis'] = f"{save_dir}/{file_name}_cmpd_nn_eq_pred.parquet.gzip"
    config['time_series']['time_series_analysis_dmso'] = f"{save_dir}/{file_name}_dmso_nn_eq_pred.parquet.gzip"
    config['time_series']['time_series_analysis_processed'] = f"{save_dir}/{file_name}_cmpd_nn_eq_pred_smooth.parquet.gzip"
    config['time_series']['time_series_dmso_processed'] = f"{save_dir}/{file_name}_dmso_nn_eq_pred_smooth.parquet.gzip"

    # Save back to YAML
    with open(compare_config_file_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

    stop_time = time.time() - start

    print(f"Finished {cell}")
    # Print runtime
    print(f"Runtime: {time.time() - start}")