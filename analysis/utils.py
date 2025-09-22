import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_metacols(df):
    """Return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]

def get_featurecols(df):
    """Return a list of feature data columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]

def get_metadata(df):
    """Return dataframe of just metadata columns"""
    return df[get_metacols(df)]

def get_featuredata(df):
    """Return dataframe of just feature data columns"""
    return df[get_featurecols(df)]

def normalize_plate_data(df, dmso_condition='dmso'):
    """Normalize data plate-wise using DMSO controls"""
    norm_dfs = []
    for plate, plate_df in df.groupby('Metadata_Plate', observed=False):
        dmso_df = plate_df[plate_df['Metadata_cmpd_cmpdname'] == dmso_condition]
        mean_dmso = get_featuredata(dmso_df).mean(axis=0)
        std_dmso = get_featuredata(dmso_df).std(axis=0)
        
        feature_data = get_featuredata(plate_df)
        normalized_data = (feature_data - mean_dmso) / std_dmso
        plate_df_copy = plate_df.copy()
        plate_df_copy.update(normalized_data)
        norm_dfs.append(plate_df_copy)
        
    return pd.concat(norm_dfs, axis=0)

def process_predictions(df, pred_labels):
    """Helper function for plotting predictions without plate-wise styling"""
    for pred_label in pred_labels:
        for plate, plate_group in df.groupby('Metadata_Plate'):
            for _, group in plate_group.groupby('Metadata_Well'):
                group['Metadata_hours'] = pd.to_numeric(group['Metadata_hours'], errors='coerce')
                group = group.dropna(subset=['Metadata_hours']).sort_values('Metadata_hours')
                
                smoothed_values = noise_reduction(group[pred_label].values)
                df.loc[group.index, f'smoothed_{pred_label}'] = smoothed_values
    return df

def process_smoothed_predictions(pred_dfs, moas):
    pred_labels = []
    for time in ['early', 'late']:
        pred_labels.extend([f'{c}_pred_{time}' for c in moas])
    if isinstance(pred_dfs, list):
        full_df = pd.concat(pred_dfs, axis=0).reset_index(drop=True)
    else:
        full_df = pred_dfs.copy()
    pred_labels = [f'smoothed_{pred_label}' for pred_label in pred_labels]
    
    metadata_cols = ['Metadata_Plate', 'Metadata_Well', 'Metadata_cmpd_cmpdname', 'Metadata_cmpd_oldcmpdname', 
                    'Metadata_cmpd_moa_group', 'Metadata_Cells']
    full_df = full_df.sort_values(by=metadata_cols).reset_index(drop=True).copy()
    df_pivot = full_df.melt(id_vars=metadata_cols + ['Metadata_hours'], 
                           value_vars=pred_labels, 
                           var_name='Prediction', 
                           value_name='Value')
    
    df_pivot['Prediction'] = df_pivot['Prediction'] + '_' + df_pivot['Metadata_hours'].astype(str) + 'h'
    df_pivot = df_pivot.drop(columns=['Metadata_hours'])
    df_pivot = df_pivot.sort_values(by=metadata_cols).reset_index(drop=True).copy()
    df_final = df_pivot.pivot_table(index=metadata_cols, 
                                   columns='Prediction', 
                                   values='Value').reset_index()
    
    sum_cols_dict = {pred: [col for col in df_final.columns if col.startswith(pred)] 
                    for pred in pred_labels}
    
    for pred, cols in sum_cols_dict.items():
        df_final[pred + '_sum'] = df_final[cols].sum(axis=1)
        
    return df_final

def noise_reduction(k):
    """
    Apply noise reduction to a time series using exponential smoothing
    
    Args:
        k: Input time series
    Returns:
        Smoothed time series
    """
    K = np.zeros_like(k)
    for i in range(len(K)):
        K[i:] = K[i] * 0.8 + k[i] * 0.2
    return K

class GeneralizedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_fn=nn.LeakyReLU, output_activation_fn=None, criterion=nn.L1Loss, optimizer=optim.AdamW, dropout_rate=None):
        super(GeneralizedNeuralNetwork, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            if activation_fn:
                layers.append(activation_fn())
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation_fn:
            layers.append(output_activation_fn())
        
        self.network = nn.Sequential(*layers)
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        return self.network(x)

    def fit(self, X_train, Y_train, epochs=50, batch_size=10, learning_rate=0.0001, validation_split=0.1, print_freq=10):
        criterion = self.criterion()
        optimizer = self.optimizer(self.parameters(), lr=learning_rate)
        
        X_train = torch.from_numpy(X_train).type(torch.float32)
        Y_train = torch.from_numpy(Y_train).type(torch.float32)
        # Move everythinig to GPU if available
        if torch.cuda.is_available():
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
        if validation_split > 0:
            val_size = int(len(X_train) * validation_split)
            train_size = len(X_train) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                torch.utils.data.TensorDataset(X_train, Y_train), 
                [train_size, val_size]
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        # Enable DataParallel if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     self.network = nn.DataParallel(self.)network
        
        if torch.cuda.is_available():
            self.network.cuda()
            
        # No need to pin memory for DataLoader since everything is on GPU
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:                    
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            if epoch % print_freq == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        if torch.cuda.is_available():
            X_train = X_train.detach().cpu()
            Y_train = Y_train.detach().cpu()

    def predict(self, X_test):
        self.eval()
        if torch.cuda.is_available():
            X_test = X_test.cuda()
        with torch.no_grad():
            outputs = self.forward(X_test)
        return outputs
    
def prepare_data_mod(df, excls, hours, moas, train=True, excl_metadatas=['Metadata_Plate', 'Metadata_cmpd_cmpdname'], dmso_hours=None):
    """Prepare training or test data for the neural network
    
    Args:
        df: Input dataframe
        excl_plate: Plate to exclude from training or use for testing
        excl_cmp: Compound to exclude from training or use for testing
        hours: Time point to use
        moas: List of MOA classes
        train: If True prepare training data, if False prepare test data
        excl_metadatas: Not used after fixed logic
    """

    df_copy = df.copy()
    excl_plate = excls[0]
    excl_cmps = excls[1]
    
    if train:
        df_main = df_copy[df_copy['Metadata_hours'].isin(hours)] # pick out hours of interest
        df_main = df_main[df_main['Metadata_cmpd_cmpdname'] != 'dmso'] # Remove dmso 

        # remove test compounds and plate
        df_main = df_main[~df_main.Metadata_cmpd_cmpdname.isin(excl_cmps)]
        df_main = df_main[~df_main.Metadata_Plate.isin(excl_plate)]

        # pick out dmso for training, excl plate. 
        df_dmso = df_copy[(df_copy.Metadata_cmpd_cmpdname == 'dmso') & (~df_copy.Metadata_Plate.isin(excl_plate))]

        # Add hours for training of dmso 
        if dmso_hours:
            df_dmso = df_dmso[df_dmso['Metadata_hours'].isin(dmso_hours)]

        data_df = pd.concat([df_main, df_dmso]) # build train dataframe with dmso hours

    else:
        # pick out train plates and cmpds, include all hours
        df_train = df_copy[df_copy.Metadata_cmpd_cmpdname.isin(excl_cmps)] # dmso is included here
        df_train = df_train[df_train.Metadata_Plate.isin(excl_plate)] # Pick out only test plate

        data_df = df_train.copy()

    # Pick out X and y values
    X = get_featuredata(data_df).values
    y = data_df['Metadata_cmpd_moa_group']
    y_encoded = np.zeros((len(y), len(moas)))
    for i, label in enumerate(y):
        if label in moas:
            y_encoded[i, moas.index(label)] = 1
    else:
        return X, y_encoded, data_df