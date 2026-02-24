import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from making_style import get_mask_style
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPullerDJepa(Dataset):
    def __init__(self,
    data_paths,
    patch_size,
    batch_size,
    ratio_patches,
    mask_ratio,
    masking_type,
    num_semantic_tokens,
    input_variables,
    timestamp_cols,
    type_data,
    val_prec = 0.1,
    test_prec = 0.25,
    epochs = 5000,
    stride = None):
        self.batch_size = batch_size
        self.ratio_patches = ratio_patches
        self.mask_ratio = mask_ratio
        self.masking_type = masking_type
        self.num_semantic_tokens = num_semantic_tokens
        self.input_variables = input_variables
        self.timestamp_cols = timestamp_cols
        self.data_paths = data_paths
        self.val_prec = val_prec
        self.test_prec = test_prec
        self.which = type_data  # 'train', 'val', 'test'
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size  # default: non-overlapping
        self.chunk_size = self.patch_size + (self.ratio_patches - 1) * self.stride
        self.all_map = {'train': [], 'val': [], 'test': []}
        self.scaler = StandardScaler()
        self.epochs_completed = 0 
        self.epochs = epochs

        processed_dfs = []
        self.Train_Val_Test_splits = {
            'train': [],
            'val': [],
            'test': []
        }
        sizee = 0
        for path, t_col, input_vars in zip(data_paths, timestamp_cols, input_variables):
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')          
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            processed_dfs.append(df)
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            val_len = int(len(df) * self.val_prec)
            test_len = int(len(df) * self.test_prec)
            train_len = len(df) - val_len - test_len
            # Fit scaler on training portion only, transform all splits
            train_portion = df.iloc[:train_len][input_vars].values
            self.scaler.fit(train_portion)
            df_scaled = self.scaler.transform(df[input_vars].values)
            df_tensor = torch.tensor(df_scaled).float()
            print(f"--- Normalization Check ({self.which}) ---")
            print(f"Mean (should be ~0): {df_tensor.mean().item():.6f}")
            print(f"Std  (should be ~1): {df_tensor.std().item():.6f}")
            train_df, val_df, test_df = torch.split(df_tensor, [train_len, val_len, test_len])
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
        for split_name in ['train', 'val', 'test']:
            for file_idx, tensor in enumerate(self.Train_Val_Test_splits[split_name]):
                num_chunks = (tensor.size(0) + self.chunk_size - 1) // self.chunk_size
                for chunk_idx in range(num_chunks):
                    self.all_map[split_name].append((file_idx, chunk_idx))

    def __len__(self):
        return len(self.all_map[self.which])
   
    def __getitem__(self, idx):
        file_idx, chunk_offset = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]
        start = chunk_offset * self.chunk_size
        end = start + self.chunk_size
        chunk = source_data[start:end]
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(-1)
        if chunk.size(0) < self.chunk_size:
            padding = torch.zeros((self.chunk_size - chunk.size(0), chunk.size(1)))
            chunk = torch.cat([chunk, padding], dim=0)
        patches = [chunk[i * self.stride : i * self.stride + self.patch_size] for i in range(self.ratio_patches)]
        patches_tensor = torch.stack(patches)  # [ratio_patches, patch_size, n_vars]
        masking_avg = self.mask_ratio + 0.3*(self.epochs_completed / 5000)
        context_idx, target_idx = get_mask_style(
            B=1, 
            num_patches=self.ratio_patches, 
            type=self.masking_type, 
            p=self.mask_ratio
        )
        self.epochs_completed += 1
        return patches_tensor, context_idx.squeeze(0), target_idx.squeeze(0)

class ForcastingDataPuller(Dataset):
    def __init__(self,config, which='train'):
        self.patch_size = config["patch_size_forcasting"]
        self.context_size = config["ratio_patches"]
        self.input_variables_forcasting = config["input_variables_forcasting"]
        self.timestamp_cols = config["timestampcols"]
        self.val_prec = config["val_prec_forcasting"]
        self.test_prec = config["test_prec_forcasting"]
        self.data_paths = config["data_paths"]
        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.scaler = StandardScaler()
        self.which = which
        processed_dfs = []
       

        for path, t_col, input_vars in zip(self.data_paths, self.timestamp_cols, self.input_variables_forcasting):
            print(f"DEBUG: Processing dataset {path}, timestamp column: {t_col}, input variables: {input_vars}")
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')          
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            processed_dfs.append(df)
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            val_len = int(len(df) * self.val_prec)
            test_len = int(len(df) * self.test_prec)
            train_len = len(df) - val_len - test_len
            # --- NORMALIZATION START ---
            # Fit scaler only on training portion
            train_portion = df.iloc[:train_len][input_vars].values
            self.scaler.fit(train_portion)
            
            # Transform the whole dataframe
            df_scaled = self.scaler.transform(df[input_vars].values)
            df_tensor = torch.tensor(df_scaled).float()
            # --- NORMALIZATION END ---
            print(f"--- Normalization Check ({self.which}) ---")
            print(f"Mean (should be ~0): {df_tensor.mean().item():.6f}")
            print(f"Std  (should be ~1): {df_tensor.std().item():.6f}")
            print(f"Min: {df_tensor.min().item():.6f}")
            print(f"Max: {df_tensor.max().item():.6f}")
            train_df, val_df, test_df = torch.split(df_tensor, [train_len, val_len, test_len])
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
        self.series = torch.cat(self.Train_Val_Test_splits[self.which], dim=0)
        # Split the entire time series into patches
        self.patches_tensor = self.split_into_patches(self.series, self.patch_size)

    def split_into_patches(self, series, patch_size):
        num_patches = len(series) // patch_size
        patches = [series[i * patch_size:(i + 1) * patch_size] for i in range(num_patches)]
        return torch.stack(patches)  # Shape will be (num_patches, patch_size)

    def __len__(self):
        # Number of available samples based on the context size
        return len(self.patches_tensor) - self.context_size

    def __getitem__(self, idx):
        # Here we ensure that each time we return a context window of 10 patches
        if idx + self.context_size + 1 > len(self.patches_tensor):
            raise IndexError("Index out of range for context window")

        # Get context patches (previous 10) and the target patch (next one)
        context_patches = self.patches_tensor[idx:idx + self.context_size]
        target_patch = self.patches_tensor[idx + self.context_size]

        return context_patches.squeeze(-1), target_patch.squeeze(-1)



class DataPullerVQVAE(Dataset):
    def __init__(self, data_paths, flag='train', chunk_size=128,
                 input_variables=None, timestamp_cols=None, scale=True, 
                 val_prec=0.1, test_prec=0.25):
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.chunk_size = chunk_size # Total window length for the Conv1D layers
        self.input_variables = input_variables
        self.timestamp_cols = timestamp_cols
        self.data_paths = data_paths
        self.scale = scale
        
        self.val_prec = val_prec
        self.test_prec = test_prec
        
        self.scaler = StandardScaler()
        self.all_map = []
        self.data_splits = []
        
        self.__read_data__()

    def __read_data__(self):
        for path, t_col in zip(self.data_paths, self.timestamp_cols):
            df_raw = pd.read_csv(path, parse_dates=[t_col])
            df_raw.sort_values(by=[t_col], inplace=True)
            df_data = df_raw[self.input_variables]
            
            # Split Borders
            val_len = int(len(df_raw) * self.val_prec)
            test_len = int(len(df_raw) * self.test_prec)
            train_len = len(df_raw) - val_len - test_len
            
            # Scale based on training part only [cite: 64]
            if self.scale:
                train_portion = df_data.iloc[:train_len]
                self.scaler.fit(train_portion.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
                
            tensor_data = torch.tensor(data).float()
            
            # Split data [cite: 64]
            train_part, val_part, test_part = torch.split(tensor_data, [train_len, val_len, test_len])
            
            split_map = {'train': train_part, 'val': val_part, 'test': test_part}
            active_tensor = split_map[self.flag]
            
            file_idx = len(self.data_splits)
            self.data_splits.append(active_tensor)
            
            # Create sliding window or non-overlapping chunk map
            num_chunks = (active_tensor.size(0) - self.chunk_size) + 1
            for start_idx in range(0, num_chunks, self.chunk_size): # Non-overlapping
                self.all_map.append((file_idx, start_idx))

    def __len__(self):
        return len(self.all_map)

    def __getitem__(self, index):
        file_idx, start = self.all_map[index]
        source_data = self.data_splits[file_idx]
        
        end = start + self.chunk_size
        chunk = source_data[start:end]

        return chunk.permute(1, 0) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class ForcastingDataPullerDescrete(Dataset):
    def __init__(self,config, which='train'):
        self.patch_size = config["patch_size_forcasting"]
        self.context_size = config["ratio_patches"]
        self.input_variables_forcasting = config["input_variables_forcasting"]
        self.timestamp_cols = config["timestampcols_forcasting"]
        self.val_prec = config["val_prec_forcasting"]
        self.test_prec = config["test_prec_forcasting"]
        self.data_paths = config["path_data_forcasting"]
        self.h = config["horizon_t"]
        self.stride = config.get("stride", config["patch_size_forcasting"])  # default: non-overlapping
        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.which = which
        self.scaler = StandardScaler()

        for run_idx, (path, t_col) in enumerate(zip(self.data_paths, self.timestamp_cols)):
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            val_len = int(len(df) * self.val_prec)
            test_len = int(len(df) * self.test_prec)
            train_len = len(df) - val_len - test_len
            input_vars = self.input_variables_forcasting[run_idx]
            # Fit scaler on training portion only, transform all splits
            train_portion = df.iloc[:train_len][input_vars].values
            self.scaler.fit(train_portion)
            df_scaled = self.scaler.transform(df[input_vars].values)
            df_tensor = torch.tensor(df_scaled).float()
            train_df, val_df, test_df = torch.split(df_tensor, [train_len, val_len, test_len])
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
        self._rebuild()

    def _rebuild(self):
        """Rebuild series and patches_tensor from the current split (self.which)."""
        self.series = torch.cat(self.Train_Val_Test_splits[self.which], dim=0)  # [total_timesteps, num_vars]
        self.patches_tensor = self.split_into_patches(self.series, self.patch_size, self.stride)

    def split_into_patches(self, series, patch_size, stride):
        # series: [timesteps, num_vars] → [num_patches, patch_size, num_vars]
        num_patches = (len(series) - patch_size) // stride + 1
        patches = [series[i * stride : i * stride + patch_size] for i in range(num_patches)]
        return torch.stack(patches)

    def inverse_transform(self, tensor):
        """Inverse-normalize a tensor of shape [..., n_vars] back to original scale.
        The last dimension must be n_vars (same order as input_variables_forcasting).
        """
        scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=tensor.device)
        mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=tensor.device)
        return tensor * scale + mean

    def __len__(self):
        return max(0, len(self.patches_tensor) - self.context_size - self.h)

    def __getitem__(self, idx):
        if idx + self.context_size + self.h > len(self.patches_tensor):
            raise IndexError("Index out of range for context window")

        # context: [context_size, patch_size, num_vars]  — full context, no masking
        context_patches = self.patches_tensor[idx:idx + self.context_size]
        # target: next h patches strictly after context — no overlap with context
        target_patch = self.patches_tensor[idx + self.context_size : idx + self.context_size + self.h]

        return context_patches, target_patch

