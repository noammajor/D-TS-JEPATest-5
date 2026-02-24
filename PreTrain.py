# Pre - Training
import time
import copy
import torch
import os
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
from torchviz import make_dot
from data_loaders.data_puller import DataPullerDJepa
from data_loaders.data_puller import ForcastingDataPuller
from data_loaders.data_puller import ForcastingDataPullerDescrete
from mask_util import apply_mask
from config_files.config_pretrain import config
from config_files.config_full_jepa_classic import  configJEPA
from main.utils import init_weights
from utils.modules import MLP, Block
from pos_embeder import PosEmbeder
from Discrete_JEPA.Descrete_Jepa import DiscreteJEPA
from VQVAE.VQVAE import vqvae
from jepa_classic.JEPA_classic import JEPAClassic
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training Script for JEPA or VQ-VAE")
    parser.add_argument('--model', type=str, required=True, choices=['VQVAE', 'DescreteJEPA', 'JEPA'],
                        help="Choose which model to run: 'VQVAE' or 'DescreteJEPA'")
    parser.add_argument('--skip', type=str, required=True, choices=['true', 'false'],
                        help="Choose which model to run: 'VQVAE' or 'DescreteJEPA'")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = args.model
    #config = prepare_args_pretrain(config)
    # Load Data
    if model == "DescreteJEPA":
        #if skip == 'false':
        train_dataset = DataPullerDJepa(
            data_paths=config["path_data"],
            patch_size=config["patch_size"],
            batch_size=config["batch_size"],
            ratio_patches=config["ratio_patches"],
            mask_ratio=config["mask_ratio"],
            masking_type=config["masking_type"],
            num_semantic_tokens=config["num_semantic_tokens"],
            input_variables=config["input_variables"],
            timestamp_cols=config["timestampcols"],
            type_data='train',
            val_prec=config["val_prec"],
            test_prec=config["test_prec"],
            stride=config["stride"]
        )

        val_dataset = copy.copy(train_dataset)
        val_dataset.which = 'val'

        test_dataset = copy.copy(train_dataset)
        test_dataset.which = 'test'

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        input_dim = len(train_loader.dataset[0][0][0])

        forcasting_data = ForcastingDataPullerDescrete(config)
        val_dataset_forcasting = copy.copy(forcasting_data)
        val_dataset_forcasting.which = 'val'

        test_dataset_forcasting = copy.copy(forcasting_data)
        test_dataset_forcasting.which = 'test'
        train_loader_forcasting = torch.utils.data.DataLoader(forcasting_data, batch_size=config["batch_size"], shuffle=True)
        val_loader_forcasting = torch.utils.data.DataLoader(val_dataset_forcasting, batch_size=config["batch_size"], shuffle=False)
        test_loader_forcasting = torch.utils.data.DataLoader(test_dataset_forcasting, batch_size=config["batch_size"], shuffle=False)

        descrete_jepa_model = DiscreteJEPA(
            config=config,
            input_dim=input_dim,
            num_patches=len(train_loader.dataset[0][0]),
            steps_per_epoch=len(train_loader),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            forcasting_train=train_loader_forcasting,
            forcasting_val=val_loader_forcasting,
            forcasting_test=test_loader_forcasting
        )
        if args.skip == 'false':
            descrete_jepa_model.train_and_evaluate()
        for epoch in range(100, 2301, 100):
            path = f"_epoch{epoch}"
            #descrete_jepa_model.forcasting(path)
            descrete_jepa_model.forcasting_zeroshot(path)
    
    if model == "VQVAE":
        VQVae = vqvae(config)
        save_dir = config.get(config["path_save"], "./results")
        train_dataset, val_dataset, test_dataset = VQVae.create_dataloaders_all_in_one(config)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config["batch_size"],
                                                   shuffle=True,
                                                   num_workers=10,
                                                   drop_last=True)

        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config["batch_size"],
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=config["batch_size"],
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)
        vqvae_config, summary, trained_model = VQVae.start_training(device, config["vqvae_config"], save_dir,  None, train_dataloader, val_dataloader, test_dataloader, args)
    if model == "JEPA":
        #config_j = prepare_args_pretrain(configJEPA)
        train_dataset = DataPullerDJepa(
            data_paths= configJEPA["data_paths"],
            patch_size=configJEPA["patch_size"],
            batch_size=configJEPA["batch_size"],
            ratio_patches=configJEPA["ratio_patches"],
            mask_ratio=configJEPA["mask_ratio"],
            masking_type=configJEPA["masking_type"],
            num_semantic_tokens=configJEPA["num_semantic_tokens"],
            input_variables=configJEPA["input_variables"],
            timestamp_cols=configJEPA["timestampcols"],
            type_data='train',
            val_prec=config["val_prec"],
            test_prec=config["test_prec"],
            stride=configJEPA.get("stride")
        )

        val_dataset = copy.copy(train_dataset)
        val_dataset.which = 'val'

        test_dataset = copy.copy(train_dataset)
        test_dataset.which = 'test'

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        input_dim = len(train_loader.dataset[0][0][0])

        forcasting_data = ForcastingDataPuller(configJEPA)
        val_dataset_forcasting = copy.copy(forcasting_data)
        val_dataset_forcasting.which = 'val'

        test_dataset_forcasting = copy.copy(forcasting_data)
        test_dataset_forcasting.which = 'test'
        train_loader_forcasting = torch.utils.data.DataLoader(forcasting_data, batch_size=config["batch_size"], shuffle=True)
        val_loader_forcasting = torch.utils.data.DataLoader(val_dataset_forcasting, batch_size=config["batch_size"], shuffle=False)
        test_loader_forcasting = torch.utils.data.DataLoader(test_dataset_forcasting, batch_size=config["batch_size"], shuffle=False)


        jepa_model = JEPAClassic(
            config=configJEPA,
            input_dim=input_dim,
            num_patches=len(train_loader.dataset[0][0]),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            forecasting_train=train_loader_forcasting,
            forecasting_val=val_loader_forcasting,
            forecasting_test=test_loader_forcasting
        )
        if args.skip == 'false':
            jepa_model.train_and_evaluate()
        jepa_model.forcast()