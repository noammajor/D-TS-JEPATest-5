import time
import copy
import torch
import os
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import matplotlib.pyplot as plt
from jepa_classic.Decoder_Classic import LinearDecoder
from torchviz import make_dot
from jepa_classic.Encoder_Classic import Encoder
from jepa_classic.Predictor_Classic import Predictor
from data_loaders.data_puller import DataPullerDJepa
from mask_util import apply_mask
from config_files.config_pretrain import config
from main.utils import init_weights
from utils.modules import MLP, Block
from pos_embeder import PosEmbeder
import torch.nn as nn
from config_files.config_full_jepa_classic import configJEPA
import numpy as np

class JEPAClassic(nn.Module):
    def __init__(self,
            config,
            input_dim,
            num_patches,
            train_loader,
            val_loader,
            test_loader,
            forecasting_train,
            forecasting_val,
            forecasting_test):
        super(JEPAClassic, self).__init__()
        self.encoder = Encoder(
            num_patches=configJEPA["ratio_patches"],
            dim_in=input_dim,
            kernel_size=configJEPA["encoder_kernel_size"],
            embed_dim=configJEPA["encoder_embed_dim"],
            embed_bias=configJEPA["encoder_embed_bias"],
            nhead=configJEPA["encoder_nhead"],
            num_layers=configJEPA["encoder_num_layers"],
            jepa=True,
        )
        self.predictor = Predictor(
            num_patches=configJEPA["ratio_patches"],
            encoder_embed_dim=configJEPA["encoder_embed_dim"],
            predictor_embed_dim=configJEPA["predictor_embed"],
            nhead=configJEPA["predictor_nhead"],
            num_layers=configJEPA["predictor_num_layers"],
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # Init weights -- Similar to VJEPA
        for m in self.encoder.modules():
            init_weights(m)

        for m in self.predictor.modules():
            init_weights(m)

        param_groups = [
            {"params": (p for n, p in self.encoder.named_parameters())},
            {"params": (p for n, p in self.predictor.named_parameters())},
        ]

        self.optimizer = torch.optim.AdamW(param_groups, lr=configJEPA["lr"])

        # Initialize the scheduler
        self.scheduler = lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=configJEPA["num_epochs"]
        )
        self.encoder_ema = copy.deepcopy(self.encoder)

        # Stop-gradient step in the EMA
        for p in self.encoder_ema.parameters():
            p.requires_grad = False

        self.path_save = configJEPA["path_save"]
        self.warmup = configJEPA["warmup_ratio"] * configJEPA["num_epochs"]
        # Initialize the EMA Scheduler (parameter m in the paper)
        self.ema_scheduler = (
            configJEPA["ema_momentum"]
            + i
            * (1 - configJEPA["ema_momentum"])
            / (configJEPA["num_epochs"] * configJEPA["ipe_scale"])
            for i in range(int(configJEPA["num_epochs"] * configJEPA["ipe_scale"]) + 1)
        )
        self.best_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Forcasting part
        self.forcast_train = forecasting_train
        self.forcast_val = forecasting_val
        self.forcast_test = forecasting_test
        self.epoch_t = configJEPA["epoch_t"]
        self.Context_t = configJEPA["context_t"]
        self.Patches_t = configJEPA["patches_t"]

        self.encoder_for = Encoder(
            num_patches=configJEPA["ratio_patches"],
            dim_in=input_dim,
            kernel_size=configJEPA["encoder_kernel_size"],
            embed_dim=configJEPA["encoder_embed_dim"],
            embed_bias=configJEPA["encoder_embed_bias"],
            nhead=configJEPA["encoder_nhead"],
            num_layers=configJEPA["encoder_num_layers"],
        )
        # Load Decoder
        self.decoder = LinearDecoder(emb_dim=configJEPA["encoder_embed_dim"], patch_size=32)
        

    def forcast(self):
        name_loader = torch.load(
        self.path_save + "best_model" + ".pt"
        )["encoder"]
        self.encoder_for.load_state_dict(name_loader)
        # 1. Device and Weight Setup
        self.encoder_for.to(self.device)
        self.decoder.to(self.device)
        
        # 2. Train only the Decoder (Frozen Encoder Protocol) [cite: 82]
        param_groups = [{"params": self.decoder.parameters()}]
        optimizer = torch.optim.AdamW(param_groups, lr=configJEPA["lr"]) # Use config lr
        
        for epoch in range(self.epoch_t):
            self.encoder_for.eval()
            self.decoder.train()
            total_loss = 0
            for context_patches, target_patch in self.forcast_train:
                print(f"Context patches shape: {context_patches.shape}, Target patch shape: {target_patch.shape}")
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                # Move data to device
                context_patches = context_patches.to(self.device)
                target_patch = target_patch.to(self.device)
                
                optimizer.zero_grad()
                with torch.no_grad():
                    encoded_patches = self.encoder_for(context_patches)
                
                # Aggregate as per paper (summing tokens)
                summed_embedding = torch.sum(encoded_patches, dim=1)
                predicted_next_patch = self.decoder(summed_embedding)
                
                loss = torch.nn.functional.mse_loss(predicted_next_patch, target_patch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} - Training Loss: {total_loss/len(self.forcast_train)}")

        # 3. Autoregressive Inference (The "Loop Way") [cite: 101]
        predictions = []
        # Calculate num_steps based on available test data
        num_steps = configJEPA["horizon_t"]
        
        # Initialize first context window
        total_context_steps = self.Context_t * self.Patches_t 

        # Access the raw series from the test dataset
        # Shape: [320, 1]
        test_series = self.forcast_test.dataset.series[:total_context_steps]

        # Replicate the paper's reshape: [10 patches, 32 steps, 1 feature]
        # Then add Batch dimension: [1, 10, 32, 1]
        current_context = (
            test_series
            .reshape(self.Patches_t, self.Context_t, 1)
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        
        self.encoder_for.eval()
        self.decoder.eval()

        with torch.no_grad():
            for _ in range(num_steps):
                encoded = self.encoder_for(current_context)
                summed = torch.sum(encoded, dim=1)
                pred_patch = self.decoder(summed) # Shape: [1, context_size]
                
                predictions.append(pred_patch.squeeze(0).cpu())
                # Add Patch dimension (1) and Feature dimension (-1)
                new_patch = pred_patch.unsqueeze(1).unsqueeze(-1) 
                
                # Drop index 0 (oldest), append new_patch at index 9 (newest)
                current_context = torch.cat([current_context[:, 1:], new_patch], dim=1)

        # 4. Evaluation [cite: 115]
        predictions = torch.cat(predictions, dim=0)
        test_series = self.forcast_test.dataset.series

        # 2. Define the window for ground truth
        # The context used the first 320 steps (Context_t * Patches_t)
        # So the real future starts at index 320
        start_idx = self.Context_t * self.Patches_t
        end_idx = start_idx + len(predictions)

        # 3. Extract the real values and move them to your device
        # .squeeze() ensures it's a flat 1D line for plotting
        real_values = test_series[start_idx:end_idx].to(self.device).squeeze()
        #real_values = self.forcast_test[self.Context_t * self.Patches_t : self.Context_t * self.Patches_t + len(predictions)]
        print(f"Predictions shape: {predictions.shape}, Real values shape: {real_values.shape}")
        # Ensure shapes match for MSE calculation
        loss_test = torch.nn.functional.mse_loss(predictions, real_values.cpu())
        print(f"Long-term forecasting test MSE: {loss_test.item()}")
        # --- Visualizing Results ---
        plt.figure(figsize=(15, 6))
        # Plot real values in blue
        plt.plot(real_values.cpu().numpy(), label='Real Values', color='blue', alpha=0.7)
        # Plot predicted values in red dashed line
        plt.plot(predictions.numpy(), label='TS-JEPA Predictions', color='red', linestyle='--', alpha=0.9)
        
        plt.title(f"TS-JEPA Long-term Forecasting Comparison (MSE: {loss_test.item():.4f})")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Save the plot
        plot_path = os.path.join(self.path_save, "forecasting_results.png")
        plt.savefig(plot_path)
        print(f"Forecasting graph saved to: {plot_path}")
        plt.show()





    def loss_pred(self, pred, target_ema):
        loss = 0.0
        for pred_i, target_ema_i in zip(pred, target_ema):
            loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
        loss /= len(pred)
        return loss


    def save_model(self, epoch):
        # Ensure the directory exists before saving
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)
        
        save_dict = {"encoder": self.encoder.state_dict(), "epoch": epoch}

        try:
            path_name = self.path_save + "best_model" + ".pt"
            torch.save(save_dict, path_name)
        except:
            print("Problem saving checkpoint")
    def save_latent_samples(self, context_data, target_data, epoch, batch_idx):
        folder = "latent_analysis_classic"
        os.makedirs(folder, exist_ok=True)
        
        # Convert to numpy
        ctx = context_data.detach().cpu().numpy()
        tgt = target_data.detach().cpu().numpy()
        # Save as numpy files
        np.save(f"{folder}/ctx_epoch{epoch}_b{batch_idx}.npy", ctx)
        np.save(f"{folder}/tgt_epoch{epoch}_b{batch_idx}.npy", tgt)
        
        print(f"Saved latent samples for epoch {epoch}")


    # Define the custom learning rate schedule
    def lr_lambda(epoch):
        start_lr = configJEPA["lr"]
        end_lr = configJEPA["end_lr"]
        if epoch < configJEPA["num_epochs"]:
            return start_lr + (end_lr - start_lr) * (epoch / (configJEPA["num_epochs"] - 1))
        else:
            return end_lr

    def train_and_evaluate(self):
        min_val_loss = float('inf')
        self.encoder = self.encoder.to(self.device)
        self.predictor = self.predictor.to(self.device)
        self.encoder_ema = self.encoder_ema.to(self.device)
        for p in self.encoder_ema.parameters():
            p.requires_grad = False
        num_batches = len(self.train_loader)

        total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0
        self.save_model(0)

        for epoch in range(configJEPA["num_epochs"]):
            print(f"Starting Epoch {epoch}/{configJEPA['num_epochs']}")
            m = next(self.ema_scheduler)
            self.encoder.train()
            self.predictor.train()

            for patches, masks, non_masks in self.train_loader:
                self.optimizer.zero_grad()
                #self.scheduler.step()
                patches = patches.to(self.device)
                masks = masks.to(self.device)
                non_masks = non_masks.to(self.device)

                # Predict targets
                with torch.no_grad():
                    target_ema = self.encoder_ema(patches)
                    target_ema = F.layer_norm(
                    target_ema, (target_ema.size(-1),))
                    target_ema = apply_mask(target_ema, masks)

                # Encode inputs
                tokens = self.encoder(patches, mask=non_masks)

                # Make predictions
                pred = self.predictor(
                    tokens, mask=masks, non_masks=non_masks
                )
                self.save_latent_samples(tokens, target_ema, epoch, 0)

                # Compute loss
                loss = self.loss_pred(pred, target_ema)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                with torch.no_grad():
                    for param_q, param_k in zip(
                        self.encoder.parameters(), self.encoder_ema.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                total_loss += loss.item()
                print(f"Batch Loss: {loss.item():.4f}")

            total_loss = total_loss / num_batches
            print(f"Epoch {epoch} completed. Average Loss: {total_loss:.4f}")
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, lr: {self.optimizer.param_groups[0]['lr']:.3g} - JEPA Loss: {total_loss:.4f},")
            loss = self.evaluate()
            print(f"Validation Loss after Epoch {epoch}: {loss:.4f}")
            if loss< min_val_loss:
                print(f"New best model found at epoch {epoch} with val loss {loss:.4f}")
                min_val_loss = loss
                self.best_model = {
                    "encoder": copy.deepcopy(self.encoder.state_dict()),
                    "predictor": copy.deepcopy(self.predictor.state_dict()),
                    "epoch": epoch,
                    "val_loss": min_val_loss
                }
                self.save_model(epoch)
    
    def evaluate(self):
        self.encoder.eval()
        self.predictor.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        with torch.no_grad():
            for patches, masks, non_masks in self.val_loader:
                patches = patches.to(self.device)
                masks = masks.to(self.device)
                non_masks = non_masks.to(self.device)

                # Predict targets
                target_ema = self.encoder_ema(patches)
                target_ema = F.layer_norm(
                    target_ema, (target_ema.size(-1),)
                )  # normalize over feature-dim  [B, N, D]
                target_ema = apply_mask(target_ema, masks)

                # Encode inputs
                tokens = self.encoder(patches, mask=non_masks)

                # Make predictions
                pred = self.predictor(tokens, mask=masks, non_masks=non_masks)

                # Compute loss
                loss = self.loss_pred(pred, target_ema)

                total_loss += loss.item()
            return total_loss / num_batches