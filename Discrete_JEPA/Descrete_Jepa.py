import time
import copy
import torch
import os
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from Discrete_JEPA.Encoder import Encoder
from Discrete_JEPA.Predictors import DiscreteJEPAPredictor as Predictor
from data_loaders.data_puller import DataPullerDJepa
from data_loaders.data_puller import ForcastingDataPullerDescrete
from mask_util import apply_mask
from config_files.config_pretrain import config
from main.utils import init_weights
from utils.modules import MLP, Block
from pos_embeder import PosEmbeder
import torch.nn as nn
from making_style import get_mask_style
from Discrete_JEPA.Decoder import LinearDecoder
from Discrete_JEPA.VQ import *

class DiscreteJEPA(nn.Module):
    def __init__(self, config, input_dim, num_patches, steps_per_epoch, train_loader, val_loader, test_loader, forcasting_train, forcasting_val, forcasting_test):
        super(DiscreteJEPA, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.encoder = Encoder(
        num_patches=len(self.train_loader.dataset[0][0]),
        num_semantic_tokens=config["num_semantic_tokens"],
        dim_in=input_dim,
        kernel_size=config["kernel_size"],
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["embed_bias"],
        nhead=config["nhead"],
        num_layers=config["num_encoder_layers"],
        mlp_ratio=config["mlp_ratio"],
        qkv_bias=config["qkv_bias"],
        qk_scale=config["qk_scale"],
        drop_rate=config.get("encoder_drop_rate", 0.0),
        attn_drop_rate=config.get("encoder_attn_drop_rate", 0.0),
        norm_layer=torch.nn.LayerNorm,
        jepa=False,
        embed_activation=torch.nn.GELU(),
        codebook_size=config["codebook_size"],
        commitment_cost=config["commitment_cost"]
        )
        self.predictor = Predictor(
            num_patches=len(self.train_loader.dataset[0][0]),
            num_semantic_tokens=config["num_semantic_tokens"],
            embed_dim=config["encoder_embed_dim"],
            predictor_embed_dim=config["predictor_embed_dim"],
            config=config
        )
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=config["codebook_size"],
            embedding_dim=config["encoder_embed_dim"],
            commitment_cost=config["commitment_cost"],
            ema_decay=config.get("vq_ema_decay", 0.99)
        )

        for m in self.encoder.modules():
            init_weights(m)

        for m in self.predictor.modules():
            init_weights(m)
        # NOTE: Do NOT call init_weights on vector_quantizer.
        # VQ.py initializes codebook on the unit sphere (L2 normalized).
        # init_weights would override it with wrong distribution.

        # EMA (teacher) vector quantizer - deepcopy preserves the unit-sphere init
        self.vector_quantizer_ema = copy.deepcopy(self.vector_quantizer)
        self.vector_quantizer_ema.eval()  # Teacher VQ: always eval (never run EMA updates)

        encoder_params = list(self.encoder.parameters())
        codebook_params = list(self.vector_quantizer.parameters())
        other_params_pred = list(self.predictor.parameters())

        # Switched to AdamW for better transformer training stability
        self.optimizer = torch.optim.AdamW([
            {
                "params": encoder_params,
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
                "betas": (0.9, 0.999)
            },
            {
                "params": other_params_pred,
                "lr": config["lr_pred"],
                "weight_decay": config["weight_decay_pred"],
                "betas": (0.9, 0.999)
            },
            {
                "params": codebook_params,
                "lr": config.get("codebook_lr", config["lr"] * 2),
                "weight_decay": 0.0,
                "betas": (0.9, 0.999)
            }
        ])
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.config["num_epochs"] * self.steps_per_epoch

        # mimicing the D-JEPA paper
        #self.scheduler = lr_scheduler.OneCycleLR(
        #self.optimizer,
        #max_lr=self.config["lr"],             # The peak learning rate from your config
        #total_steps=self.total_steps,
        #pct_start=0.05,                  # 5% warmup as per TD-JEPA
        #anneal_strategy='cos',           # Cosine decay is standard used in D-JEPA]
        #div_factor=10.0,                 # changed from 25 to 10
        #final_div_factor=1e4             # defualt
        #)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        self.optimizer,
        max_lr=[config["lr"], config["lr_pred"], config.get("codebook_lr", config["lr"] * 2)],
        epochs=config["num_epochs"],
        steps_per_epoch=len(self.train_loader),
        pct_start=0.1,    # Spend 10% of time warming up
        anneal_strategy='cos',
        div_factor=25.0,  # Initial lr = max_lr / 25
        final_div_factor=1000.0 # Final lr = max_lr / 1000
        )
        
        self.encoder_ema = copy.deepcopy(self.encoder)
        self.encoder_ema.jepa = True
        self.encoder_ema.type_enc = "target"
        self.codebooksize = config["codebook_size"]
        self.checkpoint_save = self.config["checkpoint_save"]
        self.checkpoint_print = self.config["checkpoint_print"]
        self.path_save = self.config["path_save"]
        self.clip_grad = self.config["clip_grad"]
        self.warmup = self.config["warmup_ratio"] * self.config["num_epochs"]
        self.ema_scheduler = (
            self.config["ema_momentum"]
            + i
            * (0.999 - self.config["ema_momentum"])
            / (self.config["num_epochs"] * len(train_loader))
            for i in range(int(self.config["num_epochs"] *len(train_loader)) + 1)
        )

        self.best_model = None
        self.log_file = "perplexity_log.csv"
        self.logsss ="info.csv"
        with open(self.log_file, "w") as f:
            f.write("epoch,step,type,perplexity\n")
        with open(self.logsss, "w") as g:
            g.write("epoch,step,type,perplexity\n")
        
        
        #forcasting
        self.Decoder_patches = LinearDecoder(emb_dim=config["predictor_embed_dim"], patch_size=config["patch_size"])
        self.Decoder_semantic = LinearDecoder(emb_dim=config["predictor_embed_dim"], patch_size=config["patch_size"])
        self.forcast_train = forcasting_train
        self.forcast_val = forcasting_val
        self.forcast_test = forcasting_test
        self.epoch_t = config["epoch_t"]
        self.Context_t = config["context_t"]
        self.Patches_to_forcast = config["patches_to_forcast"]
        self.patches_size_forecasting = config["patch_size_forcasting"]
        self.h = config["horizon_t"]

        self.encoder_for = Encoder(
            num_patches=config["ratio_patches"],
            dim_in=input_dim,
            kernel_size=config["encoder_kernel_size"],
            embed_dim=config["encoder_embed_dim"],
            embed_bias=config["encoder_embed_bias"],
            nhead=config["nhead"],
            num_layers=config["num_encoder_layers"],
            num_semantic_tokens=config["num_semantic_tokens"],
            type_enc = "target"
        )
        self.predictor_for = Predictor(
            num_patches=len(self.train_loader.dataset[0][0]),
            num_semantic_tokens=config["num_semantic_tokens"],
            embed_dim=config["encoder_embed_dim"],
            predictor_embed_dim=config["predictor_embed_dim"],
            config=config,
        )


    def verify_encoder(self,model):
        print("\n--- Encoder Integrity Check ---")
        total_params = 0
        for name, param in model.named_parameters():
            p_min = param.min().item()
            p_max = param.max().item()
            p_mean = param.mean().item()
            p_std = param.std().item()

            # Check for NaN or Inf (The "Death" Check)
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"❌ CRITICAL ERROR: {name} contains NaNs or Infs!")

            # Check for Dead Layers (Zero Variance)
            if p_std < 1e-9:
                print(f"⚠️ WARNING: {name} appears to be dead (std ~ 0).")

            # Check for Saturated Layers (Extreme values)
            if abs(p_max) > 100 or abs(p_min) > 100:
                print(f"⚠️ WARNING: {name} has very high magnitudes (Possible explosion).")

            print(f"{name:40} | Mean: {p_mean:8.4f} | Std: {p_std:8.4f} | Range: [{p_min:6.2f}, {p_max:6.2f}]")
        print("--- Check Complete ---\n")
    def forcasting(self, path):
        epoch_tag = path  # e.g. "_epoch100"
        checkpoint_path = f"{self.path_save}{path}best_model.pt"
        print(f"path:{checkpoint_path}")
        name_loader = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.encoder_for.to(self.device)
        self.Decoder_patches.to(self.device)
        self.Decoder_semantic.to(self.device)
        self.predictor_for.to(self.device)
        self.vector_quantizer.to(self.device)
        self.vector_quantizer_ema.to(self.device)
        self.encoder_for.load_state_dict(name_loader["encoder"], strict=False)
        self.predictor_for.load_state_dict(name_loader["predictor"], strict=False)

        # Load vector_quantizer if available in checkpoint
        if "vector_quantizer" in name_loader:
            self.vector_quantizer.load_state_dict(name_loader["vector_quantizer"])
            print("Loaded vector_quantizer (student) from checkpoint")
        else:
            print("Warning: vector_quantizer not found in checkpoint, using initialized weights")

        # Load VQ EMA (teacher codebook)
        if "vector_quantizer_ema" in name_loader:
            self.vector_quantizer_ema.load_state_dict(name_loader["vector_quantizer_ema"])
            print("Loaded vector_quantizer_ema (teacher) from checkpoint")
        else:
            print("Warning: vector_quantizer_ema not found in checkpoint, using student VQ weights")
            self.vector_quantizer_ema.load_state_dict(self.vector_quantizer.state_dict())

        file_keys = set(name_loader["encoder"].keys())
        live_keys = set(self.encoder_for.state_dict().keys())
        print(f"Missing in file: {live_keys - file_keys}")
        print(f"Extra in file: {file_keys - live_keys}")
        self.verify_encoder(self.encoder_for)

        print(f"Loaded Encoder Weight Sum: {sum(p.sum() for p in self.encoder_for.parameters())}")

        param_groups = [{"params": self.Decoder_patches.parameters()}, {"params": self.Decoder_semantic.parameters()}]
        optimizer = torch.optim.AdamW(param_groups, lr=config["lr_forcasting"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, self.epoch_t // 4), T_mult=1, eta_min=1e-6)
        for epoch in range(self.epoch_t):
            self.encoder_for.eval()
            self.predictor_for.eval()
            self.Decoder_patches.train()
            self.Decoder_semantic.train()
            total_loss=0.0
            total_loss_sem = 0.0
            total_loss_patch = 0.0
            for context_patches, target_patch in self.forcast_train:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                # Move data to device
                context_patches = context_patches.to(self.device)
                target_patch = target_patch.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    encoder_out = self.encoder_for(context_patches)
                    encoder_patches = encoder_out["data_patches"]
                    encoder_semantic = encoder_out["quantized_semantic"]
                    vq_loss, encoder_semantic,soft_probs,perplexity, indices, _, _ = self.vector_quantizer(encoder_semantic)
                    print(f"{encoder_patches.mean().item():.4f}, mean semantic:{encoder_semantic.mean().item():.4f}")
                    enc_std=encoder_patches.std(dim=-1, keepdim=True, correction=0).mean().item()
                    enc_s_std = encoder_semantic.std(dim=-1, keepdim=True, correction=0).mean().item()
                    predicted_latent_patch = self.predictor_for(encoder_patches, target_mask=None, task='P2P')
                    predicted_latent_semantic = self.predictor_for(encoder_semantic, target_mask=None, task='P2S')
                    predicted_patches_from_semantic = self.predictor_for(encoder_semantic, target_mask=None, task='S2P')
                predicted_next_patch = self.Decoder_patches(predicted_latent_patch)
                predicted_next_tokenic = self.Decoder_semantic(predicted_latent_semantic)
                predicted_next_patch_from_semantic = self.Decoder_patches(predicted_patches_from_semantic)
                loss_sem = torch.nn.functional.mse_loss(predicted_next_patch_from_semantic, target_patch)
                loss_patch = torch.nn.functional.mse_loss(predicted_next_patch, target_patch)
                total_loss_sem += loss_sem
                total_loss_patch += loss_patch
                loss = loss_sem + loss_patch
                total_loss += loss
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} - Training Loss: {total_loss/len(self.forcast_train)}")
        predictions_p2p = []
        predictions_s2p = []
        predictions_p2s = []
        total_context_steps = self.Context_t * self.patches_size_forecasting
        test_series = self.forcast_test.dataset.series[:total_context_steps]

        current_context = (
            test_series
            .reshape(self.patches_size_forecasting, self.Context_t, 1)
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        
        self.encoder_for.eval()
        self.Decoder_patches.eval()
        self.Decoder_semantic.eval()
        self.predictor_for.eval()

        with torch.no_grad():
            for step in range(self.Patches_to_forcast):
                encoder_out = self.encoder_for(current_context)
                encoder_patches = encoder_out["data_patches"]
                encoder_semantic = encoder_out["quantized_semantic"]
                vq_loss, encoder_semantic,soft_probs,perplexity, indices,_,_ = self.vector_quantizer(encoder_semantic)
                #predict
                predicted_latent_patch = self.predictor_for(encoder_patches, target_mask=None, task='P2P')
                predicted_latent_semantic = self.predictor_for(encoder_patches, target_mask=None, task='P2S')
                predicted_latent_s2p = self.predictor_for(encoder_semantic, target_mask=None, task='S2P')
                #vals
                predicted_next_patch = self.Decoder_patches(predicted_latent_patch)
                predicted_next_tokenic = self.Decoder_semantic(predicted_latent_semantic)
                predicted_next_patch_from_semantic = self.Decoder_patches(predicted_latent_s2p)
                # 4. Store for comparison graphs
                predictions_p2s.append(predicted_next_tokenic.squeeze(0).cpu())
                predictions_s2p.append(predicted_next_patch_from_semantic.squeeze(0).cpu())
                predictions_p2p.append(predicted_next_patch.squeeze(0).cpu())
                time_step_patch = predicted_next_patch[:, 0, :]
                time_step_patch = time_step_patch.unsqueeze(1).unsqueeze(-1)
                current_context = torch.cat([current_context[:, 1:], time_step_patch], dim=1)

        predict_p2p = torch.cat(predictions_p2p, dim=0)
        predict_p2s = torch.cat(predictions_p2s, dim=0)
        predict_s2p = torch.cat(predictions_s2p, dim=0)

        test_series = self.forcast_test.dataset.series

        # 2. Define the window for ground truth
        # The context used the first 320 steps (Context_t * Patches_t)
        # So the real future starts at index 320
        start_idx = self.Context_t * self.patches_size_forecasting
        total_forecasted_steps = predict_p2p.shape[0] * self.patches_size_forecasting
        end_idx = start_idx + total_forecasted_steps

        # 3. Extract the real values and move them to your device
        # .squeeze() ensures it's a flat 1D line for plotting
        real_values = test_series[start_idx:end_idx].to(self.device)
        predict_p2p = predict_p2p.reshape(-1)
        predict_s2p = predict_s2p.reshape(-1)
        truth_np = real_values.cpu().numpy()

        loss_test_p2p = torch.nn.functional.mse_loss(predict_p2p, real_values.cpu())
        #loss_test_p2s = torch.nn.functional.mse_loss(predict_p2s, real_values.cpu())
        loss_test_s2p = torch.nn.functional.mse_loss(predict_s2p, real_values.cpu())
        num_s = self.Patches_to_forcast * self.patches_size_forecasting
        print(f"Test MSE Loss - Patch-to-Patch (P2P): {loss_test_p2p.item()}, loss normalized per step: {loss_test_p2p.item()/num_s}")
        #print(f"Test MSE Loss - Patch-to-Sequence (P2S): {loss_test_p2s.item()}")
        print(f"Test MSE Loss - Sequence-to-Patch (S2P): {loss_test_s2p.item()}, loss normalized per step: {loss_test_p2p.item()/num_s}")

        # 2. Update dictionary with parentheses in every key to satisfy .split('(')
        predict_types = {
            "Raw Patch-to-Patch (P2P)": predict_p2p.cpu().numpy(),
            "Raw Sequence-to-Patch (S2P)": predict_s2p.cpu().numpy(),
        }
        
        colors = ['green', 'red']
        
        for (title, pred_data), color in zip(predict_types.items(), colors):
            plt.figure(figsize=(15, 5))
            
            # Use Normalized Truth if the title says "Normalized", else use Raw Truth
            plt.plot(truth_np, label='Raw Ground Truth', color='black', alpha=0.5, linewidth=2)
            plt.ylabel("Original OT Value")

            # Plot Prediction
            plt.plot(pred_data, label=f'TS-JEPA {title}', color=color, linestyle='--', alpha=0.9)
            
            plt.title(f"Comparison: {title}")
            plt.xlabel("Future Time Steps (720 Horizon)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # This will now work for all 4 keys!
            # It extracts 'p2p' or 's2p' from the parentheses
            slug = title.split('(')[1][:3].lower()
            
            # Add a prefix so 'Raw' and 'Normalized' don't overwrite each other
            prefix = "norm" if "Normalized" in title else "raw"
            save_name = f"compare_{prefix}_{slug}{epoch_tag}.png"
            path_s = os.path.join(self.path_save, "output_model")

            # 2. CRITICAL: Create the directory if it doesn't exist
            # Without this, savefig will throw a FileNotFoundError
            os.makedirs(path_s, exist_ok=True)
            full_save_path = os.path.join(path_s, save_name)
            # 4. Save the figure
            plt.savefig(full_save_path)
            plt.close()

    def forcasting_zeroshot(self, path):
        """Non-autoregressive forecasting: slides real context forward, no prediction feedback.
        Runs TWICE: first with random encoder (baseline), then with trained encoder."""
        epoch_tag = path
        checkpoint_path = f"{self.path_save}{path}best_model.pt"

        # Load checkpoint for normalization stats and later use
        name_loader = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        # Data is normalized by the dataset's StandardScaler — no model-level stats needed

        # Run twice: random encoder, then trained encoder
        for run_type in ['RANDOM', 'TRAINED']:
            print(f"\n=== Zero-Shot Forecasting ({run_type}) ===")

            # Move models to device
            self.encoder_for.to(self.device)
            self.vector_quantizer.to(self.device)
            self.vector_quantizer_ema.to(self.device)
            h = self.config["horizon_t"]

            if run_type == 'TRAINED':
                # Load trained weights from checkpoint
                print(f"Loading checkpoint: {checkpoint_path}")
                self.encoder_for.load_state_dict(name_loader["encoder"], strict=False)
                if "vector_quantizer" in name_loader:
                    self.vector_quantizer.load_state_dict(name_loader["vector_quantizer"])
                if "vector_quantizer_ema" in name_loader:
                    self.vector_quantizer_ema.load_state_dict(name_loader["vector_quantizer_ema"])
                else:
                    self.vector_quantizer_ema.load_state_dict(self.vector_quantizer.state_dict())
            # For RANDOM: models already randomly initialized from __init__
            else:
                # Re-initialize to fresh random weights every time
                # (previous TRAINED run overwrites these, so we must reset)
                for m in self.encoder_for.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                for m in self.vector_quantizer.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                for m in self.vector_quantizer_ema.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()

            # Linear heads: whole context → h future patches (no predictor, no pooling)
            # Patch head: all ratio_patches embeddings flattened → h * patch_size
            # Semantic head: all num_semantic_tokens embeddings flattened → h * patch_size
            head_patch = nn.Linear(
                config["ratio_patches"] * config["encoder_embed_dim"],
                h * self.patches_size_forecasting
            ).to(self.device)
            head_sem = nn.Linear(
                config["num_semantic_tokens"] * config["encoder_embed_dim"],
                h * self.patches_size_forecasting
            ).to(self.device)

            param_groups = [
                {"params": head_patch.parameters()},
                {"params": head_sem.parameters()}
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=config["lr_forcasting"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, self.epoch_t // 4), T_mult=1, eta_min=1e-6)
            for epoch in range(self.epoch_t):
                self.encoder_for.eval()
                self.vector_quantizer.eval()
                head_patch.train()
                head_sem.train()
                total_loss = 0.0
                for context_patches, target_patch in self.forcast_train:
                    context_patches = context_patches.to(self.device)
                    target_patch = target_patch.to(self.device)
                    B, h_t, P_L, n_v = target_patch.shape
                    target_patch = target_patch.permute(0, 3, 1, 2).reshape(B * n_v, h_t, P_L)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        encoder_out = self.encoder_for(context_patches)
                        encoder_patches = encoder_out["data_patches"]
                        encoder_semantic = encoder_out["quantized_semantic"]
                        vq_loss, encoder_semantic, soft_probs, perplexity, indices,_,_ = self.vector_quantizer(encoder_semantic)
                        # No predictor: flatten full context → single linear layer
                        flat_patch = encoder_patches.flatten(1)    # [B*n_v, ratio_patches * embed_dim]
                        flat_sem = encoder_semantic.flatten(1)      # [B*n_v, num_sem_tokens * embed_dim]
                    pred_patch = head_patch(flat_patch).view(B * n_v, h, self.patches_size_forecasting)
                    pred_s2p = head_sem(flat_sem).view(B * n_v, h, self.patches_size_forecasting)

                    
                    loss = F.mse_loss(pred_patch, target_patch) + F.mse_loss(pred_s2p, target_patch)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if epoch % 10 == 0:
                    print(f"[{run_type}] Epoch: {epoch} - Loss: {total_loss/len(self.forcast_train):.4f}")

            # --- Zero-shot inference: no autoregressive feedback ---
            num_ctx = self.patches_size_forecasting
            data_pp = self.Context_t #patches

            self.encoder_for.eval()
            head_patch.eval()
            head_sem.eval()

            with torch.no_grad():
                for context_patches, target_patch in self.forcast_test:
                    target_patch = target_patch.to(self.device)
                    context_patches = context_patches.to(self.device)
                    B, h_t, P_L, n_v = target_patch.shape
                    target_patch = target_patch.permute(0, 3, 1, 2).reshape(B * n_v, h_t, P_L)
                    target_patch_normalizes = target_patch
                    # Target is strictly after context — full context is visible
                    encoder_out = self.encoder_for(context_patches)
                    encoder_patches = encoder_out["data_patches"]
                    encoder_semantic = encoder_out["quantized_semantic"]
                    vq_loss, encoder_semantic, soft_probs, perplexity, indices,_,_ = self.vector_quantizer(encoder_semantic)
                    # No predictor: flatten full context → linear heads
                    flat_patch = encoder_patches.flatten(1)
                    flat_sem = encoder_semantic.flatten(1)
                    pred_p2p = head_patch(flat_patch).view(B * n_v, h, self.patches_size_forecasting)
                    pred_s2p = head_sem(flat_sem).view(B * n_v, h, self.patches_size_forecasting)
                    break
                else:
                    print(f"WARNING: forcast_test is empty (len={len(self.forcast_test.dataset)}), skipping evaluation.")
                    continue

            # target_patch: [B, h, P_L, n_vars] (4D from dataloader)
            # pred_p2p / pred_s2p: [B*n_vars, h, P_L] (3D from channel-independence encoder+decoder)
            B, n_vars = 32, 21
            h_t, P_L = self.h, self.patches_size_forecasting
            pred_p2p_4d = pred_p2p.view(B, n_vars, h_t, P_L)
            pred_s2p_4d = pred_s2p.view(B, n_vars, h_t, P_L)
            # Reshape predictions: [B*n_vars, h, P_L] → [B, n_vars, h, P_L]
            pred_p2p_4d = pred_p2p.view(B, n_vars, h_t, P_L)
            pred_s2p_4d = pred_s2p.view(B, n_vars, h_t, P_L)
            target_norm_4d = target_patch_normalizes.view(B, n_vars, h_t, P_L)  # [B, n_vars, h, P_L]
            target_4d = target_patch.view(B, n_vars, h_t, P_L)            # [B, n_vars, h, P_L]

            # Aggregate MSE over all samples and variables
            norm_lossP2P = F.mse_loss(pred_p2p_4d.cpu(), target_norm_4d.cpu())
            norm_lossS2P = F.mse_loss(pred_s2p_4d.cpu(), target_norm_4d.cpu())
            mix_4d = (pred_p2p_4d + pred_s2p_4d) / 2.0
            norm_mixloss = F.mse_loss(mix_4d.cpu(), target_norm_4d.cpu())
            num_s = h * self.patches_size_forecasting
            print(f"[{run_type}] Test MSE P2P Norm:{norm_lossP2P.item():.4f}, Test MSE S2P Norm:{norm_lossS2P.item():.4f}, mix: {norm_mixloss.item():.4f}")

            # Plot each variable separately for the first sample in the batch
            sample = 0
            path_s = os.path.join(self.path_save, "output_model")
            os.makedirs(path_s, exist_ok=True)
            for var_idx in range(n_vars):
                # Flatten h patches into a continuous time axis: [h, P_L] → [h*P_L]
                gt = target_norm_4d[sample, var_idx].reshape(-1).cpu().numpy()
                p2p = pred_p2p_4d[sample, var_idx].reshape(-1).cpu().numpy()
                s2p = pred_s2p_4d[sample, var_idx].reshape(-1).cpu().numpy()
                mixed = (p2p + s2p) / 2.0

                plt.figure(figsize=(15, 5))
                plt.plot(gt, label='Ground Truth', color='black', alpha=0.7, linewidth=2)
                plt.plot(p2p, label='P2P', color='blue', linestyle='--', alpha=0.9)
                plt.plot(s2p, label='S2P', color='orange', linestyle='--', alpha=0.9)
                plt.plot(mixed, label='Mixed', color='red', linestyle='--', alpha=0.9)
                plt.title(f"Zero-Shot {run_type} — Variable {var_idx} ({h_t * P_L} steps)")
                plt.xlabel("Time Steps")
                plt.ylabel("Normalized Value")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)

                save_name = f"{run_type.lower()}_var{var_idx}{epoch_tag}.png"
                plt.savefig(os.path.join(path_s, save_name))
                plt.close()

            print(f"[{run_type}] Plots saved to {os.path.join(self.path_save, 'output_model')}")



    def compute_var_loss(self, z):
        eps = 1e-4
        std_z = torch.sqrt(z.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(1.0 - std_z))
        return var_loss
    def calculate_perplexity_regularizer(self, context_ppl, target_ppl):
        eps = 1e-8
        loss = -torch.log(context_ppl + target_ppl + eps)
        return loss
        
    def _calculate_vicreg_loss(self, x: torch.Tensor):
        std = torch.sqrt(x.var(dim=0, unbiased=True) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x_flat = x.reshape(-1, num_features) 
        x_centered = x_flat - x_flat.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
        cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / num_features
        
        return var_loss, cov_loss
    def _calculate_token_diversity_loss(self, semantic_tokens: torch.Tensor):
        """Penalizes semantic tokens for being too similar within each sample.
        semantic_tokens: [B, num_semantic_tokens, D]
        Returns mean off-diagonal cosine similarity (lower = more diverse).
        """
        normed = F.normalize(semantic_tokens, dim=-1)  # [B, S, D]
        sim = torch.bmm(normed, normed.transpose(1, 2))  # [B, S, S]
        S = semantic_tokens.shape[1]
        mask = ~torch.eye(S, device=semantic_tokens.device).bool()
        diversity_loss = sim[:, mask].mean()
        return diversity_loss
        
    def compute_discrete_jepa_loss(
    self,
    context_out,
    target_out,
    masks,
    epoch,
    lambda_weights={'s2p': 1.0, 'p2s': 1.0, 'p2p': 1.0},
    beta_vq=1.0,
    current_global_step=0,
    total_training_steps=100000,
    vq_warmup = 0.15,
    batch_idx=0
    ):
        #warm up for VQ loss
        #vq_weight = min(1.0, current_global_step /int(vq_warmup * total_training_steps))
        z_s_target = target_out["quantized_semantic"]
        # z_p_target: only the masked patch embeddings [B*F, num_masked, D] (apply_mask was called before)
        z_p_target = target_out["data_patches"]
        z_s_context = context_out["quantized_semantic"]
        # z_p_context: context-only embeddings; encoder never saw the missing patches
        z_p_context = context_out["data_patches"]

        # Apply VQ consistently (removed conditional epoch logic)
        # Context: Student VQ learns from student encoder (gradients flow to codebook)
        l_vq, z_s_context, probs, perplexity, indices, active_codes, usage_pct = self.vector_quantizer(z_s_context)
        # Target: EMA VQ (teacher codebook, updated via momentum) on detached EMA encoder output
        _, z_s_target, _, _, _,_,_ = self.vector_quantizer_ema(z_s_target.detach())

        # MLP predicts all num_patches positions; apply_mask selects the masked ones to match the target
        pred_s2p = apply_mask(self.predictor(z_s_context, task='S2P'), masks)
        l_s2p = F.mse_loss(pred_s2p, z_p_target.detach())
        pred_p2s = self.predictor(z_p_context, task='P2S')
        l_p2s = F.mse_loss(pred_p2s, z_s_target.detach())
        pred_p2p = apply_mask(self.predictor(z_p_context, task='P2P'), masks)
        l_p2p = F.mse_loss(pred_p2p, z_p_target.detach())
        avg_soft_probs = torch.mean(probs, dim=0)
        diff_entropy = -torch.sum(avg_soft_probs * torch.log(avg_soft_probs + 1e-10))
        l_preplexity = torch.log(torch.tensor(float(config["codebook_size"]))) - diff_entropy
        var_loss_context_patch, cov_loss_context_patch = self._calculate_vicreg_loss(z_p_context)
        var_loss_context_token, cov_loss_context_token = context_out["var_loss"], context_out["covar_loss"]
        token_div_loss = self._calculate_token_diversity_loss(z_s_context)
        total_loss = (
            lambda_weights["S2P"] * l_s2p +
            lambda_weights["P2P"] * l_p2p +
            lambda_weights["P2S"] * l_p2s +
            1.0 * beta_vq * l_vq +
            0.5 * l_preplexity +
            0.30*token_div_loss+
            0.35 * (var_loss_context_token+var_loss_context_patch+cov_loss_context_patch+ cov_loss_context_token)
        )
        if batch_idx % 5 == 0:
            print(f"TOTAL: {total_loss.item():.4f} | P2P: {l_p2p.item():.4f}, S2P: {l_s2p.item():.4f}, P2S: {l_p2s.item():.4f}, VQ: {l_vq.item():.4f}, Perp: {l_preplexity:.4f}, TokDiv: {token_div_loss.item():.4f}")
            print(f"  var[patch={var_loss_context_patch.item():.4f} tok={var_loss_context_token.item():.4f}] cov[patch={cov_loss_context_patch.item():.4f} tok={cov_loss_context_token.item():.4f}] | codes={active_codes} ({usage_pct:.1f}%)")

        return total_loss, {
            'l_s2p': l_s2p.item(),
            'l_p2p': l_p2p.item(),
            'l_p2s': l_p2s.item(),
            'l_vq': l_vq.item(),
            'l_preplexity': l_preplexity,
            'var_loss_context_patch': var_loss_context_patch.item(),
            'var_loss_context_token': var_loss_context_token.item(),
            'cov_loss_context_patch': cov_loss_context_patch.item(),
            'cov_loss_context_token': cov_loss_context_token.item(),
        }

    def evaluate(self, val_loader, lambda_weights, beta_vq, current_global_step, total_training_steps, vq_warmup, epoch):
        self.encoder.eval()
        self.encoder_ema.eval()
        self.predictor.eval()
        self.vector_quantizer.eval()  # Disable EMA codebook updates during validation
        val_loss = 0.0
        val_metrics = {'l_s2p': 0.0, 'l_p2s': 0.0, 'l_p2p': 0.0, 'l_vq': 0.0, 'l_preplexity': 0.0,'var_loss_context_patch': 0.0, 
        'var_loss_context_token': 0.0, 'cov_loss_context_patch':0.0, 'cov_loss_context_token':0.0}
        with torch.no_grad():
            for patches, masks, non_masks in val_loader:

                patches, masks, non_masks = patches.to(self.device), masks.to(self.device), non_masks.to(self.device)

                target_out = self.encoder_ema(patches)
                target_out = apply_mask(target_out, masks)
                
                context_out = self.encoder(patches, mask=non_masks)
                loss, loss_dict = self.compute_discrete_jepa_loss(
                    context_out,
                    target_out,
                    masks,
                    epoch,
                    lambda_weights=lambda_weights,
                    beta_vq=beta_vq,
                    current_global_step=current_global_step,
                    total_training_steps=total_training_steps,
                    vq_warmup=vq_warmup
                )
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_metrics[k] += v
        return val_loss / len(val_loader), {k: v / len(val_loader) for k, v in val_metrics.items()} 
        
    def save_model(self, encoder, target_encoder, predictor, optimizer, epoch, path_save):
        checkpoint_dir = os.path.dirname(path_save)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"Created directory: {checkpoint_dir}")
            except Exception as e:
                print(f"Could not create directory {checkpoint_dir}: {e}")
                return # Exit if we can't create the folder

        save_dict = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "vector_quantizer": self.vector_quantizer.state_dict(),
            "vector_quantizer_ema": self.vector_quantizer_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        try:
            # 3. Save the file
            path_name = f"{path_save}best_model.pt"
            print(f"Saving checkpoint to: {path_name}")
            torch.save(save_dict, path_name)
            print(f"Checkpoint saved: {path_name}")
        except Exception as e:
            print(f"Problem saving checkpoint: {e}")

    # we can use diffrent one - this is TS_JEPA basic
    def lr_lambda(self, epoch):
        start_lr = self.config["lr"]
        end_lr = self.config["end_lr"]
        if epoch < self.config["num_epochs"]:
            return start_lr + (end_lr - start_lr) * (epoch / (self.config["num_epochs"] - 1))
        else:
            return end_lr

    def train_and_evaluate(self):
        self.encoder = self.encoder.to(self.device)
        self.predictor = self.predictor.to(self.device)
        self.encoder_ema = self.encoder_ema.to(self.device)
        self.vector_quantizer = self.vector_quantizer.to(self.device)
        self.vector_quantizer_ema = self.vector_quantizer_ema.to(self.device)
        for p in self.encoder_ema.parameters():
            p.requires_grad = False
        for p in self.vector_quantizer_ema.parameters():
            p.requires_grad = False
        
        num_batches = self.steps_per_epoch
        best_val_loss = float("inf")
        total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0
        print("=" * 60)
        print(f"DATASET INFO:")
        print(f"  Train samples: {len(self.train_loader.dataset):,}")
        print(f"  Val samples:   {len(self.val_loader.dataset):,}")
        print(f"  Test samples:  {len(self.test_loader.dataset):,}")
        print(f"  Batch size:    {self.config['batch_size']}")
        print(f"  Steps/epoch:   {len(self.train_loader)}")
        print("=" * 60)
        self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, 0, f"{self.path_save}_INITIAL")
        current_global_step = 0
        # Training Loop
        for epoch in range(self.config["num_epochs"]):
            print(f"Starting Epoch {epoch}/{self.config['num_epochs']}")
            self.encoder.train()
            self.predictor.train()
            self.vector_quantizer.train()  # Enable EMA codebook updates during training
            running_loss = 0.0

            for batch_idx, (patches, masks, non_masks) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                m = next(self.ema_scheduler)
                patches = patches.to(self.device)

                current_global_step += 1
                with torch.no_grad():
                    target_out = self.encoder_ema(patches)
                    target_out = apply_mask(target_out, masks)

                context_out = self.encoder(patches, mask=non_masks)
                loss, loss_dict = self.compute_discrete_jepa_loss(
                    context_out,
                    target_out,
                    masks,
                    epoch,
                    lambda_weights=self.config["lambda_weights"],
                    beta_vq=self.config["beta_vq"],
                    current_global_step=current_global_step,
                    total_training_steps=self.total_steps,
                    vq_warmup=self.config["vq_warmup"],
                    batch_idx=batch_idx
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config["clip_grad"])
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config["clip_grad"])
                torch.nn.utils.clip_grad_norm_(self.vector_quantizer.parameters(), self.config["clip_grad"])
                self.optimizer.step()
                self.scheduler.step()

                with torch.no_grad():
                    for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
                        p_ema.data.mul_(m).add_((1.0-m)*p.detach().data)
                    # VQ EMA update (teacher codebook tracks student codebook)
                    for p, p_ema in zip(self.vector_quantizer.parameters(), self.vector_quantizer_ema.parameters()):
                        p_ema.data.mul_(m).add_((1.0-m)*p.detach().data)
                
                running_loss += loss.item()

            epoch_avg_loss = running_loss / len(self.train_loader)
            total_loss += epoch_avg_loss

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, lr: {self.optimizer.param_groups[0]['lr']:.3g} - JEPA Loss: {total_loss:.4f},")
            print(f"Validating set of Epoch: {epoch}")
            val_loss, val_dict = self.evaluate( self.val_loader, self.config["lambda_weights"], self.config["beta_vq"], current_global_step, self.total_steps, self.config["vq_warmup"], epoch)
            
            
            # Save Best Model
            if val_loss < best_val_loss and epoch >= self.warmup:
                best_val_loss = val_loss
                self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, epoch, f"{self.path_save}")
                self.best_model = {
                    "encoder": copy.deepcopy(self.encoder.state_dict()),
                    "predictor": copy.deepcopy(self.predictor.state_dict()),
                    "encoder_ema": copy.deepcopy(self.encoder_ema.state_dict()),
                    "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                    "epoch": epoch
                }
                print("New best validation loss! Model saved.")
            if epoch >= 100 and epoch %100 == 0:
                self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, epoch, f"{self.path_save}_epoch{epoch}")
                self.best_model = {
                    "encoder": copy.deepcopy(self.encoder.state_dict()),
                    "predictor": copy.deepcopy(self.predictor.state_dict()),
                    "encoder_ema": copy.deepcopy(self.encoder_ema.state_dict()),
                    "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                    "epoch": epoch
                }
                print("saved at epoch")
                

        print("Training complete. Starting Final Test:")
        test_loss, test_dict = self.evaluate(self.test_loader, self.config["lambda_weights"], self.config["beta_vq"], current_global_step, self.total_steps, self.config["vq_warmup"], 101)
        print(f"FINAL TEST RESULTS | Loss: {test_loss:.4f} | S2P: {test_dict['l_s2p']:.4f}")
        return test_loss, test_dict, self.best_model
