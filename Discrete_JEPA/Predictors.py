import torch
import torch.nn as nn


class DiscreteJEPAPredictor(nn.Module):
    def __init__(self, num_patches, num_semantic_tokens, embed_dim, predictor_embed_dim, config):
        super().__init__()
        self.num_patches = num_patches
        self.num_semantic_tokens = num_semantic_tokens
        self.embed_dim = embed_dim

        # Per-patch/per-token embedding (same as old predictor) — applied to all tasks
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)

        # P2P: per-patch → predict all num_patches positions, then sum over context
        self.proj_p2p = nn.Sequential(
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, num_patches * embed_dim)
        )

        # S2P: flatten projected semantic tokens (fixed size: S*pdim) → all patch positions
        self.mlp_s2p = nn.Sequential(
            nn.Linear(num_semantic_tokens * predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, num_patches * embed_dim)
        )

        # P2S: per-patch → predict all semantic tokens, then sum over context
        self.proj_p2s = nn.Sequential(
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, num_semantic_tokens * embed_dim)
        )

    def forward(self, x_input, task='P2P', **kwargs):
        # **kwargs absorbs legacy args (target_mask, target_start_idx) for compatibility
        B = x_input.shape[0]
        # Project each patch/token through predictor_embed (same as old predictor)
        x = self.predictor_embed(x_input)               # [B, N, pdim]
        if task == 'P2P':
            # Per-patch predictions for all positions, summed over context
            pred_all = self.proj_p2p(x).sum(dim=1)      # [B, num_patches * D]
            return pred_all.view(B, self.num_patches, self.embed_dim)
        elif task == 'S2P':
            # Flatten projected semantic tokens, run MLP
            z = x.flatten(1)                             # [B, S * pdim]
            return self.mlp_s2p(z).view(B, self.num_patches, self.embed_dim)
        elif task == 'P2S':
            # Per-patch predictions for semantic tokens, summed over context
            pred_all = self.proj_p2s(x).sum(dim=1)      # [B, S * D]
            return pred_all.view(B, self.num_semantic_tokens, self.embed_dim)
