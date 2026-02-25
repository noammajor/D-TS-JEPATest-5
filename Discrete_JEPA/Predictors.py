import torch
import torch.nn as nn


class DiscreteJEPAPredictor(nn.Module):
    def __init__(self, num_patches, num_semantic_tokens, embed_dim, predictor_embed_dim, config):
        super().__init__()
        self.num_patches = num_patches
        self.num_semantic_tokens = num_semantic_tokens
        self.embed_dim = embed_dim

        # Learnable positional embeddings for target positions
        self.target_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.target_pos_embed, std=0.02)

        # P2P: concat(mean-pooled context [D], target_pos [D]) → predict one target patch [D]
        self.mlp_p2p = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, embed_dim)
        )

        # S2P: compress semantic tokens first, then same concat+MLP as P2P
        self.sem_compress = nn.Linear(num_semantic_tokens * embed_dim, embed_dim)
        self.mlp_s2p = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, embed_dim)
        )

        # P2S: mean-pooled context → predict all semantic tokens (no positional conditioning)
        self.mlp_p2s = nn.Sequential(
            nn.Linear(embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, num_semantic_tokens * embed_dim)
        )

    def _gather_target_pos(self, masks, B_total):
        """Gather target positional embeddings at mask indices, expanding for F>1."""
        B = masks.shape[0]
        F = B_total // B
        if F > 1:
            m = masks.unsqueeze(1).repeat(1, F, 1).view(B_total, -1)  # [B*F, N_masked]
        else:
            m = masks                                                   # [B, N_masked]
        pos = self.target_pos_embed.expand(B_total, -1, -1)            # [B*F, num_patches, D]
        idx = m.unsqueeze(-1).expand(-1, -1, self.embed_dim)           # [B*F, N_masked, D]
        return torch.gather(pos, dim=1, index=idx)                     # [B*F, N_masked, D]

    def forward(self, x_input, task='P2P', masks=None, **kwargs):
        # **kwargs absorbs legacy args for compatibility
        B_total = x_input.shape[0]

        if task == 'P2P':
            # Mean-pool context patches → single context summary
            z = x_input.mean(dim=1)                                    # [B*F, D]
            pos = self._gather_target_pos(masks, B_total)              # [B*F, N_masked, D]
            N_masked = pos.shape[1]
            # Each target position gets: concat(context_summary, target_pos_embed)
            z_exp = z.unsqueeze(1).expand(-1, N_masked, -1)            # [B*F, N_masked, D]
            combined = torch.cat([z_exp, pos], dim=-1)                 # [B*F, N_masked, 2D]
            out = self.mlp_p2p(combined.view(B_total * N_masked, -1))  # [B*F*N_masked, D]
            return out.view(B_total, N_masked, self.embed_dim)

        elif task == 'S2P':
            # Flatten and compress semantic tokens to a single summary
            z = self.sem_compress(x_input.flatten(1))                  # [B*F, D]
            pos = self._gather_target_pos(masks, B_total)              # [B*F, N_masked, D]
            N_masked = pos.shape[1]
            z_exp = z.unsqueeze(1).expand(-1, N_masked, -1)            # [B*F, N_masked, D]
            combined = torch.cat([z_exp, pos], dim=-1)                 # [B*F, N_masked, 2D]
            out = self.mlp_s2p(combined.view(B_total * N_masked, -1))  # [B*F*N_masked, D]
            return out.view(B_total, N_masked, self.embed_dim)

        elif task == 'P2S':
            # Mean-pool context patches → predict all semantic tokens
            z = x_input.mean(dim=1)                                    # [B*F, D]
            return self.mlp_p2s(z).view(B_total, self.num_semantic_tokens, self.embed_dim)
