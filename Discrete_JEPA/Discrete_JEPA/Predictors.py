import math
import torch
import torch.nn as nn


# ─── Predictor Layer ───────────────────────────────────────────────────────────

class PredictorLayer(nn.Module):
    """Pre-norm transformer layer (self-attention + FFN)."""

    def __init__(self, embed_dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.drop    = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.drop_ff = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_sa(x), self.norm_sa(x), self.norm_sa(x))
        x = x + self.drop(attn_out)
        x = x + self.drop_ff(self.ff(self.norm_ff(x)))
        return x


# ─── Predictor ─────────────────────────────────────────────────────────────────

class DiscreteJEPAPredictor(nn.Module):
    """
    Single-stream predictor for Discrete JEPA (I-JEPA style).

    For each task, the context tokens and target query tokens are concatenated
    into one sequence and processed with standard self-attention. After the
    transformer, only the target token positions are extracted and projected
    back to encoder_dim.

    Three prediction tasks:
      P2P — context patch embeddings → target patch embeddings
      S2P — semantic tokens          → target patch embeddings
      P2S — context patch embeddings → semantic token embeddings
    """

    def __init__(
        self,
        num_semantic_tokens: int,
        embed_dim: int,
        config: dict,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim           = embed_dim
        self.num_semantic_tokens = num_semantic_tokens

        num_patches = config["ratio_patches"]
        pred_dim    = config.get("predictor_embed_dim", embed_dim // 2)
        nhead       = config.get("predictor_nhead",     4)
        num_layers  = config.get("predictor_num_layers", 2)
        dropout     = config.get("predictor_drop_rate",  0.1)
        self.pred_dim = pred_dim

        # ── Input projection: encoder_dim → predictor_dim ─────────────────────
        self.ctx_proj = nn.Linear(embed_dim, pred_dim)   # projects context tokens
        self.pe_proj  = nn.Linear(embed_dim, pred_dim)   # projects sincos PE for mask tokens

        # ── Learnable query tokens (at predictor_dim) ─────────────────────────
        self.mask_token     = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.semantic_query = nn.Parameter(torch.zeros(1, num_semantic_tokens, pred_dim))

        # ── Positional encoding (fixed sincos at encoder embed_dim) ───────────
        self.register_buffer("pos_embed", self._make_sincos_pe(num_patches, embed_dim))

        # ── Single-stream transformer stacks (at predictor_dim) ───────────────
        self.p2p_layers = nn.ModuleList([
            PredictorLayer(pred_dim, nhead, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.s2p_layers = nn.ModuleList([
            PredictorLayer(pred_dim, nhead, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.p2s_layers = nn.ModuleList([
            PredictorLayer(pred_dim, nhead, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Output projections: predictor_dim → encoder_dim ───────────────────
        self.out_proj_p2p = nn.Sequential(nn.LayerNorm(pred_dim), nn.Linear(pred_dim, embed_dim))
        self.out_proj_s2p = nn.Sequential(nn.LayerNorm(pred_dim), nn.Linear(pred_dim, embed_dim))
        self.out_proj_p2s = nn.Sequential(nn.LayerNorm(pred_dim), nn.Linear(pred_dim, embed_dim))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_sincos_pe(num_patches: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(num_patches, d_model)
        pos = torch.arange(0, num_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
        return pe  # [num_patches, encoder_dim]

    def _build_mask_tokens(self, target_mask: torch.Tensor, B_total: int) -> torch.Tensor:
        """
        Build mask tokens with positional encoding for target patch positions.

        target_mask: [B, N_t]  — integer patch indices
        B_total:     B*F       — channel-independent batch size

        Returns: [B*F, N_t, pred_dim]
        """
        B_orig, N_t = target_mask.shape
        F = B_total // B_orig

        pe = self.pos_embed[target_mask]                          # [B, N_t, encoder_dim]
        pe = pe.unsqueeze(1).expand(-1, F, -1, -1)               # [B, F, N_t, encoder_dim]
        pe = pe.reshape(B_total, N_t, self.embed_dim)            # [B*F, N_t, encoder_dim]
        pe = self.pe_proj(pe)                                     # [B*F, N_t, pred_dim]

        return self.mask_token.expand(B_total, N_t, -1) + pe     # [B*F, N_t, pred_dim]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_input: torch.Tensor,
        task: str = "P2P",
        target_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        x_input:     [B*F, N_ctx, encoder_dim]  context patches   (P2P, P2S)
                  OR [B*F, S,     encoder_dim]  semantic tokens   (S2P)
        target_mask: [B, N_t]                   target patch indices (P2P, S2P)

        Returns:
            P2P → [B*F, N_t, encoder_dim]
            S2P → [B*F, N_t, encoder_dim]
            P2S → [B*F, S,   encoder_dim]
        """
        B   = x_input.shape[0]
        ctx = self.ctx_proj(x_input)    # [B*F, N_ctx_or_S, pred_dim]

        if task == "P2P":
            tgt    = self._build_mask_tokens(target_mask, B)   # [B*F, N_t, pred_dim]
            N_ctx  = ctx.shape[1]
            seq    = torch.cat([ctx, tgt], dim=1)
            for layer in self.p2p_layers:
                seq = layer(seq)
            return self.out_proj_p2p(seq[:, N_ctx:, :])        # [B*F, N_t, encoder_dim]

        elif task == "S2P":
            tgt    = self._build_mask_tokens(target_mask, B)   # [B*F, N_t, pred_dim]
            N_ctx  = ctx.shape[1]
            seq    = torch.cat([ctx, tgt], dim=1)
            for layer in self.s2p_layers:
                seq = layer(seq)
            return self.out_proj_s2p(seq[:, N_ctx:, :])        # [B*F, N_t, encoder_dim]

        elif task == "P2S":
            tgt    = self.semantic_query.expand(B, -1, -1)     # [B*F, S, pred_dim]
            N_ctx  = ctx.shape[1]
            seq    = torch.cat([ctx, tgt], dim=1)
            for layer in self.p2s_layers:
                seq = layer(seq)
            return self.out_proj_p2s(seq[:, N_ctx:, :])        # [B*F, S, encoder_dim]

        else:
            raise ValueError(f"Unknown predictor task: '{task}'")
