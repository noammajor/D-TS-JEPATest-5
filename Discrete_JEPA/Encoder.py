# based on TS_JEPA: https://github.com/Sennadir/TS_JEPA/blob/main/src/models/encoder.py
import torch
import torch.nn as nn
import numpy as np
import math
from Discrete_JEPA.VQ import *
from mask_util import *
from utils.modules import *
from pos_embeder import PosEmbeder
from Discrete_JEPA.Tokenizer import TS_Tokenizer

class SemanticSelfAttention(nn.Module):
    """Self-attention block where semantic tokens attend only to each other."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        """x: [B, S, D] — semantic tokens only"""
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        num_patches,
        num_semantic_tokens,
        dim_in,
        kernel_size,
        embed_dim,
        embed_bias,
        nhead,
        num_layers,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        jepa=False,
        embed_activation=nn.GELU(),
        codebook_size=128,
        commitment_cost=2.0,
        type_enc = "context"
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_semantic_tokens = num_semantic_tokens
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        #Semantic tokens
        self.semantic_tokens = nn.Parameter(torch.randn(1, self.num_semantic_tokens, embed_dim))
        torch.nn.init.trunc_normal_(self.semantic_tokens, std=0.02)

        self.pos_embed_layer = PosEmbeder(dim=self.embed_dim, num_patches=self.num_patches)

        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, act_layer=nn.GELU, norm_layer=norm_layer
            ) for _ in range(num_layers)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

        #self.vector_quantizer = VectorQuantizer(
        #    num_embeddings=codebook_size,
        #    embedding_dim=embed_dim,
        #commitment_cost=commitment_cost
        #)
        self.tokenizer = TS_Tokenizer(
            dim_in=dim_in,  
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            embed_bias=embed_bias,
            activation=embed_activation
        )
        self.semantic_self_attn = SemanticSelfAttention(dim=embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio)
        self.jepa = jepa
        self.type_enc = type_enc

    def forward(self, x, typenow= None,mask=None):
        B, P, P_L , F = x.shape #[Batch, Patches, Patch_Len]
        #RevIN normalization
        #self.mu = x.mean(dim=1, keepdim=True)
        #self.sigma = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        #x = (x - self.mu) / self.sigma
        #channel independence
        x = x.permute(0, 3, 1, 2).reshape(B * F, P, P_L)
        #Encoder embedding
        x = self.tokenizer(x)
        if typenow == None:
            pos_embs = self.pos_embed_layer.pos_embed[:, :P, :]
        else:
            pos_embs = self.pos_embed_layer.pos_embed[:, :P+typenow, :]
        x = x + pos_embs

        #x = self.pos_embed_layer(x)
        if mask is not None:
            x = apply_mask(x, mask)  #[B, num_patches, D]
        sem_tokens = self.semantic_tokens.expand(B * F, -1, -1) #creates pointer copies of tokens for each example in the batch
        B_scaled, N_curr, D = x.shape 
        S = self.num_semantic_tokens
        
        # 1. Create the Asymmetric Mask dynamically
        total_len = N_curr + S
        
        # Initialize a mask of zeros (meaning allow all attention)
        # attn_mask shape: [total_len, total_len]
        attn_mask = torch.zeros((total_len, total_len), device=x.device)
        
        # DISALLOW Patches from seeing Semantic Tokens
        # This prevents the raw data from being "polluted" by the learned labels
        attn_mask[:N_curr, N_curr:] = -float('inf')
        x = torch.cat((x, sem_tokens), dim=1) #[B, num_patches + num_semantic, D]

        #transformer blocks
        #added attnmask
        for blk in self.predictor_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.encoder_norm(x)
        out_semantic = x[:, -self.num_semantic_tokens:, :]
        data_patches = x[:, :-self.num_semantic_tokens, :]
        var_loss, covar_loss =self._calculate_vicreg_loss(out_semantic)
        out_semantic = self.semantic_self_attn(out_semantic)
        var_loss, cov_loss=self._calculate_vicreg_loss(out_semantic)
        return {
            "quantized_semantic": out_semantic,
            "data_patches": data_patches,
            "orig_B": B,
            "orig_F": F,
            "var_loss":var_loss,
            "covar_loss":cov_loss
        }

    def init_embed(self):
        """
        This function serves for the positional encoder which is based on a
        Sin-Cos Pos Encoder.
        ---
        Users can choose any other Positional Encoder that they may see fit.
        """
        assert self.embed_dim % 2 == 0

        omega = np.arange(self.embed_dim // 2, dtype=float)
        omega /= self.embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = np.arange(self.num_patches, dtype=float)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)

        self.pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))

        return emb
    def _calculate_vicreg_loss(self, x: torch.Tensor):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x_flat = x.reshape(-1, num_features) 
        x_centered = x_flat - x_flat.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
        cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / num_features
        
        return var_loss, cov_loss