import numpy as np
import torch
import torch.nn as nn

class PosEmbeder(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        pos_tensor = self.sinusoidal_positional_encoding()
        self.register_buffer('pos_embed', pos_tensor)
        
    def get_pos_embed(self, type='sine_cosine'):
        if type == 'sine_cosine':
            return self.sinusoidal_positional_encoding()
        else:
            raise NotImplementedError(f"Positional embedding type '{type}' is not implemented.")
    def sinusoidal_positional_encoding(self):
        assert self.dim % 2 == 0, "Embedding dimension must be even for sine-cosine positional encoding."
        teta = np.arange(self.dim // 2, dtype=np.float32)
        teta /= self.dim / 2.0
        teta = 1.0 / (10000 ** teta)
        pos = np.arange(self.num_patches, dtype=np.float32)
        pos = pos.reshape(-1)
        out = np.einsum('i,j->ij', pos, teta)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        pos_emb = np.concatenate([emb_sin, emb_cos], axis=1)
        tensor = torch.from_numpy(pos_emb).float().unsqueeze(0)  # Shape: [1, num_patches, dim]
        #return nn.Parameter(tensor, requires_grad=False)
        return tensor
    def forward(self, x):
        return x + self.pos_embed