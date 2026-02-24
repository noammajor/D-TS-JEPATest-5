import torch
import torch.nn as nn
def init_weights(m):
    # 1. Linear Layers: The workhorses for S2P, P2S, and P2P predictions
    if isinstance(m, torch.nn.Linear):
        # Truncated normal prevents 'outlier' weights from saturating the GELU
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

    # 2. Embeddings: For your codebook and semantic tokens
    elif isinstance(m, torch.nn.Embedding):
        # Initializing the codebook uniformly helps all 'words' be chosen early on
        torch.nn.init.uniform_(m.weight, -1.0 / m.weight.size(0), 1.0 / m.weight.size(0))

    # 3. Layer Normalization: Crucial for the Predictor and Encoder Norms
    elif isinstance(m, torch.nn.LayerNorm):
        # Bias 0 and Weight 1.0 keeps the distribution 'standard' at the start
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)