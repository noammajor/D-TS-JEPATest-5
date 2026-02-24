import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, ema_decay=0.99):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._ema_decay = ema_decay

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # Initialize codebook on the unit sphere (L2 normalized random directions)
        self._embedding.weight.data.normal_(0, 1)
        self._embedding.weight.data = F.normalize(self._embedding.weight.data, dim=-1)

        # EMA buffers for codebook updates (replaces gradient-based codebook learning)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self._embedding.weight.data.clone())

    def forward(self, inputs):
        # 1. Flatten inputs [Batch, Seq, Dim] -> [N, Dim]
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # 2. L2 normalize both inputs and codebook before distance computation.
        # This makes VQ direction-based (cosine similarity), removing sensitivity
        # to the scale differences caused by dropout between train and eval modes.
        flat_input_norm = F.normalize(flat_input, dim=-1)
        # Detach codebook so perplexity loss gradients only flow to encoder, not codebook
        # (codebook is managed by EMA updates, not gradient descent)
        codebook_norm = F.normalize(self._embedding.weight.detach(), dim=-1)

        # 3. Cosine distance: ||x_norm - e_norm||^2 = 2 - 2*cos(x, e)
        # Range [0, 4], practically [0, 2] for reasonable vectors.
        distances = 2.0 - 2.0 * torch.matmul(flat_input_norm, codebook_norm.t())

        # 4. Soft assignment (perplexity loss gradients flow to encoder to encourage diversity)
        soft_probs = F.softmax(-distances / 0.1, dim=1)

        # 5. Hard assignment
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings_hard = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings_hard.scatter_(1, encoding_indices, 1)

        # 6. EMA codebook update (only during training)
        # Replaces gradient-based q_latent_loss with stable running-average tracking.
        if self.training:
            # Track cluster sizes with EMA
            self._ema_cluster_size = (self._ema_cluster_size * self._ema_decay
                                      + (1 - self._ema_decay) * torch.sum(encodings_hard, 0))
            # Laplace smoothing prevents empty clusters from collapsing
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + 1e-5) / (n + self._num_embeddings * 1e-5) * n
            )
            # Track sum of assigned (normalized) encoder outputs
            dw = torch.matmul(encodings_hard.t(), flat_input_norm)
            self._ema_w = self._ema_w * self._ema_decay + (1 - self._ema_decay) * dw
            # Update codebook: centroid = sum / count, then re-normalize to unit sphere
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            self._embedding.weight.data = F.normalize(self._embedding.weight.data, dim=-1)

        # 7. Quantize using normalized codebook
        quantized = torch.matmul(encodings_hard, codebook_norm).view(inputs.shape)

        # 8. Commitment loss only (codebook is updated via EMA, not gradients)
        # Pushes encoder outputs toward their nearest codebook entry direction.
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input_norm.view(inputs.shape))
        vq_loss = self._commitment_cost * e_latent_loss

        # 9. Straight-Through Estimator (on normalized vectors)
        quantized = flat_input_norm.view(inputs.shape) + (quantized - flat_input_norm.view(inputs.shape)).detach()

        # 10. Perplexity monitoring (from hard assignments)
        avg_probs = torch.mean(encodings_hard, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return vq_loss, quantized, soft_probs, perplexity, encoding_indices
