import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_(0, 0.5)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input (inputs: B, L, C)
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Hard perplexity (metric only, no gradient)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Soft assignments: differentiable via distances (has gradient to both
        # encoder outputs and codebook weights)
        # soft_token_probs: per-token distribution over codebook [N, codebook_size]
        # soft_avg_probs:   batch-averaged distribution           [codebook_size]
        soft_token_probs = F.softmax(-distances, dim=-1)
        soft_avg_probs   = soft_token_probs.mean(dim=0)

        return loss, quantized, perplexity, encoding_indices, encodings, soft_avg_probs




