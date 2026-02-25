import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantizer with EMA codebook updates (VQ-VAE-2 style).

    The codebook is updated via exponential moving averages — equivalent to
    online k-means — instead of backpropagation. This decouples codebook
    placement from the encoder's optimizer: the codebook smoothly tracks
    wherever encoder outputs cluster, independent of learning rate.

    Only the commitment loss trains the encoder (pulls encoder outputs toward
    their nearest codebook entry). The Straight-Through Estimator lets
    downstream losses (S2P, P2P) flow back through the quantization step.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay

        # Codebook weights: NOT a learnable parameter — updated by EMA only
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._embedding.weight.requires_grad_(False)

        # EMA running stats (buffers are saved in state_dict but not optimizer params)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self._embedding.weight.data.clone())

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # L2 distances between encoder outputs and codebook entries
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Soft assignment for monitoring / perplexity loss
        soft_probs = F.softmax(-distances, dim=1)

        # Hard assignment
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings_hard = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings_hard.scatter_(1, encoding_indices, 1)

        # Codebook lookup
        quantized = torch.matmul(encodings_hard, self._embedding.weight).view(input_shape)

        # EMA update — runs only during training, no gradients involved
        if self.training:
            # Track how many encoder outputs are assigned to each entry
            self._ema_cluster_size = (
                self._decay * self._ema_cluster_size
                + (1 - self._decay) * encodings_hard.sum(0)
            )
            # Track the sum of encoder outputs for each entry
            dw = encodings_hard.t() @ flat_input.detach()
            self._ema_w = self._decay * self._ema_w + (1 - self._decay) * dw

            # Laplace smoothing: prevents dead codes from staying dead
            n = self._ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self._ema_cluster_size + 1e-5)
                / (n + self._num_embeddings * 1e-5) * n
            )
            # Set each codebook entry = mean of encoder outputs assigned to it
            self._embedding.weight.data = self._ema_w / smoothed_cluster_size.unsqueeze(1)

        # Commitment loss: pulls encoder outputs toward their nearest codebook entry
        # (No q_latent_loss — codebook is updated above, not via backprop)
        vq_loss = self._commitment_cost * F.mse_loss(quantized.detach(), inputs)

        # Straight-Through Estimator: downstream losses (S2P, P2P) flow to encoder
        quantized = inputs + (quantized - inputs).detach()

        # Monitoring
        avg_probs = torch.mean(encodings_hard, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        active_codes = (torch.sum(encodings_hard, dim=0) > 0).sum()
        usage_pct = active_codes.float() / self._num_embeddings * 100

        return vq_loss, quantized, soft_probs, perplexity, encoding_indices, active_codes, usage_pct
