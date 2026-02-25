import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Input shape: [B, P, P_L, F]
    Statistics are computed per-variable (F) across patches (P) and patch length (P_L),
    giving one mean/std per instance per channel — consistent with the RevIN paper.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor):
        """
        x: [B, P, P_L, F]
        Returns:
            x_norm: [B, P, P_L, F]  — normalized input to pass to encoder
            mu:     [B, 1, 1, F]    — store and pass to denormalize / normalize_target
            sigma:  [B, 1, 1, F]
        """
        mu = x.mean(dim=(1, 2), keepdim=True)                                        # [B, 1, 1, F]
        sigma = (x.var(dim=(1, 2), unbiased=False, keepdim=True) + self.eps).sqrt()  # [B, 1, 1, F]
        return (x - mu) / sigma, mu, sigma

    @staticmethod
    def denormalize(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """Reverse normalization on channel-independent predictions.
        x:     [B*F, h, P_L]
        mu:    [B, 1, 1, F]
        sigma: [B, 1, 1, F]
        """
        B_F = x.shape[0]
        mu_flat = mu.permute(0, 3, 1, 2).reshape(B_F, 1, 1)      # [B*F, 1, 1]
        sigma_flat = sigma.permute(0, 3, 1, 2).reshape(B_F, 1, 1)
        return x * sigma_flat + mu_flat

    @staticmethod
    def normalize_target(target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """Normalize target patches with the context RevIN stats.
        target: [B*F, h, P_L]
        mu:     [B, 1, 1, F]
        sigma:  [B, 1, 1, F]
        """
        B_F = target.shape[0]
        mu_flat = mu.permute(0, 3, 1, 2).reshape(B_F, 1, 1)      # [B*F, 1, 1]
        sigma_flat = sigma.permute(0, 3, 1, 2).reshape(B_F, 1, 1)
        return (target - mu_flat) / sigma_flat
