import torch
import torch.nn.functional as F

    
def _calculate_vicreg_loss(self, x: torch.Tensor):
    # x: [B*F, N_ctx, embed_dim]
    num_features = x.shape[-1]

    # Variance across batch — prevents inter-sample collapse
    std_batch = torch.sqrt(x.var(dim=0, unbiased=True) + 1e-4)   # [N_ctx, D]
    var_loss_batch = torch.mean(F.relu(1.0 - std_batch))

    # Variance across patch positions — prevents intra-sample positional collapse
    std_pos = torch.sqrt(x.var(dim=1, unbiased=True) + 1e-4)     # [B*F, D]
    var_loss_pos = torch.mean(F.relu(1.0 - std_pos))

    x_flat = x.reshape(-1, num_features)
    x_centered = x_flat - x_flat.mean(dim=0)
    cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / num_features

    return var_loss_pos,var_loss_batch, cov_loss