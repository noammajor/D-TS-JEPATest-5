import torch
import torch.nn.functional as F

def compute_var_loss(self, z):
    eps = 1e-4
    std_z = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_z))
    return var_loss
def calculate_perplexity_regularizer(self, context_ppl, target_ppl):
    eps = 1e-8
    loss = -torch.log(context_ppl + target_ppl + eps)
    return loss
    
def _calculate_vicreg_loss(self, x: torch.Tensor):
    num_features = x.shape[-1]
    x_flat = x.reshape(-1, num_features)
    std = torch.sqrt(x_flat.var(dim=0, unbiased=True) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))
    x_centered = x_flat - x_flat.mean(dim=0)
    cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / num_features
    return var_loss, cov_loss
def _calculate_token_diversity_loss(self, semantic_tokens: torch.Tensor):
    """Penalizes semantic tokens for being too similar within each sample.
    semantic_tokens: [B, num_semantic_tokens, D]
    Returns mean off-diagonal cosine similarity (lower = more diverse).
    """
    normed = F.normalize(semantic_tokens, dim=-1)  # [B, S, D]
    sim = torch.bmm(normed, normed.transpose(1, 2))  # [B, S, S]
    S = semantic_tokens.shape[1]
    mask = ~torch.eye(S, device=semantic_tokens.device).bool()
    diversity_loss = sim[:, mask].mean()
    return diversity_loss

def _grounding_loss(self, pred_p2p: torch.Tensor, patches: torch.Tensor, non_masks: torch.Tensor) -> torch.Tensor:
    """Per-patch grounding: decode each predicted target embedding to raw patch values.

    For every target position i, the grounding head decodes pred_p2p[:,i,:]
    back to patch-space [P_L] and compares with the actual raw patch at that
    position. Prevents the predictor from predicting in a collapsed space that
    has no relation to the real signal.

    pred_p2p:  [B*F, N_target, D]  — predictor P2P output at target positions
    patches:   [B, P, P_L, F]      — normalized raw input patches
    non_masks: [B, N_target]       — target (hidden) patch indices
    """
    BF, N_target, D = pred_p2p.shape
    B = non_masks.size(0)
    n_vars = BF // B
    P_L = patches.shape[2]

    # Reshape raw patches to [B*n_vars, P, P_L]
    x_raw = patches.permute(0, 3, 1, 2).reshape(BF, -1, P_L)

    # Gather target positions: [B, N_target] → [B*n_vars, N_target]
    masks_expanded = non_masks.unsqueeze(1).repeat(1, n_vars, 1).view(BF, N_target)
    idx = masks_expanded.unsqueeze(-1).expand(-1, -1, P_L)       # [B*n_vars, N_target, P_L]
    target_raw = torch.gather(x_raw, 1, idx)                     # [B*n_vars, N_target, P_L]

    # Decode each predicted embedding individually → [B*n_vars, N_target, P_L]
    pred_raw = self.grounding_head(pred_p2p)

    return torch.nn.functional.mse_loss(pred_raw, target_raw.detach())

def _vq_kl_loss_batch(self, teacher_soft_probs: torch.Tensor, student_soft_probs: torch.Tensor) -> torch.Tensor:
    """Batch-level KL divergence between teacher and student codebook usage distributions.

    teacher_soft_probs: [codebook_size] — soft_avg_probs from teacher VQ (averaged over all tokens)
    student_soft_probs: [codebook_size] — soft_avg_probs from student VQ

    Minimizes KL(teacher || student): forces the student's overall codebook usage pattern
    to match the teacher's. Coarser than per-token but zero overhead — uses the
    soft_avg_probs already returned by VectorQuantizer.forward().
    """
    eps = 1e-8
    p = teacher_soft_probs.clamp(min=eps).detach()  # teacher = target, no grad
    q = student_soft_probs.clamp(min=eps)
    return F.kl_div(q.log(), p, reduction='sum')


def _vq_kl_loss_per_token(self, teacher_token_probs: torch.Tensor, student_token_probs: torch.Tensor) -> torch.Tensor:
    """Per-token KL divergence between teacher and student soft codebook assignments.

    teacher_token_probs: [N, codebook_size] — soft_token_probs from teacher VQ (before mean)
    student_token_probs: [N, codebook_size] — soft_token_probs from student VQ (before mean)
    N = B * L (flattened batch × sequence length)

    DINO-style alignment at the token level: each individual token is pushed to quantize
    to the same codebook entry as the teacher's version of that token.
    Teacher probs are detached — gradients flow only through the student.
    Requires VectorQuantizer to return soft_token_probs (see VQ.py).
    """
    eps = 1e-8
    p = teacher_token_probs.clamp(min=eps).detach()  # teacher = target, no grad
    q = student_token_probs.clamp(min=eps)
    return F.kl_div(q.log(), p, reduction='batchmean')


def _cross_decorrelation_loss(self, z_patches: torch.Tensor, z_tokens: torch.Tensor) -> torch.Tensor:
    """Cross-modal decorrelation: penalizes every patch for being similar to every token.

    Computes pairwise cosine similarity between all context patches and all
    semantic tokens within each sample [B*F, N_ctx, S], then penalizes
    large values. Patches and tokens should occupy different regions of the
    shared latent space and not interfere with each other's gradients.

    z_patches: [B*F, N_ctx, D]
    z_tokens:  [B*F, S,     D]
    Returns: scalar loss.
    """
    p = F.normalize(z_patches, dim=-1)              # [B*F, N_ctx, D]
    s = F.normalize(z_tokens,  dim=-1)              # [B*F, S,     D]
    cross_sim = torch.bmm(p, s.transpose(1, 2))     # [B*F, N_ctx, S]
    return cross_sim.pow(2).mean()