import torch

def get_mask_style( B, num_patches, type="bernoulli", p=0.5, device='cpu'):
    """
    B: Batch size
    num_patches: Total number of patches
    """
    if type == "bernoulli":
        return get_bernoulli_indices(B, num_patches, p, device)
    elif type == "block":
        return get_block_indices(B, num_patches, int(num_patches * p), device)

def get_bernoulli_indices(B, num_patches, p, device):
    num_keep = int(num_patches * (1 - p))
    
    # Generate random indices for the whole batch
    # Each row is a shuffled version of [0, ..., num_patches-1]
    batch_indices = torch.stack([torch.randperm(num_patches, device=device) for _ in range(B)])
    
    context_idx = batch_indices[:, :num_keep]
    target_idx = batch_indices[:, num_keep:]
    
    return context_idx, target_idx

def get_block_indices(B, num_patches, block_size, device):
    context_list = []
    target_list = []
    
    for _ in range(B):
        all_idx = torch.arange(num_patches, device=device)
        start = torch.randint(0, num_patches - block_size+1, (1,)).item()
        
        t_idx = all_idx[start : start + block_size]
        c_idx = torch.cat([all_idx[:start], all_idx[start + block_size :]])
        
        context_list.append(c_idx)
        target_list.append(t_idx)
        
    return torch.stack(context_list), torch.stack(target_list)