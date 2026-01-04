import torch
import numpy as np

def compute_low_rank_approximation(matrix, rank):
    """
    Compute low-rank approximation of a matrix using SVD
    
    Args:
        matrix: 2D tensor
        rank: Target rank
    
    Returns:
        A, B matrices such that matrix ≈ B @ A
    """
    # Perform SVD
    U, S, Vh = torch.svd(matrix)
    
    # Keep only top-k singular values
    rank = min(rank, min(matrix.shape))
    
    # A = V_k^T * sqrt(S_k)
    # B = U_k * sqrt(S_k)
    A = Vh[:rank, :].T * torch.sqrt(S[:rank])
    B = U[:, :rank] * torch.sqrt(S[:rank])
    
    return A, B.T


def compute_rank(matrix, threshold=1e-5):
    """
    Compute the numerical rank of a matrix
    
    Args:
        matrix: 2D tensor
        threshold: Singular value threshold
    
    Returns:
        Numerical rank
    """
    _, S, _ = torch.svd(matrix)
    rank = torch.sum(S > threshold).item()
    return rank


def frobenius_norm(matrix):
    """
    Compute Frobenius norm of a matrix
    
    Args:
        matrix: 2D tensor
    
    Returns:
        Frobenius norm
    """
    return torch.norm(matrix, p='fro').item()


def compute_delta_weight_stats(lora_A, lora_B):
    """
    Compute statistics about the delta weight ΔW = B @ A
    
    Args:
        lora_A: A matrix (down-projection)
        lora_B: B matrix (up-projection)
    
    Returns:
        Dictionary with statistics
    """
    # Compute ΔW
    delta_W = torch.matmul(lora_B.T, lora_A.T)
    
    stats = {
        'shape': list(delta_W.shape),
        'mean': delta_W.mean().item(),
        'std': delta_W.std().item(),
        'min': delta_W.min().item(),
        'max': delta_W.max().item(),
        'frobenius_norm': frobenius_norm(delta_W),
        'rank': compute_rank(delta_W)
    }
    
    return stats


def relative_magnitude(base_weight, delta_weight):
    """
    Compute relative magnitude of ΔW compared to W
    
    Args:
        base_weight: Original weight matrix
        delta_weight: Delta weight matrix
    
    Returns:
        Ratio of norms
    """
    base_norm = frobenius_norm(base_weight)
    delta_norm = frobenius_norm(delta_weight)
    
    if base_norm > 0:
        return delta_norm / base_norm
    return 0.0


def gradient_norm(model, param_type='all'):
    """
    Compute total gradient norm for model parameters
    
    Args:
        model: PyTorch model
        param_type: 'all', 'lora', or 'base'
    
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param_type == 'all':
                include = True
            elif param_type == 'lora':
                include = 'lora' in name
            elif param_type == 'base':
                include = 'lora' not in name
            else:
                include = True
            
            if include:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    return total_norm


def compute_parameter_efficiency(model):
    """
    Compute parameter efficiency metrics for LoRA model
    
    Args:
        model: PyTorch model with LoRA
    
    Returns:
        Dictionary with efficiency metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    
    metrics = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'parameter_reduction': 1 - (trainable_params / total_params) if total_params > 0 else 0
    }
    
    return metrics


def cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two tensors
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
    
    Returns:
        Cosine similarity (-1 to 1)
    """
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    
    dot_product = torch.dot(tensor1_flat, tensor2_flat)
    norm1 = torch.norm(tensor1_flat)
    norm2 = torch.norm(tensor2_flat)
    
    if norm1 > 0 and norm2 > 0:
        return (dot_product / (norm1 * norm2)).item()
    return 0.0


def analyze_weight_changes(weight_before, weight_after):
    """
    Analyze how weights changed during training
    
    Args:
        weight_before: Weight tensor before training
        weight_after: Weight tensor after training
    
    Returns:
        Dictionary with change statistics
    """
    delta = weight_after - weight_before
    
    analysis = {
        'l2_distance': torch.norm(delta).item(),
        'relative_change': torch.norm(delta).item() / torch.norm(weight_before).item(),
        'mean_change': delta.mean().item(),
        'max_change': delta.abs().max().item(),
        'cosine_sim': cosine_similarity(weight_before, weight_after),
        'frobenius_norm_delta': frobenius_norm(delta)
    }
    
    return analysis


def estimate_memory_savings(base_params, lora_rank, num_layers):
    """
    Estimate memory savings from using LoRA
    
    Args:
        base_params: Number of parameters in base model
        lora_rank: Rank of LoRA matrices
        num_layers: Number of layers with LoRA
    
    Returns:
        Dictionary with memory estimates
    """
    # Assume typical dimensions
    typical_dim = int(np.sqrt(base_params / num_layers))
    
    # Full fine-tune: all params trainable
    full_finetune_params = base_params
    
    # LoRA: only low-rank adapters trainable
    # For each layer: A (d x r) + B (r x d) = 2*d*r params
    lora_params_per_layer = 2 * typical_dim * lora_rank
    total_lora_params = lora_params_per_layer * num_layers
    
    # Memory (assuming float32 = 4 bytes)
    bytes_per_param = 4
    
    full_memory_mb = (full_finetune_params * bytes_per_param) / (1024 ** 2)
    lora_memory_mb = (total_lora_params * bytes_per_param) / (1024 ** 2)
    
    savings = {
        'full_params': full_finetune_params,
        'lora_params': total_lora_params,
        'param_reduction': 1 - (total_lora_params / full_finetune_params),
        'full_memory_mb': full_memory_mb,
        'lora_memory_mb': lora_memory_mb,
        'memory_savings_mb': full_memory_mb - lora_memory_mb,
        'memory_reduction': 1 - (lora_memory_mb / full_memory_mb)
    }
    
    return savings