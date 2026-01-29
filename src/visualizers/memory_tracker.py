import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models.lora import LinearWithLoRA, get_lora_parameters

class MemoryTracker:
    """
    Track memory usage of tensors and parameters in the model
    """
    def __init__(self):
        self.tensors = []
        self.reset()
    
    def reset(self):
        """Reset all tracked tensors"""
        self.tensors = []
    
    def track_tensor(self, name, tensor, trainable=True):
        """
        Track a tensor's memory usage
        
        Args:
            name: Name of the tensor
            tensor: PyTorch tensor
            trainable: Whether this tensor is trainable
        """
        if tensor is None:
            return
        
        info = {
            'name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'dtype': str(tensor.dtype),
            'size_bytes': tensor.numel() * tensor.element_size(),
            'size_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
            'trainable': trainable
        }
        self.tensors.append(info)
    
    def track_model(self, model):
        """
        Track all parameters in a model
        
        Args:
            model: PyTorch model
        """
        self.reset()
        
        for name, param in model.named_parameters():
            self.track_tensor(name, param, trainable=param.requires_grad)
    
    def get_summary(self):
        """
        Get summary statistics
        
        Returns:
            Dictionary with summary stats
        """
        if not self.tensors:
            return {
                'total_params': 0,
                'trainable_params': 0,
                'frozen_params': 0,
                'total_memory_mb': 0,
                'trainable_memory_mb': 0,
                'frozen_memory_mb': 0
            }
        
        total_params = sum(t['numel'] for t in self.tensors)
        trainable_params = sum(t['numel'] for t in self.tensors if t['trainable'])
        frozen_params = sum(t['numel'] for t in self.tensors if not t['trainable'])
        
        total_memory = sum(t['size_mb'] for t in self.tensors)
        trainable_memory = sum(t['size_mb'] for t in self.tensors if t['trainable'])
        frozen_memory = sum(t['size_mb'] for t in self.tensors if not t['trainable'])
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'total_memory_mb': total_memory,
            'trainable_memory_mb': trainable_memory,
            'frozen_memory_mb': frozen_memory
        }
    
    def print_summary(self):
        """Print a formatted summary"""
        summary = self.get_summary()
        
        print("="*60)
        print("Memory Usage Summary")
        print("="*60)
        print(f"Total Parameters: {summary['total_params']:,}")
        print(f"  Trainable: {summary['trainable_params']:,}")
        print(f"  Frozen: {summary['frozen_params']:,}")
        print()
        print(f"Total Memory: {summary['total_memory_mb']:.5f} MB")
        print(f"  Trainable: {summary['trainable_memory_mb']:.5f} MB")
        print(f"  Frozen: {summary['frozen_memory_mb']:.5f} MB")
        print("="*60)


def compare_memory(model, lora_enabled, lora_rank):
    """
    Compare memory usage between different training strategies
    
    Args:
        model: Current model
        lora_enabled: Whether LoRA is enabled
        lora_rank: Rank of LoRA (if enabled)
    
    Returns:
        matplotlib figure and statistics dictionary
    """
    # Calculate stats for different scenarios
    
    # Get base model params (all weights)
    base_params = 0
    for name, param in model.named_parameters():
        if 'lora' not in name:
            base_params += param.numel()
    
    # Full fine-tune: all parameters trainable
    full_finetune_params = base_params
    
    # LoRA: only LoRA parameters trainable
    if lora_enabled:
        lora_params = get_lora_parameters(model)
    else:
        # Calculate what LoRA params would be
        lora_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                # A: in_f x rank, B: rank x out_f
                lora_params += (in_f * lora_rank) + (lora_rank * out_f)
    
    # QLoRA: same as LoRA but with quantization (simulate 4-bit)
    # Frozen weights in 4-bit = base_params * 0.25
    # LoRA weights in full precision
    qlora_effective_params = (base_params * 0.25) + lora_params
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter count comparison
    methods = ['Full\nFine-tune', 'LoRA', 'QLoRA\n(simulated)']
    trainable_params = [full_finetune_params, lora_params, lora_params]
    frozen_params = [0, base_params, base_params * 0.25]
    
    x = np.arange(len(methods))
    width = 0.6
    
    bars1 = ax1.bar(x, trainable_params, width, label='Trainable', color='#4169E1')
    bars2 = ax1.bar(x, frozen_params, width, bottom=trainable_params, 
                    label='Frozen', color='#D3D3D3')
    
    ax1.set_ylabel('Parameters', fontsize=12)
    ax1.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (train, frozen) in enumerate(zip(trainable_params, frozen_params)):
        total = train + frozen
        ax1.text(i, total, f'{total:,}', ha='center', va='bottom', fontsize=9)
    
    # Memory usage comparison (approximate: 4 bytes per float32 param)
    bytes_per_param = 4
    memory_mb = [
        full_finetune_params * bytes_per_param / (1024**2),
        (lora_params + base_params) * bytes_per_param / (1024**2),
        (lora_params * bytes_per_param + base_params * 0.5) / (1024**2)  # 4-bit ≈ 0.5 bytes
    ]
    
    bars = ax2.bar(methods, memory_mb, color=['#FF6B6B', '#4169E1', '#90EE90'], width=0.6)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mem in zip(bars, memory_mb):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.3f} MB', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Statistics dictionary
    stats = {
        'Full Fine-tune': {
            'Trainable Params': f"{full_finetune_params:,}",
            'Memory': f"{memory_mb[0]:.3f} MB"
        },
        'LoRA': {
            'Trainable Params': f"{lora_params:,}",
            'Total Params': f"{base_params + lora_params:,}",
            'Memory': f"{memory_mb[1]:.3f} MB",
            'Reduction': f"{(1 - lora_params/full_finetune_params)*100:.1f}%"
        },
        'QLoRA (simulated)': {
            'Trainable Params': f"{lora_params:,}",
            'Memory': f"{memory_mb[2]:.3f} MB",
            'Reduction': f"{(1 - memory_mb[2]/memory_mb[0])*100:.1f}%"
        }
    }
    
    return fig, stats


def plot_parameter_distribution(model, lora_enabled):
    """
    Plot distribution of parameters across layers
    
    Args:
        model: PyTorch model
        lora_enabled: Whether LoRA is enabled
    
    Returns:
        matplotlib figure
    """
    layer_names = []
    param_counts = []
    trainable_counts = []
    
    for name, param in model.named_parameters():
        layer_names.append(name)
        param_counts.append(param.numel())
        trainable_counts.append(param.numel() if param.requires_grad else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layer_names))
    width = 0.35
    
    ax.bar(x - width/2, param_counts, width, label='Total', alpha=0.8)
    ax.bar(x + width/2, trainable_counts, width, label='Trainable', alpha=0.8)
    
    ax.set_ylabel('Parameter Count')
    ax.set_title('Parameter Distribution by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig