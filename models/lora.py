import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer
    
    Implements: h = W_0 x + (B @ A) x
    where:
        - W_0 is the frozen pretrained weight
        - A is a down-projection matrix (trainable)
        - B is an up-projection matrix (trainable)
        - @ is matrix multiplication
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRALayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        # A: (in_features, rank) - projects down to low-rank space
        # B: (rank, out_features) - projects back up to original space
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize A with random values, B with zeros
        # This ensures ΔW = B @ A starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        Compute the LoRA adaptation
        
        Args:
            x: Input tensor
        
        Returns:
            LoRA output: (B @ A) @ x, scaled by alpha/rank
        """
        # x @ A @ B^T, but we compute it as (x @ A) @ B^T for efficiency
        result = x @ self.lora_A  # (batch, rank)
        result = result @ self.lora_B.T  # (batch, out_features)
        result = result * self.scaling
        return result
    
    def get_delta_weight(self):
        """
        Get the delta weight matrix: ΔW = B @ A
        
        Returns:
            Delta weight matrix of shape (out_features, in_features)
        """
        return (self.lora_B.T @ self.lora_A.T) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation
    
    Wraps a frozen linear layer and adds LoRA matrices
    """
    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super(LinearWithLoRA, self).__init__()
        
        # Store the original linear layer (frozen)
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        """
        Forward pass: original linear + LoRA adaptation
        
        Returns:
            W_0 @ x + (B @ A) @ x
        """
        # Original frozen transformation
        base_output = self.linear(x)
        
        # LoRA adaptation
        lora_output = self.lora(x)
        
        return base_output + lora_output
    
    def get_effective_weight(self):
        """
        Get the effective weight: W = W_0 + ΔW
        
        Returns:
            Effective weight matrix
        """
        delta_w = self.lora.get_delta_weight()
        return self.linear.weight + delta_w


def inject_lora(model, rank=4, alpha=1.0, target_modules=None):
    """
    Inject LoRA into all Linear layers of a model
    
    Args:
        model: PyTorch model
        rank: Rank of LoRA matrices
        alpha: Scaling parameter
        target_modules: List of module names to inject LoRA into (None = all Linear layers)
    
    Returns:
        Modified model with LoRA injected
    """
    if target_modules is None:
        target_modules = ['fc1', 'fc2']  # Default for ToyMLP
    
    for name, module in model.named_children():
        if name in target_modules and isinstance(module, nn.Linear):
            # Replace with LoRA version
            lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha)
            setattr(model, name, lora_layer)
    
    return model


def get_lora_parameters(model):
    """
    Count the number of trainable LoRA parameters
    
    Args:
        model: PyTorch model with LoRA
    
    Returns:
        Number of LoRA parameters
    """
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params += param.numel()
    return lora_params


def get_lora_state_dict(model):
    """
    Extract only LoRA parameters from model
    
    Args:
        model: PyTorch model with LoRA
    
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_state[name] = param.data.clone()
    return lora_state


def merge_lora_weights(model):
    """
    Merge LoRA weights into the base model
    Creates W_merged = W_0 + B @ A
    
    Args:
        model: PyTorch model with LoRA
    
    Returns:
        Model with merged weights (LoRA removed)
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # Get effective weight
            effective_weight = module.get_effective_weight()
            
            # Create new linear layer with merged weights
            merged_linear = nn.Linear(
                module.linear.in_features,
                module.linear.out_features,
                bias=module.linear.bias is not None
            )
            merged_linear.weight.data = effective_weight
            if module.linear.bias is not None:
                merged_linear.bias.data = module.linear.bias.data.clone()
            
            # Replace in parent
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            module_name = name.rsplit('.', 1)[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, module_name, merged_linear)
            else:
                setattr(model, module_name, merged_linear)
    
    return model


def print_lora_info(model):
    """
    Print information about LoRA layers in the model
    
    Args:
        model: PyTorch model with LoRA
    """
    print("="*60)
    print("LoRA Configuration")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = get_lora_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print()
    
    print("LoRA Layers:")
    print("-"*60)
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            print(f"{name}:")
            print(f"  Rank: {module.lora.rank}")
            print(f"  A shape: {list(module.lora.lora_A.shape)}")
            print(f"  B shape: {list(module.lora.lora_B.shape)}")
            print(f"  Parameters: {module.lora.lora_A.numel() + module.lora.lora_B.numel():,}")
    print("="*60)