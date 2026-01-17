# models/__init__.py
"""
Neural network models and LoRA implementation
"""

from .toy_model import ToyMLP, ToyTransformer
from .lora import (
    LoRALayer,
    LinearWithLoRA,
    inject_lora,
    get_lora_parameters,
    get_lora_state_dict,
    merge_lora_weights,
    print_lora_info
)

__all__ = [
    'ToyMLP',
    'ToyTransformer',
    'LoRALayer',
    'LinearWithLoRA',
    'inject_lora',
    'get_lora_parameters',
    'get_lora_state_dict',
    'merge_lora_weights',
    'print_lora_info'
]

