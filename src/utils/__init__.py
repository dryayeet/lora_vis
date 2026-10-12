# ================================================================
# utils/__init__.py
"""
Utility functions and helpers
"""

from .dataset import (
    ToyDataset,
    create_toy_text_pairs,
    text_to_tensor,
    KnowledgeChangeTracker
)
from .math_utils import (
    compute_low_rank_approximation,
    compute_rank,
    frobenius_norm,
    compute_delta_weight_stats,
    relative_magnitude,
    gradient_norm,
    compute_parameter_efficiency,
    cosine_similarity,
    analyze_weight_changes,
    estimate_memory_savings
)

__all__ = [
    'ToyDataset',
    'create_toy_text_pairs',
    'text_to_tensor',
    'KnowledgeChangeTracker',
    'compute_low_rank_approximation',
    'compute_rank',
    'frobenius_norm',
    'compute_delta_weight_stats',
    'relative_magnitude',
    'gradient_norm',
    'compute_parameter_efficiency',
    'cosine_similarity',
    'analyze_weight_changes',
    'estimate_memory_savings'
]