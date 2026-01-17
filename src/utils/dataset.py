import torch
import numpy as np

class ToyDataset:
    """
    Simple toy dataset for demonstrating fine-tuning
    Generates synthetic input-output pairs
    """
    def __init__(self, input_dim=8, output_dim=4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.samples = []
        self.labels = []
        
        # Initialize with some default samples
        self._generate_default_samples()
    
    def _generate_default_samples(self, num_samples=10):
        """Generate synthetic samples"""
        for i in range(num_samples):
            # Random input
            x = torch.randn(self.input_dim)
            
            # Simple synthetic target (some function of input)
            y = torch.randn(self.output_dim)
            
            self.samples.append(x)
            self.labels.append(y)
    
    def add_samples(self, text_samples):
        """
        Add samples from text (for demonstration)
        
        Args:
            text_samples: List of text strings
        """
        for text in text_samples:
            # Simple hash-based encoding
            # In a real system, you'd use proper text encoding
            hash_val = hash(text)
            np.random.seed(hash_val % (2**31))
            
            x = torch.tensor(np.random.randn(self.input_dim), dtype=torch.float32)
            y = torch.tensor(np.random.randn(self.output_dim), dtype=torch.float32)
            
            self.samples.append(x)
            self.labels.append(y)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
    def get_batch(self, batch_size=4):
        """Get a random batch"""
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        batch_x = torch.stack([self.samples[i] for i in indices])
        batch_y = torch.stack([self.labels[i] for i in indices])
        
        return batch_x, batch_y
    
    def clear(self):
        """Clear all samples"""
        self.samples = []
        self.labels = []
        self._generate_default_samples()


def create_toy_text_pairs():
    """
    Create simple text pairs for demonstration
    
    Returns:
        List of (input_text, output_text) tuples
    """
    pairs = [
        ("The cat sat on the mat", "A feline rests on a rug"),
        ("The dog played in the yard", "A canine frolics outside"),
        ("Birds fly in the sky", "Avians soar through the air"),
        ("Fish swim in the ocean", "Marine life navigates the sea"),
        ("The sun shines brightly", "Solar rays illuminate intensely"),
    ]
    return pairs


def text_to_tensor(text, dim=8):
    """
    Convert text to a tensor (simple hash-based encoding)
    
    Args:
        text: Input text string
        dim: Dimension of output tensor
    
    Returns:
        Tensor representation of text
    """
    # Use hash for reproducible encoding
    hash_val = hash(text)
    np.random.seed(hash_val % (2**31))
    
    # Generate normalized random tensor
    tensor = torch.tensor(np.random.randn(dim), dtype=torch.float32)
    tensor = tensor / torch.norm(tensor)  # Normalize
    
    return tensor


class KnowledgeChangeTracker:
    """
    Track how model predictions change during fine-tuning
    """
    def __init__(self):
        self.before_outputs = {}
        self.after_outputs = {}
    
    def record_before(self, input_text, output):
        """Record output before fine-tuning"""
        self.before_outputs[input_text] = output.detach().clone()
    
    def record_after(self, input_text, output):
        """Record output after fine-tuning"""
        self.after_outputs[input_text] = output.detach().clone()
    
    def compute_change(self, input_text):
        """
        Compute the change in output for given input
        
        Returns:
            Change magnitude (L2 distance)
        """
        if input_text not in self.before_outputs or input_text not in self.after_outputs:
            return None
        
        before = self.before_outputs[input_text]
        after = self.after_outputs[input_text]
        
        change = torch.norm(after - before).item()
        return change
    
    def get_summary(self):
        """
        Get summary of all changes
        
        Returns:
            Dictionary with statistics
        """
        changes = []
        for text in self.before_outputs.keys():
            if text in self.after_outputs:
                change = self.compute_change(text)
                if change is not None:
                    changes.append(change)
        
        if not changes:
            return {
                'mean_change': 0.0,
                'max_change': 0.0,
                'min_change': 0.0,
                'num_samples': 0
            }
        
        return {
            'mean_change': np.mean(changes),
            'max_change': np.max(changes),
            'min_change': np.min(changes),
            'num_samples': len(changes)
        }