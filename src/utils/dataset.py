import torch
import numpy as np
import re
from typing import List, Tuple, Optional

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


def text_to_tensor(text, dim=8, method='hash'):
    """
    Convert text to a tensor using various encoding methods
    
    Args:
        text: Input text string
        dim: Dimension of output tensor
        method: 'hash' for hash-based, 'tfidf_like' for TF-IDF-like features
    
    Returns:
        Tensor representation of text
    """
    if method == 'hash':
        # Use hash for reproducible encoding
        hash_val = hash(text)
        np.random.seed(hash_val % (2**31))
        
        # Generate normalized random tensor
        tensor = torch.tensor(np.random.randn(dim), dtype=torch.float32)
        tensor = tensor / torch.norm(tensor)  # Normalize
        
        return tensor
    
    elif method == 'tfidf_like':
        # Simple feature extraction: word counts, length, etc.
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        features = np.zeros(dim, dtype=np.float32)
        
        # Feature 0: Text length (normalized)
        features[0] = min(len(text) / 100.0, 1.0)
        
        # Feature 1: Word count
        features[1] = min(len(words) / 20.0, 1.0)
        
        # Features 2-4: Common sentiment words (simple heuristic)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'happy', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'sad', 'horrible']
        neutral_words = ['the', 'is', 'are', 'was', 'were', 'this', 'that']
        
        features[2] = sum(1 for w in words if w in positive_words) / max(len(words), 1)
        features[3] = sum(1 for w in words if w in negative_words) / max(len(words), 1)
        features[4] = sum(1 for w in words if w in neutral_words) / max(len(words), 1)
        
        # Features 5-7: Hash-based for remaining dimensions
        hash_val = hash(text)
        np.random.seed(hash_val % (2**31))
        features[5:] = np.random.randn(dim - 5) * 0.1
        
        tensor = torch.tensor(features, dtype=torch.float32)
        if torch.norm(tensor) > 0:
            tensor = tensor / torch.norm(tensor)
        
        return tensor
    
    else:
        # Default to hash
        return text_to_tensor(text, dim, 'hash')


class TaskDataset:
    """
    Dataset for real-world NLP tasks with meaningful targets
    """
    
    TASK_TYPES = {
        'sentiment': {
            'output_dim': 2,  # [negative, positive] probabilities
            'labels': ['Negative', 'Positive'],
            'description': 'Classify text as positive or negative sentiment'
        },
        'classification': {
            'output_dim': 4,  # 4 categories
            'labels': ['News', 'Review', 'Question', 'Comment'],
            'description': 'Classify text into categories'
        },
        'similarity': {
            'output_dim': 8,  # Embedding vector for similarity comparison
            'labels': None,
            'description': 'Generate embeddings for text similarity'
        },
        'regression': {
            'output_dim': 4,  # Generic regression task
            'labels': None,
            'description': 'Generic regression task (current default)'
        }
    }
    
    def __init__(self, task_type='sentiment', input_dim=8):
        self.task_type = task_type
        self.task_config = self.TASK_TYPES.get(task_type, self.TASK_TYPES['regression'])
        self.input_dim = input_dim
        self.output_dim = self.task_config['output_dim']
        self.samples = []
        self.labels = []
        self.text_samples = []
    
    def infer_label_from_text(self, text: str) -> torch.Tensor:
        """Infer label from text based on task type"""
        text_lower = text.lower()
        
        if self.task_type == 'sentiment':
            # Simple sentiment heuristic
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                            'love', 'happy', 'best', 'fantastic', 'brilliant', 
                            'awesome', 'perfect', 'beautiful', 'nice', 'enjoy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 
                            'sad', 'horrible', 'disappointed', 'poor', 'boring',
                            'disgusting', 'ugly', 'hate', 'worst']
            
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            # Create probability distribution
            if pos_count > neg_count:
                label = torch.tensor([0.2, 0.8], dtype=torch.float32)  # Mostly positive
            elif neg_count > pos_count:
                label = torch.tensor([0.8, 0.2], dtype=torch.float32)  # Mostly negative
            else:
                label = torch.tensor([0.5, 0.5], dtype=torch.float32)  # Neutral
        
        elif self.task_type == 'classification':
            # Classify into 4 categories based on patterns
            text_lower = text_lower.lower()
            if any(w in text_lower for w in ['?', 'what', 'how', 'why', 'when', 'where']):
                label = torch.tensor([0.1, 0.1, 0.8, 0.0], dtype=torch.float32)  # Question
            elif any(w in text_lower for w in ['review', 'rating', 'star', 'bought', 'product']):
                label = torch.tensor([0.1, 0.8, 0.05, 0.05], dtype=torch.float32)  # Review
            elif any(w in text_lower for w in ['news', 'report', 'announced', 'breaking']):
                label = torch.tensor([0.8, 0.1, 0.05, 0.05], dtype=torch.float32)  # News
            else:
                label = torch.tensor([0.2, 0.2, 0.1, 0.5], dtype=torch.float32)  # Comment
        
        elif self.task_type == 'similarity':
            # For similarity, create a deterministic embedding based on text
            hash_val = hash(text)
            np.random.seed(hash_val % (2**31))
            label = torch.tensor(np.random.randn(self.output_dim), dtype=torch.float32)
            label = label / torch.norm(label) if torch.norm(label) > 0 else label
        
        else:  # regression
            hash_val = hash(text)
            np.random.seed(hash_val % (2**31))
            label = torch.tensor(np.random.randn(self.output_dim), dtype=torch.float32)
        
        return label
    
    def add_samples(self, text_samples: List[str], encoding_method='tfidf_like'):
        """Add text samples with task-specific encoding"""
        for text in text_samples:
            # Encode text to input tensor
            x = text_to_tensor(text, dim=self.input_dim, method=encoding_method)
            
            # Get task-specific label
            y = self.infer_label_from_text(text)
            
            self.samples.append(x)
            self.labels.append(y)
            self.text_samples.append(text)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
    def get_task_info(self):
        """Get information about the current task"""
        return {
            'type': self.task_type,
            'output_dim': self.output_dim,
            'labels': self.task_config.get('labels'),
            'description': self.task_config.get('description')
        }


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