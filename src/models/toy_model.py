import torch
import torch.nn as nn

class ToyMLP(nn.Module):
    """
    A simple 3-layer MLP for demonstration purposes.
    Lightweight enough to run on CPU with clear visualization.
    """
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=4):
        super(ToyMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Layer 1: input -> hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        
        # Layer 2: hidden -> hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        # Layer 3: hidden -> output
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize with small random weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Layer 1
        h1 = self.fc1(x)
        h1_activated = self.relu1(h1)
        
        # Layer 2
        h2 = self.fc2(h1_activated)
        h2_activated = self.relu2(h2)
        
        # Layer 3
        output = self.fc3(h2_activated)
        
        return output
    
    def forward_with_intermediates(self, x):
        """
        Forward pass that returns all intermediate activations
        Useful for visualization
        
        Returns:
            dict with all intermediate values
        """
        intermediates = {}
        
        # Input
        intermediates['input'] = x
        
        # Layer 1
        h1 = self.fc1(x)
        intermediates['fc1_output'] = h1
        
        h1_activated = self.relu1(h1)
        intermediates['relu1_output'] = h1_activated
        
        # Layer 2
        h2 = self.fc2(h1_activated)
        intermediates['fc2_output'] = h2
        
        h2_activated = self.relu2(h2)
        intermediates['relu2_output'] = h2_activated
        
        # Layer 3
        output = self.fc3(h2_activated)
        intermediates['fc3_output'] = output
        intermediates['final_output'] = output
        
        return output, intermediates
    
    def get_layer_info(self):
        """
        Get information about each layer
        
        Returns:
            List of dicts with layer information
        """
        layers = []
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                layer_info = {
                    'name': name,
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'weight_shape': list(module.weight.shape),
                    'bias_shape': list(module.bias.shape) if module.bias is not None else None,
                    'num_params': module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                }
                layers.append(layer_info)
        
        return layers
    
    def count_parameters(self, trainable_only=False):
        """
        Count total parameters in the model
        
        Args:
            trainable_only: If True, only count trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class ToyTransformer(nn.Module):
    """
    A minimal transformer for more advanced visualization (optional).
    Can be used instead of MLP if needed.
    """
    def __init__(self, vocab_size=100, d_model=32, nhead=2, num_layers=1):
        super(ToyTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                     dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        embedded = self.embedding(x)
        transformed = self.transformer(embedded)
        output = self.fc_out(transformed)
        return output