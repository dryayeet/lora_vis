import torch
import torch.nn as nn
import networkx as nx
from pyvis.network import Network
from models.lora import LinearWithLoRA

class BackwardAnimator:
    """
    Animate backward pass (gradient flow) through the network
    Shows which parameters receive gradients
    """
    def __init__(self, model, lora_enabled):
        self.model = model
        self.lora_enabled = lora_enabled
        self.frames = []
    
    def animate(self, loss):
        """
        Generate animation frames for backward pass
        
        Args:
            loss: Loss tensor to backpropagate from
        
        Returns:
            List of HTML frames showing the backward pass
        """
        self.frames = []
        
        # Perform backward pass
        loss.backward()
        
        # Frame 1: Loss computed
        self.frames.append(self._create_frame(
            active_nodes=['output'],
            step_desc="Step 1: Loss computed at output",
            show_grads=False
        ))
        
        # Build backward frames
        step = 2
        
        # Get layers in reverse order
        layer_names = [name for name, _ in self.model.named_children()]
        
        current_node = "output"
        
        for name in reversed(layer_names):
            module = getattr(self.model, name)
            
            if isinstance(module, LinearWithLoRA) and self.lora_enabled:
                # Gradients flow back through LoRA path
                
                # Output to effective weight
                self.frames.append(self._create_frame(
                    active_nodes=[current_node, f"{name}_effective"],
                    active_edges=[(f"{name}_effective", current_node)],
                    step_desc=f"Step {step}: Gradient flows to effective weight",
                    show_grads=True
                ))
                step += 1
                
                # Effective weight splits to W and delta
                self.frames.append(self._create_frame(
                    active_nodes=[f"{name}_effective", f"{name}_W", f"{name}_delta"],
                    active_edges=[(f"{name}_W", f"{name}_effective"), 
                                 (f"{name}_delta", f"{name}_effective")],
                    step_desc=f"Step {step}: Gradient splits - W (frozen), ΔW (flows)",
                    show_grads=True,
                    frozen_nodes=[f"{name}_W"]
                ))
                step += 1
                
                # Delta to B
                self.frames.append(self._create_frame(
                    active_nodes=[f"{name}_delta", f"{name}_B"],
                    active_edges=[(f"{name}_B", f"{name}_delta")],
                    step_desc=f"Step {step}: Gradient flows through B (trainable)",
                    show_grads=True
                ))
                step += 1
                
                # B to A
                self.frames.append(self._create_frame(
                    active_nodes=[f"{name}_B", f"{name}_A"],
                    active_edges=[(f"{name}_A", f"{name}_B")],
                    step_desc=f"Step {step}: Gradient flows through A (trainable)",
                    show_grads=True
                ))
                step += 1
                
                # A to previous layer
                prev_layer_output = self._get_prev_output(name, layer_names)
                self.frames.append(self._create_frame(
                    active_nodes=[f"{name}_A", prev_layer_output],
                    active_edges=[(prev_layer_output, f"{name}_A")],
                    step_desc=f"Step {step}: Gradient continues to previous layer",
                    show_grads=True
                ))
                step += 1
                
                current_node = prev_layer_output
                
            elif isinstance(module, nn.Linear):
                # Standard linear layer - all weights get gradients
                weight_node = f"{name}_weight"
                prev_layer_output = self._get_prev_output(name, layer_names)
                
                self.frames.append(self._create_frame(
                    active_nodes=[current_node, weight_node, prev_layer_output],
                    active_edges=[(weight_node, current_node), (prev_layer_output, weight_node)],
                    step_desc=f"Step {step}: Gradient through {name} (all weights trainable)",
                    show_grads=True
                ))
                step += 1
                
                current_node = prev_layer_output
            
            elif isinstance(module, nn.ReLU):
                # ReLU - gradient passes through
                prev_layer_output = self._get_prev_output(name, layer_names)
                
                self.frames.append(self._create_frame(
                    active_nodes=[current_node, prev_layer_output],
                    active_edges=[(prev_layer_output, current_node)],
                    step_desc=f"Step {step}: Gradient through ReLU",
                    show_grads=True
                ))
                step += 1
                
                current_node = prev_layer_output
        
        # Final frame: Summary
        trainable_nodes = []
        frozen_nodes = []
        
        if self.lora_enabled:
            for name, _ in self.model.named_children():
                if isinstance(getattr(self.model, name), LinearWithLoRA):
                    trainable_nodes.extend([f"{name}_A", f"{name}_B"])
                    frozen_nodes.append(f"{name}_W")
        
        self.frames.append(self._create_frame(
            active_nodes=trainable_nodes,
            frozen_nodes=frozen_nodes,
            step_desc=f"Step {step}: Backward pass complete! {'LoRA matrices updated, base weights frozen' if self.lora_enabled else 'All weights updated'}",
            show_grads=True
        ))
        
        return self.frames
    
    def _get_prev_output(self, current_layer, layer_names):
        """Get the output node of the previous layer"""
        try:
            idx = layer_names.index(current_layer)
            if idx > 0:
                return f"{layer_names[idx-1]}_output"
            else:
                return "input"
        except:
            return "input"
    
    def _create_frame(self, active_nodes=None, active_edges=None, frozen_nodes=None, 
                     step_desc="", show_grads=False):
        """
        Create a single frame of the animation
        
        Args:
            active_nodes: List of node IDs to highlight (receiving gradients)
            active_edges: List of edge tuples to highlight
            frozen_nodes: List of node IDs to mark as frozen (no gradients)
            step_desc: Description of this step
            show_grads: Whether to show gradient indicators
        
        Returns:
            HTML string of the graph
        """
        if active_nodes is None:
            active_nodes = []
        if active_edges is None:
            active_edges = []
        if frozen_nodes is None:
            frozen_nodes = []
        
        G = nx.DiGraph()
        
        # Build graph
        if self.lora_enabled:
            G = self._build_lora_graph()
        else:
            G = self._build_standard_graph()
        
        # Highlight active nodes (receiving gradients)
        for node in G.nodes():
            if node in active_nodes:
                if node in frozen_nodes:
                    # Frozen - show in gray with X
                    G.nodes[node]['color'] = '#A9A9A9'
                    G.nodes[node]['label'] = G.nodes[node].get('label', node) + '\n❌'
                else:
                    # Active - show in green
                    G.nodes[node]['color'] = '#32CD32'  # Lime green for gradient flow
                    if show_grads:
                        G.nodes[node]['label'] = G.nodes[node].get('label', node) + '\n∇'
                G.nodes[node]['size'] = G.nodes[node].get('size', 25) * 1.3
        
        # Mark frozen nodes even if not active
        for node in frozen_nodes:
            if node in G.nodes():
                G.nodes[node]['color'] = '#D3D3D3'
                if '❌' not in str(G.nodes[node].get('label', '')):
                    G.nodes[node]['label'] = G.nodes[node].get('label', node) + '\n(frozen)'
        
        # Create PyVis network
        net = Network(height="450px", width="100%", directed=True, notebook=False)
        net.from_nx(G)
        
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 120
                }
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 150
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true
                    }
                }
            }
        }
        """)
        
        html = net.generate_html()
        
        # Add step description
        legend = """
        <div style="text-align:left;padding:5px;font-size:12px;">
            <span style="color:#32CD32;">●</span> Receives gradient (∇) | 
            <span style="color:#D3D3D3;">●</span> Frozen (no gradient)
        </div>
        """
        html = html.replace('<body>', 
                           f'<body><div style="text-align:center;padding:10px;background:#f0f0f0;"><b>{step_desc}</b></div>{legend}')
        
        return html
    
    def _build_standard_graph(self):
        """Build graph for standard model (no LoRA)"""
        G = nx.DiGraph()
        
        G.add_node("input", label="Input", color="#90EE90", size=20)
        
        prev = "input"
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                w_node = f"{name}_weight"
                out_node = f"{name}_output"
                G.add_node(w_node, label=f"{name}.W", color="#87CEEB", size=25)
                G.add_node(out_node, label=f"{name}", color="#87CEEB", size=20)
                G.add_edge(prev, w_node)
                G.add_edge(w_node, out_node)
                prev = out_node
            elif isinstance(module, nn.ReLU):
                act_node = f"{name}_output"
                G.add_node(act_node, label="ReLU", color="#FFA500", size=20)
                G.add_edge(prev, act_node)
                prev = act_node
        
        G.add_node("output", label="Output", color="#FF6B6B", size=20)
        G.add_edge(prev, "output")
        
        return G
    
    def _build_lora_graph(self):
        """Build graph for LoRA model"""
        G = nx.DiGraph()
        
        G.add_node("input", label="Input", color="#90EE90", size=20)
        
        prev = "input"
        for name, module in self.model.named_children():
            if isinstance(module, LinearWithLoRA):
                G.add_node(f"{name}_W", label="W", color="#D3D3D3", size=25)
                G.add_node(f"{name}_A", label="A", color="#4169E1", size=20)
                G.add_node(f"{name}_B", label="B", color="#4169E1", size=20)
                G.add_node(f"{name}_delta", label="ΔW", color="#9370DB", size=18)
                G.add_node(f"{name}_effective", label="W_eff", color="#FFD700", size=28)
                G.add_node(f"{name}_output", label=name, color="#FFD700", size=20)
                
                G.add_edge(prev, f"{name}_W")
                G.add_edge(prev, f"{name}_A")
                G.add_edge(f"{name}_A", f"{name}_B")
                G.add_edge(f"{name}_B", f"{name}_delta")
                G.add_edge(f"{name}_W", f"{name}_effective")
                G.add_edge(f"{name}_delta", f"{name}_effective")
                G.add_edge(f"{name}_effective", f"{name}_output")
                
                prev = f"{name}_output"
            elif isinstance(module, nn.ReLU):
                G.add_node(f"{name}_output", label="ReLU", color="#FFA500", size=18)
                G.add_edge(prev, f"{name}_output")
                prev = f"{name}_output"
        
        G.add_node("output", label="Output", color="#FF6B6B", size=20)
        G.add_edge(prev, "output")
        
        return G