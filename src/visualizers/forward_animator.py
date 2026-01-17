import torch
import torch.nn as nn
import networkx as nx
from pyvis.network import Network
from models.lora import LinearWithLoRA

class ForwardAnimator:
    """
    Animate forward pass through the network
    Shows step-by-step activation flow
    """
    def __init__(self, model, lora_enabled):
        self.model = model
        self.lora_enabled = lora_enabled
        self.frames = []
    
    def animate(self, input_tensor):
        """
        Generate animation frames for forward pass
        
        Args:
            input_tensor: Input to the model
        
        Returns:
            List of HTML frames showing the forward pass
        """
        self.frames = []
        
        # Frame 1: Input arrives
        self.frames.append(self._create_frame(active_nodes=['input'], 
                                              step_desc="Step 1: Input received"))
        
        # Track forward pass
        step = 2
        prev_node = "input"
        
        for name, module in self.model.named_children():
            if isinstance(module, LinearWithLoRA):
                # LoRA path
                if self.lora_enabled:
                    # Activate W path
                    self.frames.append(self._create_frame(
                        active_nodes=[prev_node, f"{name}_W"],
                        active_edges=[(prev_node, f"{name}_W")],
                        step_desc=f"Step {step}: Processing through frozen weight W"
                    ))
                    step += 1
                    
                    # Activate A path
                    self.frames.append(self._create_frame(
                        active_nodes=[prev_node, f"{name}_A"],
                        active_edges=[(prev_node, f"{name}_A")],
                        step_desc=f"Step {step}: LoRA path - down-projection through A"
                    ))
                    step += 1
                    
                    # Activate B path
                    self.frames.append(self._create_frame(
                        active_nodes=[f"{name}_A", f"{name}_B"],
                        active_edges=[(f"{name}_A", f"{name}_B")],
                        step_desc=f"Step {step}: LoRA path - up-projection through B"
                    ))
                    step += 1
                    
                    # Combine to delta
                    self.frames.append(self._create_frame(
                        active_nodes=[f"{name}_B", f"{name}_delta"],
                        active_edges=[(f"{name}_B", f"{name}_delta")],
                        step_desc=f"Step {step}: Computing ΔW = B @ A"
                    ))
                    step += 1
                    
                    # Merge to effective weight
                    self.frames.append(self._create_frame(
                        active_nodes=[f"{name}_W", f"{name}_delta", f"{name}_effective"],
                        active_edges=[(f"{name}_W", f"{name}_effective"), 
                                     (f"{name}_delta", f"{name}_effective")],
                        step_desc=f"Step {step}: Combining W + ΔW = W_effective"
                    ))
                    step += 1
                    
                    # Output
                    self.frames.append(self._create_frame(
                        active_nodes=[f"{name}_effective", f"{name}_output"],
                        active_edges=[(f"{name}_effective", f"{name}_output")],
                        step_desc=f"Step {step}: Computing layer output"
                    ))
                    step += 1
                    
                    prev_node = f"{name}_output"
                else:
                    # Standard linear layer
                    self.frames.append(self._create_frame(
                        active_nodes=[prev_node, f"{name}_weight", f"{name}_output"],
                        active_edges=[(prev_node, f"{name}_weight"), 
                                     (f"{name}_weight", f"{name}_output")],
                        step_desc=f"Step {step}: Linear transformation"
                    ))
                    step += 1
                    prev_node = f"{name}_output"
            
            elif isinstance(module, nn.Linear):
                self.frames.append(self._create_frame(
                    active_nodes=[prev_node, f"{name}_weight", f"{name}_output"],
                    active_edges=[(prev_node, f"{name}_weight"), 
                                 (f"{name}_weight", f"{name}_output")],
                    step_desc=f"Step {step}: Linear transformation"
                ))
                step += 1
                prev_node = f"{name}_output"
            
            elif isinstance(module, nn.ReLU):
                self.frames.append(self._create_frame(
                    active_nodes=[prev_node, f"{name}_output"],
                    active_edges=[(prev_node, f"{name}_output")],
                    step_desc=f"Step {step}: ReLU activation"
                ))
                step += 1
                prev_node = f"{name}_output"
        
        # Final output
        self.frames.append(self._create_frame(
            active_nodes=[prev_node, "output"],
            active_edges=[(prev_node, "output")],
            step_desc=f"Step {step}: Final output produced!"
        ))
        
        return self.frames
    
    def _create_frame(self, active_nodes=None, active_edges=None, step_desc=""):
        """
        Create a single frame of the animation
        
        Args:
            active_nodes: List of node IDs to highlight
            active_edges: List of edge tuples to highlight
            step_desc: Description of this step
        
        Returns:
            HTML string of the graph
        """
        if active_nodes is None:
            active_nodes = []
        if active_edges is None:
            active_edges = []
        
        G = nx.DiGraph()
        
        # Build graph based on whether LoRA is enabled
        if self.lora_enabled:
            G = self._build_lora_graph()
        else:
            G = self._build_standard_graph()
        
        # Highlight active nodes
        for node in G.nodes():
            if node in active_nodes:
                G.nodes[node]['color'] = '#FF4500'  # Orange-red for active
                G.nodes[node]['size'] = G.nodes[node].get('size', 25) * 1.5
        
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
            }
        }
        """)
        
        html = net.generate_html()
        
        # Add step description at the top
        html = html.replace('<body>', f'<body><div style="text-align:center;padding:10px;background:#f0f0f0;"><b>{step_desc}</b></div>')
        
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
                # Add all LoRA nodes
                G.add_node(f"{name}_W", label="W", color="#D3D3D3", size=25)
                G.add_node(f"{name}_A", label="A", color="#4169E1", size=20)
                G.add_node(f"{name}_B", label="B", color="#4169E1", size=20)
                G.add_node(f"{name}_delta", label="ΔW", color="#9370DB", size=18)
                G.add_node(f"{name}_effective", label="W_eff", color="#FFD700", size=28)
                G.add_node(f"{name}_output", label=name, color="#FFD700", size=20)
                
                # Add edges
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