import networkx as nx
from pyvis.network import Network
import torch
import torch.nn as nn
from models.lora import LinearWithLoRA

def build_model_graph(model, tracker, lora=False):
    """
    Build an interactive graph visualization of the model
    
    Args:
        model: PyTorch model
        tracker: MemoryTracker instance
        lora: Whether to show LoRA-specific coloring
    
    Returns:
        HTML string of the interactive graph
    """
    G = nx.DiGraph()
    
    # Track layers
    tracker.reset()
    
    # Add input node
    G.add_node("input", 
               label="Input\n[1, 8]",
               color="#90EE90",
               size=25,
               title="Input tensor")
    
    prev_node = "input"
    
    # Process each layer
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, LinearWithLoRA)):
            # Get layer info
            if isinstance(module, LinearWithLoRA):
                linear = module.linear
                in_f = linear.in_features
                out_f = linear.out_features
                weight_shape = f"[{out_f}, {in_f}]"
                is_lora = True
            else:
                in_f = module.in_features
                out_f = module.out_features
                weight_shape = f"[{out_f}, {in_f}]"
                is_lora = False
            
            # Add weight node
            weight_node = f"{name}_weight"
            if is_lora and lora:
                color = "#D3D3D3"  # Gray for frozen weights
                title = f"Frozen Weight {name}\nShape: {weight_shape}\n(Not trainable)"
            else:
                color = "#87CEEB"  # Sky blue
                title = f"Weight {name}\nShape: {weight_shape}"
            
            G.add_node(weight_node,
                      label=f"{name}.weight\n{weight_shape}",
                      color=color,
                      size=30,
                      title=title)
            
            # Add computation node
            comp_node = f"{name}_output"
            G.add_node(comp_node,
                      label=f"{name} output\n[1, {out_f}]",
                      color="#FFD700" if lora else "#87CEEB",
                      size=25,
                      title=f"Output of {name}")
            
            # Add edges
            G.add_edge(prev_node, weight_node, color="#666666")
            G.add_edge(weight_node, comp_node, color="#666666")
            
            prev_node = comp_node
            
        elif isinstance(module, nn.ReLU):
            # Add activation node
            act_node = f"{name}_output"
            G.add_node(act_node,
                      label=f"ReLU\n{name}",
                      color="#FFA500",
                      size=20,
                      title="ReLU activation")
            G.add_edge(prev_node, act_node, color="#666666")
            prev_node = act_node
    
    # Add output node
    G.add_node("output",
               label="Output\n[1, 4]",
               color="#FF6B6B",
               size=25,
               title="Final output")
    G.add_edge(prev_node, "output", color="#666666")
    
    # Create PyVis network
    net = Network(height="550px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    
    # Configure physics
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 150,
                "springConstant": 0.01,
                "nodeDistance": 200
            },
            "solver": "hierarchicalRepulsion"
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 200
            }
        }
    }
    """)
    
    return net.generate_html()


def build_lora_graph(model, tracker):
    """
    Build a graph showing LoRA injection details
    
    Args:
        model: PyTorch model with LoRA
        tracker: MemoryTracker instance
    
    Returns:
        HTML string of the interactive graph
    """
    G = nx.DiGraph()
    
    # Add input
    G.add_node("input",
               label="Input\n[1, 8]",
               color="#90EE90",
               size=25,
               title="Input tensor")
    
    prev_node = "input"
    
    for name, module in model.named_children():
        if isinstance(module, LinearWithLoRA):
            # Get dimensions
            in_f = module.linear.in_features
            out_f = module.linear.out_features
            rank = module.lora.rank
            
            # Frozen weight node
            w_node = f"{name}_W"
            G.add_node(w_node,
                      label=f"W (frozen)\n[{out_f}, {in_f}]",
                      color="#D3D3D3",
                      size=30,
                      title=f"Frozen original weight\nNot trainable")
            
            # LoRA A node
            a_node = f"{name}_A"
            G.add_node(a_node,
                      label=f"LoRA A\n[{in_f}, {rank}]",
                      color="#4169E1",
                      size=25,
                      title=f"LoRA down-projection\nTrainable\nRank: {rank}")
            
            # LoRA B node
            b_node = f"{name}_B"
            G.add_node(b_node,
                      label=f"LoRA B\n[{rank}, {out_f}]",
                      color="#4169E1",
                      size=25,
                      title=f"LoRA up-projection\nTrainable\nRank: {rank}")
            
            # Delta W node
            delta_node = f"{name}_delta"
            G.add_node(delta_node,
                      label=f"ΔW = B@A\n[{out_f}, {in_f}]",
                      color="#9370DB",
                      size=20,
                      title="Low-rank update")
            
            # Effective weight
            eff_node = f"{name}_effective"
            G.add_node(eff_node,
                      label=f"W_eff = W + ΔW\n[{out_f}, {in_f}]",
                      color="#FFD700",
                      size=35,
                      title="Effective combined weight")
            
            # Output
            out_node = f"{name}_output"
            G.add_node(out_node,
                      label=f"{name} output\n[1, {out_f}]",
                      color="#FFD700",
                      size=25,
                      title=f"Layer output")
            
            # Add edges
            G.add_edge(prev_node, w_node, color="#666666")
            G.add_edge(prev_node, a_node, color="#4169E1")
            G.add_edge(a_node, b_node, color="#4169E1", width=2)
            G.add_edge(b_node, delta_node, color="#9370DB", width=2)
            G.add_edge(w_node, eff_node, color="#666666")
            G.add_edge(delta_node, eff_node, color="#9370DB", width=2)
            G.add_edge(eff_node, out_node, color="#FFD700", width=3)
            
            prev_node = out_node
            
        elif isinstance(module, nn.Linear):
            # Regular linear layer
            in_f = module.in_features
            out_f = module.out_features
            
            w_node = f"{name}_W"
            G.add_node(w_node,
                      label=f"{name}\n[{out_f}, {in_f}]",
                      color="#87CEEB",
                      size=30)
            
            out_node = f"{name}_output"
            G.add_node(out_node,
                      label=f"{name} output\n[1, {out_f}]",
                      color="#87CEEB",
                      size=25)
            
            G.add_edge(prev_node, w_node, color="#666666")
            G.add_edge(w_node, out_node, color="#666666")
            prev_node = out_node
            
        elif isinstance(module, nn.ReLU):
            act_node = f"{name}"
            G.add_node(act_node,
                      label=f"ReLU",
                      color="#FFA500",
                      size=20)
            G.add_edge(prev_node, act_node, color="#666666")
            prev_node = act_node
    
    # Output
    G.add_node("output",
               label="Output\n[1, 4]",
               color="#FF6B6B",
               size=25)
    G.add_edge(prev_node, "output", color="#666666")
    
    # Create visualization
    net = Network(height="550px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 120,
                "springConstant": 0.01,
                "nodeDistance": 150
            },
            "solver": "hierarchicalRepulsion"
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
    
    return net.generate_html()