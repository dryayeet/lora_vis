import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models.toy_model import ToyMLP
from models.lora import inject_lora, get_lora_parameters
from visualizers.graph_builder import build_model_graph, build_lora_graph
from visualizers.memory_tracker import MemoryTracker, compare_memory
from visualizers.forward_animator import ForwardAnimator
from visualizers.backward_animator import BackwardAnimator
from utils.dataset import ToyDataset

st.set_page_config(page_title="LoRA Visualizer", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ToyMLP(input_dim=8, hidden_dim=16, output_dim=4)
    st.session_state.lora_enabled = False
    st.session_state.lora_rank = 2
    st.session_state.dataset = ToyDataset()
    st.session_state.trained = False

st.title("🧠 LoRA Fine-Tuning Simulator & Neural Memory Graph Visualizer")
st.markdown("*Interactive visualization of how LoRA modifies neural networks*")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    lora_enabled = st.checkbox("Enable LoRA", value=st.session_state.lora_enabled)
    
    if lora_enabled:
        lora_rank = st.slider("LoRA Rank", min_value=1, max_value=8, value=st.session_state.lora_rank)
        
        if not st.session_state.lora_enabled or lora_rank != st.session_state.lora_rank:
            st.session_state.model = ToyMLP(input_dim=8, hidden_dim=16, output_dim=4)
            inject_lora(st.session_state.model, rank=lora_rank)
            st.session_state.lora_enabled = True
            st.session_state.lora_rank = lora_rank
            st.session_state.trained = False
    else:
        if st.session_state.lora_enabled:
            st.session_state.model = ToyMLP(input_dim=8, hidden_dim=16, output_dim=4)
            st.session_state.lora_enabled = False
            st.session_state.trained = False

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Overview", 
    "🔧 LoRA Injection", 
    "💾 Memory Dashboard",
    "➡️ Forward Pass",
    "⬅️ Backward Pass",
    "🎯 Knowledge Change"
])

with tab1:
    st.header("Base Model Architecture")
    st.markdown("### Network Structure")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tracker = MemoryTracker()
        graph_html = build_model_graph(st.session_state.model, tracker, lora=False)
        st.components.v1.html(graph_html, height=600, scrolling=True)
    
    with col2:
        st.markdown("### Model Info")
        total_params = sum(p.numel() for p in st.session_state.model.parameters())
        st.metric("Total Parameters", f"{total_params:,}")
        st.metric("Input Dimension", 8)
        st.metric("Hidden Dimension", 16)
        st.metric("Output Dimension", 4)
        
        st.markdown("### Layer Details")
        for name, param in st.session_state.model.named_parameters():
            if 'lora' not in name:
                st.text(f"{name}: {list(param.shape)}")

with tab2:
    st.header("LoRA-Injected Model")
    
    if st.session_state.lora_enabled:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tracker = MemoryTracker()
            graph_html = build_lora_graph(st.session_state.model, tracker)
            st.components.v1.html(graph_html, height=600, scrolling=True)
        
        with col2:
            st.markdown("### LoRA Configuration")
            st.metric("LoRA Rank", st.session_state.lora_rank)
            
            lora_params = get_lora_parameters(st.session_state.model)
            st.metric("LoRA Parameters", f"{lora_params:,}")
            
            st.markdown("### LoRA Layers")
            for name, param in st.session_state.model.named_parameters():
                if 'lora' in name:
                    st.text(f"{name}: {list(param.shape)}")
            
            st.markdown("---")
            st.info("🔵 Blue nodes = Trainable LoRA matrices (A, B)")
            st.info("⚪ Gray nodes = Frozen original weights (W)")
            st.info("🟡 Gold nodes = Effective combined weight")
    else:
        st.warning("⚠️ Enable LoRA in the sidebar to see LoRA injection")

with tab3:
    st.header("Memory Comparison Dashboard")
    
    fig, stats = compare_memory(st.session_state.model, st.session_state.lora_enabled, st.session_state.lora_rank)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Statistics")
        for key, value in stats.items():
            if isinstance(value, dict):
                st.markdown(f"**{key}**")
                for k, v in value.items():
                    st.metric(k, v)
            else:
                st.metric(key, value)
        
        st.markdown("---")
        st.markdown("### Memory Efficiency")
        if st.session_state.lora_enabled:
            # Count only non-LoRA parameters
            base_params = sum(p.numel() for name, p in st.session_state.model.named_parameters() if 'lora' not in name)
            lora_params = get_lora_parameters(st.session_state.model)
            total_params = sum(p.numel() for p in st.session_state.model.parameters())
            
            efficiency = (1 - lora_params / total_params) * 100
            st.metric("Parameter Reduction", f"{efficiency:.1f}%")

with tab4:
    st.header("Forward Pass Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("▶️ Run Forward Pass", key="forward"):
            animator = ForwardAnimator(st.session_state.model, st.session_state.lora_enabled)
            
            # Create sample input
            x = torch.randn(1, 8)
            
            with st.spinner("Running forward pass..."):
                frames = animator.animate(x)
                
                # Display frames
                frame_placeholder = st.empty()
                for i, frame_html in enumerate(frames):
                    frame_placeholder.markdown(f"**Step {i+1}/{len(frames)}**")
                    st.components.v1.html(frame_html, height=500, scrolling=True)
                    
                st.success("✅ Forward pass complete!")
    
    with col2:
        st.markdown("### Input")
        x_input = torch.randn(1, 8)
        st.code(f"Shape: {list(x_input.shape)}\n{x_input.numpy()}")
        
        if st.session_state.lora_enabled:
            st.markdown("### LoRA Flow")
            st.markdown("""
            1. Input → Linear Layer
            2. W (frozen) + B @ A (trainable)
            3. Activation → Next Layer
            """)
        else:
            st.markdown("### Standard Flow")
            st.markdown("""
            1. Input → Linear Layer (W)
            2. Activation Function
            3. Output → Next Layer
            """)

with tab5:
    st.header("Gradient Flow Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("⬅️ Run Backward Pass", key="backward"):
            animator = BackwardAnimator(st.session_state.model, st.session_state.lora_enabled)
            
            # Create sample input and run forward
            x = torch.randn(1, 8)
            output = st.session_state.model(x)
            loss = output.sum()
            
            with st.spinner("Running backward pass..."):
                frames = animator.animate(loss)
                
                # Display frames
                frame_placeholder = st.empty()
                for i, frame_html in enumerate(frames):
                    frame_placeholder.markdown(f"**Step {i+1}/{len(frames)}**")
                    st.components.v1.html(frame_html, height=500, scrolling=True)
                    
                st.success("✅ Backward pass complete!")
    
    with col2:
        st.markdown("### Gradient Info")
        
        if st.session_state.lora_enabled:
            st.markdown("### ✅ Gradients Flow To:")
            st.markdown("- LoRA A matrices")
            st.markdown("- LoRA B matrices")
            
            st.markdown("### ❌ Gradients DON'T Flow To:")
            st.markdown("- Original W weights (frozen)")
            
            st.info("LoRA only updates low-rank adapters A and B, keeping the original model frozen!")
        else:
            st.markdown("### Gradients Flow To:")
            st.markdown("- All weight matrices")
            st.markdown("- All bias terms")

with tab6:
    st.header("Knowledge Change Simulator")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Fine-tune on Custom Data")
        
        user_input = st.text_area("Enter training text (one sample per line):", 
                                   "The cat sat on the mat\nThe dog played in the yard\nBirds fly in the sky",
                                   height=100)
        
        num_epochs = st.slider("Training Epochs", 1, 20, 5)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        
        if st.button("🎯 Fine-tune Model"):
            if not st.session_state.lora_enabled:
                st.error("⚠️ Please enable LoRA first!")
            else:
                with st.spinner("Training..."):
                    # Prepare data
                    samples = user_input.strip().split('\n')
                    st.session_state.dataset.add_samples(samples)
                    
                    # Get model outputs before training
                    x_test = torch.randn(1, 8)
                    with torch.no_grad():
                        output_before = st.session_state.model(x_test)
                    
                    # Training loop
                    optimizer = torch.optim.Adam(
                        [p for p in st.session_state.model.parameters() if p.requires_grad],
                        lr=learning_rate
                    )
                    
                    progress_bar = st.progress(0)
                    loss_history = []
                    
                    for epoch in range(num_epochs):
                        total_loss = 0
                        for sample in samples:
                            # Create dummy input/target from sample
                            x = torch.randn(1, 8)
                            target = torch.randn(1, 4)
                            
                            optimizer.zero_grad()
                            output = st.session_state.model(x)
                            loss = torch.nn.functional.mse_loss(output, target)
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                        
                        loss_history.append(total_loss / len(samples))
                        progress_bar.progress((epoch + 1) / num_epochs)
                    
                    # Get outputs after training
                    with torch.no_grad():
                        output_after = st.session_state.model(x_test)
                    
                    st.session_state.trained = True
                    st.session_state.output_before = output_before
                    st.session_state.output_after = output_after
                    st.session_state.loss_history = loss_history
                    
                    st.success(f"✅ Training complete! Trained on {len(samples)} samples for {num_epochs} epochs")
    
    with col2:
        st.markdown("### Training Results")
        
        if st.session_state.trained:
            # Loss plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(st.session_state.loss_history, marker='o')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Output comparison
            st.markdown("### Before vs After")
            
            diff = torch.abs(st.session_state.output_after - st.session_state.output_before)
            change_magnitude = diff.mean().item()
            
            st.metric("Output Change", f"{change_magnitude:.4f}")
            
            # Heatmap of LoRA matrices
            st.markdown("### LoRA Matrix Updates")
            
            for name, param in st.session_state.model.named_parameters():
                if 'lora_A' in name:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    im = ax.imshow(param.detach().numpy(), cmap='coolwarm', aspect='auto')
                    ax.set_title(f"{name}")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                    break
        else:
            st.info("👆 Train the model to see results")

# Footer
st.markdown("---")
st.markdown("*Built with PyTorch, NetworkX, PyVis, and Streamlit | CPU-only, no GPU required*")