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
from utils.dataset import ToyDataset, TaskDataset

st.set_page_config(page_title="LoRA Visualizer", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ToyMLP(input_dim=8, hidden_dim=16, output_dim=4)
    st.session_state.lora_enabled = False
    st.session_state.lora_rank = 2
    st.session_state.dataset = ToyDataset()
    st.session_state.trained = False
    st.session_state.task_output_dim = 4
    st.session_state.task_type = 'regression'

st.title("LoRA Fine-Tuning Simulator & Neural Memory Graph Visualizer")
st.markdown("*Interactive visualization of how LoRA modifies neural networks*")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Overview", 
    "LoRA Injection", 
    "Forward & Backward Pass",
    "Knowledge Change",
    "Memory Dashboard"
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
            st.info("Blue nodes = Trainable LoRA matrices (A, B)")
            st.info("Gray nodes = Frozen original weights (W)")
            st.info("Gold nodes = Effective combined weight")
    else:
        st.warning("Enable LoRA in the sidebar to see LoRA injection")

with tab3:
    st.header("Forward & Backward Pass Simulation")
    
    # Initialize session state for frames
    if 'forward_frames' not in st.session_state:
        st.session_state.forward_frames = []
    if 'backward_frames' not in st.session_state:
        st.session_state.backward_frames = []
    if 'pass_type' not in st.session_state:
        st.session_state.pass_type = 'forward'  # 'forward' or 'backward'
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Single button to run both passes
        if st.button("Run Forward & Backward Pass", key="run_passes"):
            # Reset gradients
            for param in st.session_state.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Create sample input
            x = torch.randn(1, 8)
            
            with st.spinner("Running forward pass..."):
                forward_animator = ForwardAnimator(st.session_state.model, st.session_state.lora_enabled)
                st.session_state.forward_frames = forward_animator.animate(x)
            
            with st.spinner("Running backward pass..."):
                # Run forward to get output
                output = st.session_state.model(x)
                loss = output.sum()
                
                backward_animator = BackwardAnimator(st.session_state.model, st.session_state.lora_enabled)
                st.session_state.backward_frames = backward_animator.animate(loss)
            
            st.session_state.pass_type = 'forward'  # Start with forward
            st.success("Both passes complete! Use the slider below to navigate.")
        
        # Select which pass to view
        pass_selection = st.radio(
            "View Pass:",
            ["Forward Pass", "Backward Pass"],
            index=0 if st.session_state.pass_type == 'forward' else 1,
            horizontal=True,
            key="pass_selection"
        )
        
        # Determine which frames to show
        if pass_selection == "Forward Pass":
            frames = st.session_state.forward_frames
            pass_type = 'forward'
            pass_label = "Forward"
        else:
            frames = st.session_state.backward_frames
            pass_type = 'backward'
            pass_label = "Backward"
        
        # Display frames with slider if available
        if frames:
            st.session_state.pass_type = pass_type
            
            # Slider to navigate frames (slider manages its own state)
            slider_key = f"frame_slider_{pass_type}"
            
            # Get initial value from session state if exists, otherwise 0
            initial_value = st.session_state.get(slider_key, 0)
            # Ensure initial value is within bounds
            if initial_value >= len(frames):
                initial_value = 0
            
            # Slider to navigate frames
            frame_idx = st.slider(
                f"{pass_label} Pass Step",
                min_value=0,
                max_value=len(frames) - 1 if len(frames) > 0 else 0,
                value=initial_value,
                key=slider_key,
                help="Use the slider to navigate through the animation steps"
            )
            
            # Display current frame
            if frame_idx < len(frames) and len(frames) > 0:
                st.markdown(f"**{pass_label} Pass - Step {frame_idx + 1}/{len(frames)}**")
                st.components.v1.html(frames[frame_idx], height=500, scrolling=True)
        else:
            st.info("Click 'Run Forward & Backward Pass' to generate the animation.")
    
    with col2:
        st.markdown("### Information")
        
        # Input info
        st.markdown("### Input")
        x_input = torch.randn(1, 8)
        st.code(f"Shape: {list(x_input.shape)}\n{x_input.numpy()}")
        
        # Forward pass info
        if st.session_state.lora_enabled:
            st.markdown("### Forward Pass Flow")
            st.markdown("""
            1. Input → Linear Layer
            2. W (frozen) + B @ A (trainable)
            3. Activation → Next Layer
            """)
        else:
            st.markdown("### Forward Pass Flow")
            st.markdown("""
            1. Input → Linear Layer (W)
            2. Activation Function
            3. Output → Next Layer
            """)
        
        st.markdown("---")
        
        # Backward pass info
        st.markdown("### Backward Pass Info")
        
        if st.session_state.lora_enabled:
            st.markdown("### Gradients Flow To:")
            st.markdown("- LoRA A matrices")
            st.markdown("- LoRA B matrices")
            
            st.markdown("### Gradients DON'T Flow To:")
            st.markdown("- Original W weights (frozen)")
            
            st.info("LoRA only updates low-rank adapters A and B, keeping the original model frozen!")
        else:
            st.markdown("### Gradients Flow To:")
            st.markdown("- All weight matrices")
            st.markdown("- All bias terms")

with tab4:
    st.header("Knowledge Change Simulator")
    
    # Task selection
    task_type = st.selectbox(
        "Select Task Type",
        options=['sentiment', 'classification', 'similarity', 'regression'],
        format_func=lambda x: {
            'sentiment': '😊 Sentiment Analysis (Positive/Negative)',
            'classification': '📂 Text Classification (Categories)',
            'similarity': '🔍 Text Similarity (Embeddings)',
            'regression': '📊 Generic Regression'
        }.get(x, x),
        key='task_type'
    )
    
    # Get task info
    temp_dataset = TaskDataset(task_type=task_type)
    task_info = temp_dataset.get_task_info()
    st.info(f"**Task:** {task_info['description']} | **Output:** {task_info['output_dim']}D vector")
    
    # Update model output dimension if needed
    if 'task_output_dim' not in st.session_state or st.session_state.task_output_dim != task_info['output_dim']:
        st.session_state.model = ToyMLP(input_dim=8, hidden_dim=16, output_dim=task_info['output_dim'])
        if st.session_state.lora_enabled:
            inject_lora(st.session_state.model, rank=st.session_state.lora_rank)
        st.session_state.task_output_dim = task_info['output_dim']
    
    # Default examples based on task
    default_examples = {
        'sentiment': "I love this amazing product!\nThis is terrible and awful\nGreat service, very happy\nHate this worst experience",
        'classification': "What is the weather today?\nI reviewed the new smartphone and it's excellent\nBreaking news: Scientists discover new planet\nThis movie was fantastic!",
        'similarity': "The cat sat on the mat\nThe dog played in the yard\nBirds fly in the sky\nA feline rests on a rug",
        'regression': "The cat sat on the mat\nThe dog played in the yard\nBirds fly in the sky"
    }
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Fine-tune on Custom Data")
        
        user_input = st.text_area("Enter training text (one sample per line):", 
                                   default_examples.get(task_type, "The cat sat on the mat\nThe dog played in the yard"),
                                   height=100)
        
        num_epochs = st.slider("Training Epochs", 1, 20, 5)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        
        if st.button("Fine-tune Model"):
            if not st.session_state.lora_enabled:
                st.error("Please enable LoRA first!")
            else:
                with st.spinner("Training..."):
                    # Prepare data with task-specific dataset
                    samples = [s.strip() for s in user_input.strip().split('\n') if s.strip()]
                    task_dataset = TaskDataset(task_type=task_type)
                    task_dataset.add_samples(samples, encoding_method='tfidf_like')
                    
                    training_data = list(zip(task_dataset.samples, task_dataset.labels))
                    
                    # Get a test input for before/after comparison (first sample)
                    x_test = training_data[0][0].unsqueeze(0) if training_data else torch.randn(1, 8)
                    test_text = samples[0] if samples else "test"
                    
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
                        for x_encoded, y_target in training_data:
                            # Use the encoded tensors from the dataset
                            x = x_encoded.unsqueeze(0)  # Add batch dimension
                            target = y_target.unsqueeze(0)  # Add batch dimension
                            
                            optimizer.zero_grad()
                            output = st.session_state.model(x)
                            loss = torch.nn.functional.mse_loss(output, target)
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                        
                        loss_history.append(total_loss / len(training_data))
                        progress_bar.progress((epoch + 1) / num_epochs)
                    
                    # Get outputs after training (on same test input)
                    with torch.no_grad():
                        output_after = st.session_state.model(x_test)
                    
                    st.session_state.trained = True
                    st.session_state.output_before = output_before
                    st.session_state.output_after = output_after
                    st.session_state.loss_history = loss_history
                    st.session_state.test_text = test_text
                    st.session_state.training_task_type = task_type  # Use different key to avoid widget conflict
                    st.session_state.task_labels = task_info.get('labels')
                    
                    st.success(f"Training complete! Trained on {len(samples)} samples for {num_epochs} epochs")
    
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
            
            # Output interpretation based on task
            st.markdown("### Output Interpretation")
            
            # Use training_task_type if available (from last training), otherwise use current task_type from widget
            current_task = st.session_state.get('training_task_type', task_type)
            task_labels = st.session_state.get('task_labels')
            
            if current_task == 'sentiment' and task_labels:
                # Show sentiment probabilities
                st.markdown(f"**Test Text:** *{st.session_state.get('test_text', 'N/A')}*")
                
                output_before_vals = torch.softmax(st.session_state.output_before, dim=1)[0]
                output_after_vals = torch.softmax(st.session_state.output_after, dim=1)[0]
                
                col_before, col_after = st.columns(2)
                with col_before:
                    st.markdown("**Before Training:**")
                    for i, label in enumerate(task_labels):
                        st.metric(label, f"{output_before_vals[i].item():.2%}")
                
                with col_after:
                    st.markdown("**After Training:**")
                    for i, label in enumerate(task_labels):
                        st.metric(label, f"{output_after_vals[i].item():.2%}")
                
                # Show prediction
                pred_before = task_labels[torch.argmax(output_before_vals)]
                pred_after = task_labels[torch.argmax(output_after_vals)]
                st.markdown(f"**Prediction Before:** {pred_before} | **After:** {pred_after}")
            
            elif current_task == 'classification' and task_labels:
                # Show classification probabilities
                st.markdown(f"**Test Text:** *{st.session_state.get('test_text', 'N/A')}*")
                
                output_before_vals = torch.softmax(st.session_state.output_before, dim=1)[0]
                output_after_vals = torch.softmax(st.session_state.output_after, dim=1)[0]
                
                # Bar chart comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                x_pos = np.arange(len(task_labels))
                
                ax1.bar(x_pos, output_before_vals.detach().numpy())
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(task_labels, rotation=45, ha='right')
                ax1.set_ylabel("Probability")
                ax1.set_title("Before Training")
                ax1.set_ylim([0, 1])
                
                ax2.bar(x_pos, output_after_vals.detach().numpy())
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(task_labels, rotation=45, ha='right')
                ax2.set_ylabel("Probability")
                ax2.set_title("After Training")
                ax2.set_ylim([0, 1])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif current_task == 'similarity':
                # Show embedding similarity
                st.markdown(f"**Test Text:** *{st.session_state.get('test_text', 'N/A')}*")
                st.markdown("**Output:** Embedding vector (normalized)")
                st.code(f"Before: {st.session_state.output_before[0][:4].numpy()}\nAfter:  {st.session_state.output_after[0][:4].numpy()}")
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    st.session_state.output_before, 
                    st.session_state.output_after, 
                    dim=1
                ).item()
                st.metric("Cosine Similarity", f"{cos_sim:.4f}")
                st.caption("Higher similarity (closer to 1.0) means embeddings changed less")
            
            else:
                # Generic output comparison
                st.markdown("### Before vs After")
                diff = torch.abs(st.session_state.output_after - st.session_state.output_before)
                change_magnitude = diff.mean().item()
                st.metric("Output Change", f"{change_magnitude:.4f}")
                st.code(f"Before: {st.session_state.output_before[0][:4].numpy()}\nAfter:  {st.session_state.output_after[0][:4].numpy()}")
            
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
            st.info("Train the model to see results")
            st.markdown("### Real-World Use Cases")
            st.markdown("""
            **Sentiment Analysis**: Classify customer reviews, social media posts, feedback as positive/negative
            
            **Text Classification**: Categorize documents, emails, articles into topics (news, reviews, questions)
            
            **Text Similarity**: Generate embeddings for search, recommendation, duplicate detection
            
            **Regression**: Generic task for learning continuous outputs from inputs
            """)

with tab5:
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

# Footer
st.markdown("---")
st.markdown("*Built with PyTorch, NetworkX, PyVis, and Streamlit | CPU-only, no GPU required*")