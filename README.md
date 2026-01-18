# LoRA Fine-Tuning Simulator & Neural Memory Graph Visualizer

An interactive visualization tool that explains how LoRA (Low-Rank Adaptation) fine-tuning works internally. Built with PyTorch, NetworkX, PyVis, and Streamlit.

## Features

1. **Model Overview** - Visualize base neural network architecture
2. **LoRA Injection** - See how LoRA matrices A and B are added to the model
3. **Forward Pass Animation** - Step-by-step visualization of data flowing through the network
4. **Backward Pass Animation** - See gradient flow and understand which parameters get updated
5. **Knowledge Change Simulator** - Fine-tune on custom data with **real-world tasks** (Sentiment Analysis, Text Classification, Text Similarity) and see how predictions change
6. **Memory Dashboard** - Compare memory usage: Full Fine-tune vs LoRA vs QLoRA

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this project

```bash
git clone <your-repo-url>
cd lora_visualizer
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run src/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Project Structure

```
lora_visualizer/
├── src/
│   ├── app.py                     # Main Streamlit application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── toy_model.py          # Base neural network (MLP)
│   │   └── lora.py                # LoRA implementation
│   ├── visualizers/
│   │   ├── __init__.py
│   │   ├── graph_builder.py       # Graph visualization
│   │   ├── memory_tracker.py      # Memory usage analysis
│   │   ├── forward_animator.py    # Forward pass animation
│   │   └── backward_animator.py   # Backward pass animation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Toy dataset utilities + TaskDataset for real-world tasks
│   │   └── math_utils.py          # Mathematical helper functions
│   └── assets/                    # Assets directory
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Usage

### Enable LoRA

- Use the sidebar checkbox to enable/disable LoRA
- Adjust the LoRA rank slider (1-8) to see how it affects the model

### Explore Tabs

#### Tab 1: Model Overview
- View the base model architecture
- See layer dimensions and parameter counts
- Understand the basic structure

#### Tab 2: LoRA Injection
- See how LoRA matrices A and B are attached
- Blue nodes represent trainable LoRA parameters
- Gray nodes represent frozen original weights
- Gold nodes represent effective combined weights

#### Tab 3: Forward Pass
- Click "Run Forward Pass" button
- Watch data flow through the network step-by-step
- See how LoRA modifies the computation

#### Tab 4: Backward Pass
- Click "Run Backward Pass" button
- Watch gradients flow backward
- Green nodes indicate nodes receiving gradients
- See how gradients skip frozen weights in LoRA mode

#### Tab 5: Knowledge Change
- **Select a real-world task** from the dropdown (Sentiment Analysis, Text Classification, Text Similarity, or Generic Regression)
- Enter custom training text (one sample per line)
- Set training epochs and learning rate
- Click "Fine-tune Model" to train
- View training loss curve
- **See meaningful output interpretation** based on task type:
  - Sentiment: Probability percentages for positive/negative
  - Classification: Category probabilities with visualizations
  - Similarity: Embedding vectors with cosine similarity scores
- Visualize updated LoRA matrices
- Compare model behavior before and after fine-tuning

#### Tab 6: Memory Dashboard
- Compare parameter counts across methods
- View memory usage bar charts
- See efficiency gains from LoRA

## Understanding LoRA

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:

1. Freezes the original model weights (W)
2. Adds trainable low-rank matrices A and B
3. Computes the effective weight as: W_effective = W + ΔW where ΔW = B @ A

### Why LoRA?

- **Memory Efficient**: Only train a small fraction of parameters
- **Fast Training**: Fewer parameters to update
- **Preserves Base Model**: Original weights remain unchanged
- **Composable**: Multiple LoRA adapters can be swapped

### Key Concepts

- **Rank (r)**: Dimensionality of the low-rank space. Lower rank = fewer parameters
- **A Matrix**: Down-projection from d dimensions to r dimensions
- **B Matrix**: Up-projection from r dimensions back to d dimensions
- **ΔW = B @ A**: The low-rank update to original weights

## Technical Details

### Model Architecture

The default toy model is a simple 3-layer MLP:
- Input: 8 dimensions
- Hidden Layer 1: 16 dimensions
- Hidden Layer 2: 16 dimensions
- Output: Variable (2-8 dimensions depending on selected task)

The output dimension automatically adjusts based on the selected task:
- **Sentiment Analysis**: 2D output (negative/positive probabilities)
- **Text Classification**: 4D output (category probabilities)
- **Text Similarity**: 8D output (embedding vectors)
- **Generic Regression**: 4D output (continuous values)

This small size allows for fast CPU-only computation, clear visualization, and easy understanding.

### LoRA Implementation

```python
# Simplified version
class LoRALayer:
    def __init__(self, in_features, out_features, rank):
        self.A = Parameter(torch.zeros(in_features, rank))
        self.B = Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x):
        return x @ self.A @ self.B.T  # Low-rank path
```

The full implementation includes proper initialization (A random, B zeros), scaling factor (alpha / rank), and integration with frozen base weights.

The base model uses Kaiming initialization (He initialization) for better performance with ReLU activations, which is more appropriate than Xavier initialization for deep networks with ReLU.

### Memory Calculations

Memory usage is estimated as:
```
Memory (MB) = (num_parameters × 4 bytes) / (1024²)
```

Where 4 bytes = float32 precision

## Visualization Features

### Color Coding

- **Green**: Input/Active nodes
- **Blue**: Trainable LoRA parameters
- **Gray**: Frozen base weights
- **Gold**: Effective combined weights
- **Orange**: Activation functions
- **Red**: Output
- **Lime**: Nodes receiving gradients

### Graph Layout

- Hierarchical left-to-right layout
- Interactive zoom and pan
- Hover over nodes for details
- Click and drag to rearrange

## Example Use Cases

### Learning LoRA Basics
- Start with LoRA disabled
- Enable LoRA and see the graph change
- Adjust rank to understand the trade-off

### Understanding Gradient Flow
- Run backward pass with LoRA disabled
- Enable LoRA and run again
- Compare which weights receive gradients

### Fine-tuning Experiments
- **Select different tasks** to see how LoRA adapts for different objectives
- Enter custom training data (examples provided for each task type)
- Try different ranks (1, 2, 4, 8) to see the trade-off between parameters and performance
- Observe how outputs change with meaningful interpretations:
  - Sentiment: See confidence scores for positive/negative classification
  - Classification: View category probability distributions
  - Similarity: Compare embedding vectors before/after training
- View updated LoRA matrices
- See how task-specific labels are generated from text

### Memory Analysis
- Compare parameter counts
- Understand memory savings
- See the impact of rank on efficiency

## Configuration

### Adjustable Parameters

In the sidebar:
- **Enable LoRA**: Toggle LoRA on/off
- **LoRA Rank**: Adjust from 1-8

In the Knowledge Change tab:
- **Task Type**: Select from Sentiment Analysis, Text Classification, Text Similarity, or Generic Regression
- **Training Text**: Custom data samples (task-specific examples are provided)
- **Epochs**: Number of training iterations
- **Learning Rate**: Step size for optimization

### Model Parameters

To modify the base model, edit `models/toy_model.py`:

```python
model = ToyMLP(
    input_dim=8,    # Change input size
    hidden_dim=16,  # Change hidden layer size
    output_dim=4    # Change output size
)
```

## Troubleshooting

### Common Issues

**Application won't start**
```bash
pip install -r requirements.txt --upgrade
```

**Graphs not displaying**
- Try a different browser (Chrome recommended)
- Clear browser cache

**Slow performance**
- The model is designed for CPU
- Reduce animation steps or model size if needed

### System Requirements

- **RAM**: 2GB minimum, 4GB recommended
- **CPU**: Any modern processor (no GPU needed)
- **Browser**: Chrome, Firefox, or Edge (latest versions)
- **Python**: 3.8 or higher

## Learning Resources

### Recommended Reading

1. "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
2. "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

### Understanding the Code

- `models/toy_model.py`: Start here to understand the base model
- `models/lora.py`: Read this to see how LoRA is implemented
- `visualizers/graph_builder.py`: Learn how graphs are constructed

## Contributing

To extend this project:

1. Add new model architectures: Edit `models/toy_model.py`
2. Improve visualizations: Modify files in `visualizers/`
3. Add new features: Extend `src/app.py` with new tabs
4. Optimize performance: Enhance animation rendering

## License

This project is for educational purposes. Feel free to use and modify for learning.

## Acknowledgments

- Built with PyTorch for neural network operations
- Streamlit for the interactive UI
- NetworkX and PyVis for graph visualization
- Matplotlib for plotting

## Support

If you encounter issues:

1. Check this README
2. Verify all dependencies are installed
3. Try restarting the application
4. Check Python version (3.8+)

## Real-World Tasks

The Knowledge Change Simulator now supports **real-world NLP tasks** with meaningful outputs:

- **😊 Sentiment Analysis**: Classify text as positive or negative sentiment (used in customer reviews, social media monitoring)
- **📂 Text Classification**: Categorize text into News, Review, Question, or Comment (used in document organization, email routing)
- **🔍 Text Similarity**: Generate embeddings for similarity comparison (used in search, duplicate detection, recommendations)
- **📊 Generic Regression**: Learn continuous mappings (used for general regression tasks)

Each task includes:
- Task-specific text encoding (TF-IDF-like features)
- Automatic label inference from text
- Meaningful output interpretation (probabilities, embeddings, etc.)
- Visual comparisons before/after training

See [TASKS.md](TASKS.md) for detailed explanations of each task type, use cases, and output interpretation.

## Educational Goals

This tool aims to help you:

- Understand LoRA architecture visually
- See gradient flow in real-time
- Compare memory efficiency
- Experiment with fine-tuning on **real-world tasks**
- Build intuition about parameter-efficient training
- See how neural network outputs can be interpreted for practical applications

For more information about LoRA and parameter-efficient fine-tuning, explore the papers and resources mentioned above.
