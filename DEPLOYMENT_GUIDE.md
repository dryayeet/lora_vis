# 🚀 Complete Deployment Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Project Structure](#project-structure)
4. [File Placement](#file-placement)
5. [Running the Application](#running-the-application)
6. [Git Repository Setup](#git-repository-setup)
7. [Troubleshooting](#troubleshooting)
8. [Development Workflow](#development-workflow)

---

## Quick Start

### For Linux/Mac:
```bash
# 1. Download and extract all files
# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run application
streamlit run app.py
```

### For Windows:
```bash
# 1. Download and extract all files
# 2. Run setup script
setup.bat

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Run application
streamlit run app.py
```

---

## Detailed Setup

### Prerequisites

**Required:**
- Python 3.8 or higher
- pip (comes with Python)
- 2GB free disk space
- 4GB RAM (recommended)

**Optional:**
- Git (for version control)
- Code editor (VS Code recommended)

### Installation Steps

#### Step 1: Create Project Directory

```bash
# Create main project folder
mkdir lora_visualizer
cd lora_visualizer
```

#### Step 2: Create Directory Structure

```bash
# Create subdirectories
mkdir -p models visualizers utils assets/example_graphs .streamlit
```

Your structure should look like:
```
lora_visualizer/
├── models/
├── visualizers/
├── utils/
├── assets/
│   └── example_graphs/
└── .streamlit/
```

#### Step 3: Place All Files

Copy or create the files in their respective locations (see [File Placement](#file-placement) below).

#### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

#### Step 5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

This will install:
- PyTorch (CPU version)
- Streamlit
- NetworkX
- PyVis
- Matplotlib
- NumPy

#### Step 6: Verify Installation

```bash
# Run test script
python test_installation.py
```

You should see all packages marked as "OK".

---

## Project Structure

### Complete File Tree

```
lora_visualizer/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── DEPLOYMENT_GUIDE.md            # This file
├── .gitignore                     # Git ignore rules
├── setup.sh                       # Linux/Mac setup script
├── setup.bat                      # Windows setup script
├── test_installation.py           # Installation test
│
├── models/                        # Neural network models
│   ├── __init__.py
│   ├── toy_model.py              # Base MLP model
│   └── lora.py                   # LoRA implementation
│
├── visualizers/                   # Visualization components
│   ├── __init__.py
│   ├── graph_builder.py          # Network graph builder
│   ├── memory_tracker.py         # Memory usage tracker
│   ├── forward_animator.py       # Forward pass animator
│   └── backward_animator.py      # Backward pass animator
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── dataset.py                # Toy dataset utilities
│   └── math_utils.py             # Math helper functions
│
├── assets/                        # Static assets
│   └── example_graphs/           # Example visualizations
│
└── .streamlit/                    # Streamlit configuration
    └── config.toml               # App configuration
```

---

## File Placement

### Step-by-Step File Organization

1. **Root Directory Files** (`lora_visualizer/`)
   - Place: `app.py`, `requirements.txt`, `README.md`, `.gitignore`
   - Place: `setup.sh`, `setup.bat`, `test_installation.py`

2. **Models Directory** (`lora_visualizer/models/`)
   - Create empty `__init__.py`
   - Place: `toy_model.py`, `lora.py`

3. **Visualizers Directory** (`lora_visualizer/visualizers/`)
   - Create empty `__init__.py`
   - Place: `graph_builder.py`, `memory_tracker.py`
   - Place: `forward_animator.py`, `backward_animator.py`

4. **Utils Directory** (`lora_visualizer/utils/`)
   - Create empty `__init__.py`
   - Place: `dataset.py`, `math_utils.py`

5. **Streamlit Config** (`lora_visualizer/.streamlit/`)
   - Create: `config.toml`

### Verification Checklist

After placing all files, verify with:

```bash
# Check all files are in place
ls -R  # Linux/Mac
tree /F  # Windows (if tree is installed)
dir /S  # Windows alternative
```

Ensure you have at least:
- ✅ 1 main file (app.py)
- ✅ 2 model files
- ✅ 4 visualizer files
- ✅ 2 utility files
- ✅ 3 package __init__.py files
- ✅ 1 requirements.txt
- ✅ 1 README.md

---

## Running the Application

### First Time Launch

```bash
# 1. Navigate to project directory
cd lora_visualizer

# 2. Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Launch Streamlit
streamlit run app.py
```

### Expected Output

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Browser Access

The app will automatically open in your default browser at `http://localhost:8501`

If it doesn't open automatically:
1. Copy the Local URL from terminal
2. Paste it in your browser

### Stopping the Application

Press `Ctrl + C` in the terminal.

---

## Git Repository Setup

### Initialize Local Repository

```bash
cd lora_visualizer

# Initialize Git
git init

# Check status
git status

# Stage all files
git add .

# First commit
git commit -m "Initial commit: Complete LoRA visualizer"
```

### Create GitHub Repository

#### Option 1: Via GitHub Website

1. Go to github.com
2. Click "New repository"
3. Name: `lora-visualizer`
4. Description: "Interactive LoRA fine-tuning visualization tool"
5. Keep "Public" or choose "Private"
6. **Do NOT** initialize with README (we have one)
7. Click "Create repository"

#### Option 2: Via GitHub CLI

```bash
# Install GitHub CLI first: https://cli.github.com/

gh repo create lora-visualizer --public --source=. --remote=origin
```

### Connect to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/lora-visualizer.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

### Verify Upload

Visit: `https://github.com/YOUR_USERNAME/lora-visualizer`

You should see all your files!

### Branch Strategy

Create development branch:
```bash
# Create and switch to develop branch
git checkout -b develop

# Push develop branch
git push -u origin develop
```

Recommended branches:
- `main` - Stable releases only
- `develop` - Active development
- `feature/*` - New features
- `bugfix/*` - Bug fixes

---

## Troubleshooting

### Common Issues

#### Issue 1: Module Not Found Error

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue 2: Streamlit Not Found

**Symptom:**
```
streamlit: command not found
```

**Solution:**
```bash
# Ensure venv is activated and reinstall
pip install streamlit

# Or use full path
./venv/bin/streamlit run app.py  # Linux/Mac
venv\Scripts\streamlit run app.py  # Windows
```

#### Issue 3: Port Already in Use

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# Linux/Mac:
lsof -ti:8501 | xargs kill -9
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

#### Issue 4: PyVis Graph Not Showing

**Symptom:**
Blank space where graph should be

**Solution:**
1. Try different browser (Chrome recommended)
2. Disable browser extensions
3. Clear browser cache
4. Check browser console for errors (F12)

#### Issue 5: Out of Memory

**Symptom:**
```
RuntimeError: out of memory
```

**Solution:**
The default model is small, but if you modified it:
```python
# In models/toy_model.py, reduce sizes:
model = ToyMLP(
    input_dim=4,    # Reduce from 8
    hidden_dim=8,   # Reduce from 16
    output_dim=2    # Reduce from 4
)
```

### Getting Help

If you encounter other issues:

1. **Check Logs:**
   ```bash
   # Streamlit logs are in terminal
   # Look for ERROR or WARNING messages
   ```

2. **Test Installation:**
   ```bash
   python test_installation.py
   ```

3. **Verify Python Version:**
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Check Dependencies:**
   ```bash
   pip list
   ```

---

## Development Workflow

### Making Changes

1. **Create Feature Branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Changes:**
   - Edit files in your code editor
   - Test changes by running app

3. **Test Your Changes:**
   ```bash
   streamlit run app.py
   ```

4. **Commit Changes:**
   ```bash
   git add .
   git commit -m "feat: Add my new feature"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin feature/my-new-feature
   ```

### Code Style

Follow these conventions:

**Python:**
- Use 4 spaces for indentation
- Follow PEP 8 style guide
- Add docstrings to functions
- Use type hints where helpful

**Git Commits:**
- Use conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `style:` for formatting
  - `refactor:` for code restructuring
  - `test:` for adding tests
  - `chore:` for maintenance

### Testing Changes

Before committing:

1. **Run the app:**
   ```bash
   streamlit run app.py
   ```

2. **Test all features:**
   - Enable/disable LoRA
   - Run forward pass
   - Run backward pass
   - Try fine-tuning

3. **Check for errors:**
   - Look at terminal output
   - Check browser console (F12)

### Creating Releases

When ready to release:

1. **Update version:**
   ```python
   # In app.py, update version number
   VERSION = "1.0.0"
   ```

2. **Create tag:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

3. **Create release on GitHub:**
   - Go to repository > Releases
   - Click "Create a new release"
   - Select your tag
   - Add release notes
   - Publish release

---

## Production Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub** (as shown above)

2. **Visit:** https://streamlit.io/cloud

3. **Sign in** with GitHub

4. **Deploy:**
   - Click "New app"
   - Select your repository
   - Select branch: `main`
   - Main file: `app.py`
   - Click "Deploy"

5. **Access:**
   Your app will be at: `https://your-app-name.streamlit.app`

### Deploy to Heroku

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create lora-visualizer

# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT" > Procfile

# Deploy
git push heroku main
```

### Deploy with Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

Build and run:
```bash
docker build -t lora-visualizer .
docker run -p 8501:8501 lora-visualizer
```

---

## Maintenance

### Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade streamlit

# Update all packages
pip install --upgrade -r requirements.txt

# Save new versions
pip freeze > requirements.txt
```

### Backup

Important files to backup:
- All `.py` files
- `requirements.txt`
- `.gitignore`
- `README.md`

If using Git, everything is already backed up in your repository!

---

## Support

For issues or questions:

1. Check this guide
2. Read the README.md
3. Check GitHub Issues
4. Create new issue with:
   - Python version
   - OS (Windows/Mac/Linux)
   - Error message
   - Steps to reproduce

---

**You're all set! Happy visualizing! 🎉**