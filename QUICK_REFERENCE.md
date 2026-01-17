# Quick Reference Guide

## 📋 Essential Commands

### Initial Setup

```bash
# Linux/Mac
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
streamlit run src/app.py

# Windows
setup.bat
venv\Scripts\activate
streamlit run src/app.py
```

---

## 🐍 Python Virtual Environment

### Create
```bash
python3 -m venv venv          # Linux/Mac
python -m venv venv           # Windows
```

### Activate
```bash
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### Deactivate
```bash
deactivate                    # All platforms
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

### Standard Launch
```bash
streamlit run src/app.py
```

### Custom Port
```bash
streamlit run src/app.py --server.port 8502
```

### Network Access
```bash
streamlit run src/app.py --server.address 0.0.0.0
```

### Debug Mode
```bash
streamlit run src/app.py --logger.level debug
```

---

## 📦 Git Commands

### Initialize Repository
```bash
git init
git add .
git commit -m "Initial commit"
```

### Connect to GitHub
```bash
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
```

### Create Feature Branch
```bash
git checkout -b feature/feature-name
git add .
git commit -m "feat: Add feature description"
git push origin feature/feature-name
```

### Merge to Main
```bash
git checkout main
git merge feature/feature-name
git push origin main
```

### Tag Release
```bash
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

---

## 🔧 Troubleshooting Commands

### Check Python Version
```bash
python --version              # Should be 3.8+
```

### List Installed Packages
```bash
pip list
```

### Test Installation
```bash
python test_installation.py
```

### Kill Process on Port
```bash
# Linux/Mac
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Clear Streamlit Cache
```bash
streamlit cache clear
```

---

## 📁 Directory Structure Quick View

```
lora_visualizer/
├── src/
│   └── app.py                 # Main app
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── models/
│   ├── toy_model.py          # Neural network
│   └── lora.py               # LoRA implementation
├── visualizers/
│   ├── graph_builder.py      # Graph viz
│   ├── memory_tracker.py     # Memory tracking
│   ├── forward_animator.py   # Forward pass
│   └── backward_animator.py  # Backward pass
└── utils/
    ├── dataset.py            # Dataset utils
    └── math_utils.py         # Math helpers
```

---

## 🎯 Common Tasks

### Add New Visualization
1. Create file in `visualizers/`
2. Import in `visualizers/__init__.py`
3. Use in `app.py`

### Modify Model Architecture
Edit `models/toy_model.py`:
```python
model = ToyMLP(
    input_dim=8,    # Input size
    hidden_dim=16,  # Hidden size
    output_dim=4    # Output size
)
```

### Change LoRA Rank Range
Edit `src/app.py` sidebar:
```python
lora_rank = st.slider(
    "LoRA Rank", 
    min_value=1, 
    max_value=16,  # Change max value
    value=4
)
```

### Add Custom Dataset
Edit `utils/dataset.py`:
```python
def add_samples(self, text_samples):
    # Your custom logic here
    pass
```

---

## 🐛 Debug Checklist

- [ ] Virtual environment activated?
- [ ] All dependencies installed?
- [ ] Python version 3.8+?
- [ ] All files in correct directories?
- [ ] Port 8501 available?
- [ ] Browser supports JavaScript?
- [ ] Correct file permissions?

---

## 📊 Performance Optimization

### Reduce Model Size
```python
# In models/toy_model.py
ToyMLP(input_dim=4, hidden_dim=8, output_dim=2)
```

### Limit Animation Frames
```python
# In forward_animator.py
max_frames = 10  # Reduce from default
```

### Cache Data
```python
# In src/app.py, add caching
@st.cache_data
def load_data():
    return data
```

---

## 🔐 Security Notes

### Before Public Deployment

1. Remove sensitive data
2. Update `.gitignore`
3. Review dependencies
4. Set proper permissions
5. Use environment variables

### .gitignore Essentials
```
__pycache__/
*.pyc
venv/
.env
*.log
.streamlit/secrets.toml
```

---

## 📱 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Firefox | ✅ Full | Works well |
| Safari | ⚠️ Partial | May have issues |
| Edge | ✅ Full | Chromium-based |
| Opera | ✅ Full | Chromium-based |

---

## 🆘 Quick Fixes

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Port in use"
```bash
streamlit run src/app.py --server.port 8502
```

### "Permission denied"
```bash
chmod +x setup.sh  # Linux/Mac
```

### "Cannot import name"
```bash
# Check __init__.py files exist
touch models/__init__.py
touch visualizers/__init__.py
touch utils/__init__.py
```

### Graphs not showing
1. Try Chrome browser
2. Clear cache: Ctrl+Shift+Delete
3. Disable ad blockers
4. Check browser console (F12)

---

## 📚 Useful Links

- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org/docs
- **NetworkX Docs**: https://networkx.org
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

---

## 🎓 Learning Path

1. Start with README.md
2. Run the application
3. Explore each tab
4. Read inline comments in code
5. Modify parameters
6. Add new features
7. Share your improvements!

---

## 💡 Pro Tips

- Use `Ctrl+C` to stop Streamlit
- Use `R` in browser to refresh app
- Add `# type: ignore` for type hints
- Use `st.write()` for debugging
- Check terminal for error messages
- Keep code modular and documented

---

## 🎨 Customization Quick List

### Change Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4169E1"        # Blue
backgroundColor = "#FFFFFF"      # White
secondaryBackgroundColor = "#F0F2F6"  # Light gray
```

### Change Port
```bash
streamlit run src/app.py --server.port 8080
```

### Add Title
In `src/app.py`:
```python
st.set_page_config(
    page_title="My LoRA Visualizer",
    page_icon="🧠"
)
```

---

**Keep this file handy for quick reference! 🚀**