@echo off
REM LoRA Visualizer Setup Script for Windows

echo ================================================
echo LoRA Visualizer Setup Script (Windows)
echo ================================================
echo.

REM Check Python installation
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo [SUCCESS] Python detected
echo.

REM Create directory structure
echo [INFO] Creating directory structure...

mkdir lora_visualizer 2>nul
cd lora_visualizer

mkdir models 2>nul
mkdir visualizers 2>nul
mkdir utils 2>nul
mkdir assets 2>nul
mkdir assets\example_graphs 2>nul
mkdir .streamlit 2>nul

echo [SUCCESS] Directory structure created
echo.

REM Create __init__.py files
echo [INFO] Creating package files...

type nul > models\__init__.py
type nul > visualizers\__init__.py
type nul > utils\__init__.py

echo [SUCCESS] Package files created
echo.

REM Create .gitignore
echo [INFO] Creating .gitignore...

(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo.
echo # Virtual Environment
echo venv/
echo env/
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo .DS_Store
echo.
echo # Streamlit
echo .streamlit/secrets.toml
echo.
echo # Data
echo data/
echo *.csv
echo *.pkl
echo.
echo # Logs
echo *.log
) > .gitignore

echo [SUCCESS] .gitignore created
echo.

REM Create Streamlit config
echo [INFO] Creating Streamlit configuration...

(
echo [theme]
echo primaryColor = "#4169E1"
echo backgroundColor = "#FFFFFF"
echo secondaryBackgroundColor = "#F0F2F6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo headless = true
echo port = 8501
echo enableCORS = false
) > .streamlit\config.toml

echo [SUCCESS] Streamlit config created
echo.

REM Create requirements.txt
echo [INFO] Creating requirements.txt...

(
echo torch^>=2.0.0
echo streamlit^>=1.28.0
echo networkx^>=3.1
echo pyvis^>=0.3.2
echo matplotlib^>=3.7.0
echo numpy^>=1.24.0
) > requirements.txt

echo [SUCCESS] requirements.txt created
echo.

REM Create virtual environment
echo [INFO] Creating virtual environment...

if not exist venv (
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)
echo.

REM Activate virtual environment and install dependencies
echo [INFO] Installing dependencies...
echo This may take a few minutes...
echo.

call venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo [SUCCESS] All dependencies installed
) else (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo.

REM Create test script
echo [INFO] Creating test script...

(
echo import sys
echo.
echo packages = ['torch', 'streamlit', 'networkx', 'pyvis', 'matplotlib', 'numpy']
echo.
echo print("Testing package imports..."^)
echo print("-" * 50^)
echo.
echo all_success = True
echo for package in packages:
echo     try:
echo         __import__(package^)
echo         print(f"√ {package:20s} - OK"^)
echo     except ImportError:
echo         print(f"× {package:20s} - FAILED"^)
echo         all_success = False
echo.
echo print("-" * 50^)
echo.
echo if all_success:
echo     print("\n√ All packages imported successfully!"^)
echo     print("\nRun: streamlit run app.py"^)
echo else:
echo     print("\n× Some packages failed to import"^)
echo.
echo sys.exit(0 if all_success else 1^)
) > test_installation.py

echo [SUCCESS] Test script created
echo.

REM Run test
echo [INFO] Running installation test...
python test_installation.py

if %errorlevel% equ 0 (
    echo.
    echo ================================================
    echo [SUCCESS] Setup completed successfully!
    echo ================================================
    echo.
    echo Next steps:
    echo   1. Place all Python files in their directories
    echo   2. Activate virtual environment: venv\Scripts\activate
    echo   3. Run application: streamlit run app.py
    echo   4. Initialize Git: git init
    echo.
) else (
    echo [ERROR] Setup completed with errors
    pause
    exit /b 1
)

pause