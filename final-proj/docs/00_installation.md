# Installation Guide

This guide covers the installation and setup of the Diabetes Classification System using the modern `uv` package manager.

## Prerequisites

- **Python**: 3.13 or higher
- **Operating System**: Linux, macOS, or Windows
- **Git**: For cloning the repository
- **Terminal**: Command-line access

## Why uv?

`uv` is an extremely fast Python package installer and resolver written in Rust. Benefits include:

- **10-100x faster** than pip
- **Reliable dependency resolution**
- **Automatic virtual environment management**
- **Lock file support** for reproducible installs
- **Drop-in replacement** for pip

## Installing uv

### Linux and macOS

```bash
# Using the standalone installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Or using Homebrew (macOS)
brew install uv
```

### Windows

```powershell
# Using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### Verify Installation

```bash
uv --version
# Expected output: uv 0.x.x
```

## Project Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/npesaras/big-data-analysis.git
cd big-data-analysis/final-proj
```

### Step 2: Initialize Environment

The `uv sync` command automatically:
- Creates a `.venv` virtual environment
- Installs all dependencies from `pyproject.toml`
- Generates a `uv.lock` file for reproducibility

```bash
uv sync
```

**Output:**
```
Resolved 52 packages in 0.5s
Installed 52 packages in 2.3s
 + streamlit==1.51.0
 + pandas==2.3.3
 + scikit-learn==1.7.2
 ...
```

### Step 3: Verify Installation

Test that all modules are correctly installed:

```bash
uv run python -c "
from src import config
from src.data_cleaning import load_diabetes_data
from src.model_selection import create_all_models
print('✅ All modules loaded successfully!')
print(f'Models configured: {len(config.MODELS_CONFIG)}')
print(f'Features: {len(config.FEATURE_COLUMNS)}')
"
```

**Expected output:**
```
✅ All modules loaded successfully!
Models configured: 9
Features: 8
```

## Running the Application

### Start the Streamlit Web App

```bash
uv run streamlit run main.py
```

The application will start and automatically open in your browser at:
```
http://localhost:8501
```

### Alternative: Activate Virtual Environment

While `uv run` executes commands in the environment automatically, you can also activate it manually:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

Then run commands normally:
```bash
streamlit run main.py
python scripts/train_classification.py
```

## Project Dependencies

Dependencies are defined in `pyproject.toml`:

### Core Dependencies

```toml
[project]
dependencies = [
    "streamlit>=1.51.0",
    "pandas>=2.3.3",
    "numpy>=2.3.5",
    "scikit-learn>=1.7.2",
    "joblib>=1.5.2",
    "plotly>=6.5.0",
    "seaborn>=0.13.2",
    "matplotlib>=3.10.7",
]
```

### Adding New Dependencies

```bash
# Add a new package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Example: Add pytest for testing
uv add --dev pytest
```

### Updating Dependencies

```bash
# Update all packages
uv sync --upgrade

# Update specific package
uv add <package-name>@latest
```

## Common Tasks

### Running Scripts

```bash
# Train classification models
uv run python scripts/train_classification.py

# Run custom training script
uv run python train_all_models.py

# Execute any Python file
uv run python <script.py>
```

### Installing Additional Packages

```bash
# The project already has these installed
uv add plotly seaborn matplotlib

# To add more packages
uv add requests beautifulsoup4
```

### Checking Installed Packages

```bash
# List all installed packages
uv pip list

# Show specific package info
uv pip show scikit-learn
```

## Troubleshooting

### Issue: `uv: command not found`

**Cause**: uv not in system PATH

**Solution 1**: Restart terminal after installation

**Solution 2**: Manually add to PATH

```bash
# Linux/macOS (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Windows: Add to System Environment Variables
# %USERPROFILE%\.cargo\bin
```

### Issue: Module import errors

**Cause**: Not using the virtual environment

**Solution**: Always use `uv run` prefix

```bash
# ✅ Correct
uv run python script.py
uv run streamlit run main.py

# ❌ Incorrect (uses system Python)
python script.py
streamlit run main.py
```

### Issue: Dependency conflicts

**Cause**: Corrupted lock file or environment

**Solution**: Clean and reinstall

```bash
# Remove virtual environment and lock file
rm -rf .venv uv.lock

# Reinstall everything
uv sync
```

### Issue: Permission denied

**Cause**: Insufficient permissions

**Solution**: Don't use sudo with uv

```bash
# ❌ Incorrect
sudo uv sync

# ✅ Correct
uv sync
```

### Issue: Slow package installation

**Cause**: Network issues or missing cache

**Solution**: Use offline mode if packages were installed before

```bash
uv sync --offline
```

## Development Setup

For contributing to the project, install development tools:

```bash
# Add development dependencies
uv add --dev pytest black flake8 mypy ipython

# Run tests
uv run pytest

# Format code
uv run black src/

# Lint code
uv run flake8 src/

# Type checking
uv run mypy src/

# Interactive Python shell
uv run ipython
```

## Environment Management

### Creating a Clean Environment

```bash
# Remove existing environment
rm -rf .venv

# Create fresh environment
uv sync
```

### Exporting Dependencies

```bash
# Generate requirements.txt (pip compatible)
uv pip compile pyproject.toml -o requirements.txt

# Generate for specific Python version
uv pip compile --python-version 3.13 pyproject.toml
```

### Using Different Python Versions

```bash
# Use specific Python version
uv venv --python 3.13

# Use system Python
uv venv --python python3.13
```

## Performance Tips

### Speed up installations

```bash
# Use parallel downloads
uv sync --no-cache

# Skip unused optional dependencies
uv sync --no-dev
```

### Reduce disk usage

```bash
# Clean uv cache
uv cache clean

# Remove unused environments
rm -rf .venv
```

## Platform-Specific Notes

### Linux

- May need to install Python development headers:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-dev

  # Fedora
  sudo dnf install python3-devel
  ```

### macOS

- Xcode Command Line Tools may be required:
  ```bash
  xcode-select --install
  ```

### Windows

- Visual C++ Build Tools may be needed for some packages
- Use PowerShell or Windows Terminal (not CMD)

## Next Steps

After successful installation:

1. **[Data Collection](01_data_collection.md)** - Learn about the dataset
2. **[Exploratory Analysis](02_exploratory_analysis.md)** - Visualize and analyze data
3. **[Train Models](06_model_training.md)** - Start training classification models

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv) - Official uv docs
- [Python Packaging](https://packaging.python.org/) - Python packaging guide
- [Virtual Environments](https://docs.python.org/3/tutorial/venv.html) - Python venv docs

---

**Need Help?** Check the [API Reference](08_api_reference.md) or open an issue on GitHub.
