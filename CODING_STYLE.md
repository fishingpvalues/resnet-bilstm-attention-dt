# Coding Style Guide

This document describes the coding standards and linting setup for the ResNet-BiLSTM-Attention Digital Twin project. All contributors should follow these guidelines to ensure consistent code quality across the project.

## Code Formatting Tools

This project uses the following linting and formatting tools:

1. **Black** - Code formatting
2. **isort** - Import sorting
3. **flake8** - Linting and style guide enforcement
4. **mypy** - Static type checking
5. **nbstripout** - Strips output from Jupyter notebooks before committing

## Setup Development Environment

### Step 1: Install Pre-commit

We use pre-commit to automatically run linters and formatters on your code before each commit. To install it:

```bash
# With pip
pip install pre-commit

# Or with UV
uv pip install pre-commit
```

### Step 2: Install Pre-commit Hooks

After cloning the repository, install the pre-commit hooks:

```bash
pre-commit install
```

This will automatically run the configured linters and formatters when you commit.

### Step 3: Run Checks Manually (Optional)

To run the checks manually on all files:

```bash
pre-commit run --all-files
```

Or run a specific hook:

```bash
pre-commit run black --all-files
```

## Python Style Guidelines

### General Rules

1. **Line Length**: Maximum 100 characters
2. **Indentation**: 4 spaces (no tabs)
3. **Documentation**: Google-style docstrings
4. **Python Version**: Code should be compatible with Python 3.12+

### Imports

Imports should be organized in the following order:
1. Standard library imports
2. Related third-party imports
3. Local application/library imports

Example:
```python
# Standard library
import os
import sys
from typing import Dict, List

# Third-party
import numpy as np
import pandas as pd
import torch

# Local
from src.models.resnet_bilstm_attn import model
from src.utils import config
```

### Docstrings

Use Google-style docstrings:

```python
def sample_function(param1, param2):
    """A brief description of the function.
    
    More details about the function if needed spanning
    multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
    """
    pass
```

### Type Hints

Use type hints where they add clarity:

```python
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """Process the input data.
    
    Args:
        data: Input DataFrame to process
        threshold: Cutoff value for processing
        
    Returns:
        Dictionary of processed arrays
    """
    # Implementation
    pass
```

## Jupyter Notebook Standards

1. **Cell Output**: Strip output before committing (automatic with nbstripout)
2. **Documentation**: Include markdown cells explaining the purpose of code sections
3. **Imports**: Keep all imports at the top of the notebook
4. **Long Running Code**: Mark cells that take long to run with comments

## Common Issues and Solutions

### Black and flake8 conflicts

Black and flake8 occasionally conflict. We've configured flake8 to ignore:
- E203 (whitespace before ':')
- W503 (line break before binary operator)

These are intentional to maintain compatibility with Black.

### Pre-commit Failed Commits

If pre-commit blocks your commit, the tools will often automatically fix issues. 
You'll need to `git add` the modified files and try the commit again.

## Custom Configuration

Advanced users can inspect the following files for detailed configuration:
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `.linter-config.yaml`: Detailed settings for individual tools

## IDE Integration

### VS Code

Install the following extensions:
- Python (Microsoft)
- Black Formatter
- isort
- Flake8
- Pylance (for type checking)

Configure settings.json:
```json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.sortImports.args": ["--profile", "black"]
}
```

### PyCharm

1. Go to Settings → Tools → Python Integrated Tools
2. Set "Formatter" to "Black"
3. Install the "Save Actions" plugin to format on save
4. Configure Flake8 and mypy under "External Tools"