"""
Configuration module for data paths and other settings.
Allows for environment variable overrides and relative path resolution.
"""

import os
from pathlib import Path

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Define default paths relative to project root
DEFAULT_PATHS = {
    "REAL_DATA": PROJECT_ROOT / "datasrc" / "real" / "real_factorydata_oclog.csv",
    "SIM_DATA": PROJECT_ROOT / "datasrc" / "sim" / "simulated_data_oclog.csv",
    "OUTPUT_DIR": PROJECT_ROOT / "hypothesis_results",
    "IDENTICAL_RESULTS_DIR": PROJECT_ROOT / "identical_data_test_results",
}


def get_data_path(path_key, default_path=None):
    """
    Get a data path from environment variable or use the default.

    Args:
        path_key: Key for the path in DEFAULT_PATHS or environment variable name
        default_path: Optional override for the default path

    Returns:
        Path object for the resolved data path
    """
    # If provided directly, use the default_path
    if default_path:
        path = Path(default_path)
    else:
        # Check environment variables first (with RESNETLSTM_ prefix)
        env_var = f"RESNETLSTM_{path_key}"
        if env_var in os.environ:
            path = Path(os.environ[env_var])
        # Otherwise use the default from the dictionary
        elif path_key in DEFAULT_PATHS:
            path = DEFAULT_PATHS[path_key]
        else:
            raise ValueError(f"Unknown path key: {path_key}")

    # Ensure the parent directory exists
    if path.parent and not path.parent.exists() and path.parent != Path("."):
        os.makedirs(path.parent, exist_ok=True)

    return path


def get_real_data_path():
    """Get the path to real factory data."""
    return get_data_path("REAL_DATA")


def get_sim_data_path():
    """Get the path to simulated factory data."""
    return get_data_path("SIM_DATA")


def get_output_dir():
    """Get the directory for hypothesis testing results."""
    path = get_data_path("OUTPUT_DIR")
    os.makedirs(path, exist_ok=True)
    return path


def get_identical_results_dir():
    """Get the directory for identical data test results."""
    path = get_data_path("IDENTICAL_RESULTS_DIR")
    os.makedirs(path, exist_ok=True)
    return path
