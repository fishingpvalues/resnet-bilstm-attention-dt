# ResNet-BiLSTM-Attention Digital Twin

A deep learning framework that combines Residual Networks (ResNet), Bidirectional Long Short-Term Memory (BiLSTM), and Attention mechanisms for validating manufacturing processes in Digital Twin applications.

## Overview

This project implements a hybrid deep learning architecture to analyze manufacturing data from CP Factory systems. It validates manufacturing processes by comparing real production data with simulated data from a digital twin to identify discrepancies and anomalies in process execution.

![Class Diagram](classes.png)

## Architecture

The model combines three advanced deep learning components:

1. **ResNet (Residual Network)**: Enables deep feature extraction while preventing vanishing gradients through skip connections
2. **BiLSTM (Bidirectional LSTM)**: Processes temporal sequences in both directions to capture manufacturing process dependencies
3. **Attention Mechanism**: Focuses the model on the most relevant parts of process sequences for improved prediction

## Dataset

The repository uses:

- Real factory data: `datasrc/real/real_factorydata.csv` and `real_factorydata_oclog.csv`
- Simulated data: `datasrc/sim/simulated_data_oclog.csv`

Data contains detailed manufacturing operations including:

- Process types and IDs
- Resource utilization and mapping
- Component and part information
- Timestamps and durations
- Order sequence information

## Features

- Process validation: Classification of valid vs. invalid manufacturing processes
- KPI calculation: Throughput, lead times, cycle times, setup times
- Feature engineering with time-based cyclical features (day of week, hour of day)
- Break detection (lunch breaks, night shifts)
- Holiday and weekend identification
- Baseline models (Decision Tree) and advanced deep learning models
- Visualization of model performance and validation metrics

## Requirements

- Python 3.12+
- UV package manager
- Dependencies as specified in pyproject.toml

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fishingpvalues/resnet-bilstm-attention-dt.git
    cd resnet-bilstm-attention-dt
    ```

2.  **Set up the environment and install dependencies:** Choose **one** of the following methods:

    **Method A: Using UV (Recommended for exact reproducibility)**

    This method uses the `uv.lock` file to install the exact dependency versions.

    ```bash
    # Create a virtual environment using uv
    uv venv

    # Activate the virtual environment
    # Linux/macOS:
    source .venv/bin/activate
    # Windows CMD:
    # .venv\Scripts\activate.bat
    # Windows PowerShell:
    # .venv\Scripts\Activate.ps1

    # Sync the environment using the uv.lock file
    uv sync
    ```

    **Method B: Using standard Python pip**

    This method uses `pyproject.toml` to install dependencies. Note that `pip` **will not** use the `uv.lock` file, so dependency versions might differ from the locked versions if newer compatible versions are available.

    ```bash
    # Create a virtual environment using Python's built-in venv
    python -m venv .venv

    # Activate the virtual environment
    # Linux/macOS:
    source .venv/bin/activate
    # Windows CMD:
    # .venv\Scripts\activate.bat
    # Windows PowerShell:
    # .venv\Scripts\Activate.ps1

    # Upgrade pip (optional but recommended)
    pip install --upgrade pip

    # Install the project and its dependencies from pyproject.toml
    # Use -e for an "editable" install (links to your source code)
    pip install -e .

    # Alternatively, for a standard install:
    # pip install .
    ```

## Usage

Once installed, run the main script:

```bash
python main.py

This command will execute two pipelines:

- **Baseline Pipeline (Decision Tree):** Uses traditional methods for classification.
- **Advanced Pipeline (ResNet-BiLSTM-Attention):** Employs a deep neural network using BiLSTM with attention mechanisms. CUDA support is used if available.

## Pipelines

- **Decision Tree Classifier:** Simple baseline model implementation.
- **ResNet-BiLSTM-Attention:** Advanced sequence model implemented under `src/models/resnet_bilstm_attn`.

## Key Files

- `main.py`: Entry point for running the full pipeline
- `src/validate.ipynb`: Jupyter notebook with detailed validation and analysis
- `src/models/resnet_bilstm_attn/`: Deep learning model implementation
- `src/models/decisiontree/`: Baseline model implementation
- `src/data/`: Data processing and feature engineering tools
- `src/utils/reporting.py`: Visualization and metrics reporting

## Decision Tree

- Fits fast because of time feature

## ResNet Bi-LSTM Attention Network

- 10 Epochs sufficient

## File Tree

```text
ResNet-BiLSTM-Attention-DT/
├── README.md [README.md](d:\resnet-bilstm-attention-dt\README.md)
├── LICENSE
├── datasrc/
│   ├── real/
│   │   ├── real_factorydata.csv
│   │   └── real_factorydata_oclog.csv
│   └── sim/
│       └── simulated_data_oclog.csv
├── src/
│   ├── main.py
│   ├── validate.ipynb
│   ├── connector/
│   │   └── ofact/
│   │       └── _put_in_ofact_project_folder/
│   │           └── output_structure.py
│   ├── models/
│   │   ├── resnet_bilstm_attn/     <— Deep learning model implementation
│   │   └── decisiontree/           <— Baseline model implementation
│   ├── data/                     <— Data processing and feature engineering tools
│   └── utils/
│       └── reporting.py
├── puml_output/
│   ├── classes.puml
│   └── packages.puml
└── decision_tree/                <— Contains decision tree related resources
```
