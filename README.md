# ResNet-BiLSTM-Attention Digital Twin

A deep learning framework that combines Residual Networks (ResNet), Bidirectional Long Short-Term Memory (BiLSTM), and Attention mechanisms for validating manufacturing processes in Digital Twin applications.

## Overview

This project implements a hybrid deep learning architecture to analyze manufacturing data from CP Factory systems. It validates manufacturing processes by comparing real production data with simulated data from a digital twin to identify discrepancies and anomalies in process execution.

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

```bash
# Clone the repository
git clone https://github.com/yourusername/resnet-bilstm-attention-dt.git
cd resnet-bilstm-attention-dt

# Using UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync

# Or install directly from pyproject.toml
uv pip install -e .
```

## Usage

Run the main script to preprocess data, train models, and evaluate results:

```bash
python main.py
```

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
