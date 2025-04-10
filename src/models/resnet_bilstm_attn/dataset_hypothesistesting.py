import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Tuple


class BiLSTMDatasetHypothesis(Dataset):
    def __init__(
        self, df: pd.DataFrame, sequence_length: int, feature_columns: List[str] = None
    ):
        """
        Dataset for the BiLSTM model used in hypothesis testing.

        Args:
            df: DataFrame containing the data
            sequence_length: Length of sequences to create
            feature_columns: List of feature columns to use (if None, all columns except target will be used)
        """
        self.sequence_length = sequence_length
        self.data = df.copy()
        # Create samples by sliding a window of sequence_length over the data
        self.samples = [
            self.data.iloc[i : i + sequence_length]
            for i in range(len(self.data) - sequence_length + 1)
        ]
        print(f"Initialized BiLSTMDatasetHypothesis with {len(self.samples)} samples")

        self.target_column = "is_valid"

        # Verify target exists
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

        # Set feature columns based on input or use all available columns except target
        if feature_columns is not None:
            self.feature_columns = [
                col for col in feature_columns if col in self.data.columns
            ]
            if len(self.feature_columns) < len(feature_columns):
                missing = set(feature_columns) - set(self.feature_columns)
                print(f"Warning: Missing requested features: {missing}")
        else:
            self.feature_columns = [
                col for col in self.data.columns if col != self.target_column
            ]

        if not self.feature_columns:
            raise ValueError("No valid feature columns found in data")

        print(f"Using features: {self.feature_columns}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_df = self.samples[idx]

        # Extract features based on the specified feature columns
        features = (
            sample_df[self.feature_columns].astype(float).values.astype(np.float32)
        )

        # Get the target (same for all rows in this sample)
        target = int(sample_df[self.target_column].values[0])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )
