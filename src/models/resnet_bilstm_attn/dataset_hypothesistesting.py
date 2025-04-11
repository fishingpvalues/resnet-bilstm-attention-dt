from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

        # Ensure numeric data types for all features
        for col in self.data.columns:
            if col != "is_valid":  # Keep target column as is
                try:
                    # Try to convert to numeric, coerce errors to NaN
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to numeric: {e}")

        # Fill NaN values with 0 to prevent errors
        self.data = self.data.fillna(0)

        # Create samples by sliding a window of sequence_length over the data
        # Ensure we don't exceed data boundaries
        max_start_idx = max(0, len(self.data) - sequence_length)
        self.samples = [
            self.data.iloc[i : i + sequence_length]
            for i in range(min(len(self.data), max_start_idx + 1))
        ]

        # Filter out samples that are too short
        self.samples = [
            sample for sample in self.samples if len(sample) == sequence_length
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
                if not self.feature_columns:  # If all requested features are missing
                    raise ValueError(
                        f"None of the requested features {feature_columns} are in the dataset"
                    )
        else:
            self.feature_columns = [
                col for col in self.data.columns if col != self.target_column
            ]

        if not self.feature_columns:
            raise ValueError("No valid feature columns found in data")

        # Log feature information
        print(f"Using features ({len(self.feature_columns)}): {self.feature_columns}")

        # Verify all features have valid numeric data
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                print(f"Warning: Feature '{col}' is not numeric, attempting to convert")
                try:
                    self.data[col] = self.data[col].astype(float)
                except:
                    print(
                        f"Error: Could not convert '{col}' to numeric type, dropping feature"
                    )
                    self.feature_columns.remove(col)

        # Final check after potential feature removal
        if not self.feature_columns:
            raise ValueError("No valid numeric features remain after type checking")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_df = self.samples[idx]

        # Extract features based on the specified feature columns
        try:
            # Additional safety check for feature presence
            available_features = [
                col for col in self.feature_columns if col in sample_df.columns
            ]

            if not available_features:
                raise ValueError(f"No features available in sample {idx}")

            features = (
                sample_df[available_features].astype(float).values.astype(np.float32)
            )

            # Get the target (same for all rows in this sample)
            target = int(sample_df[self.target_column].values[0])

            return torch.tensor(features, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32
            )
        except Exception as e:
            # Provide detailed error information for debugging
            print(f"Error in __getitem__ for idx={idx}: {e}")
            print(f"Sample columns: {sample_df.columns.tolist()}")
            print(f"Feature columns: {self.feature_columns}")
            raise
