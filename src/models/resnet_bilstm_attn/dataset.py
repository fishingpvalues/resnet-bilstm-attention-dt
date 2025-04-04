from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# --- Dataset Definition ---
class BiLSTMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int):
        self.sequence_length = sequence_length
        self.data = df.copy()
        self.samples = [
            self.data.iloc[i : i + sequence_length]
            for i in range(len(self.data) - sequence_length + 1)
        ]
        print("Initialized BiLSTMDataset with samples:", len(self.samples))

        # Define feature and target columns (ensure these exist in your DataFrame)
        self.feature_columns = [
            "duration",
            "part_id",
            "process_type",
            "process_id",
            "resource_id",
            "sequence_number",
            "day_of_week_sin",
            "day_of_week_cos",
            "hour_of_day_sin",
            "hour_of_day_cos",
            "is_not_weekday",
            "is_break",  # TODO: Add KPI Features to df and convert to right dtype. Right now I get error
        ]
        self.target_column = "is_valid"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_df = self.samples[idx]
        features = (
            sample_df[self.feature_columns].astype(float).values.astype(np.float32)
        )
        target = int(sample_df[self.target_column].values[0])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.long
        )
