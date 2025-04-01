import os
import sys

# Add the src folder to Python path so that modules under 'data' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd

from data.featureeng import add_kpi_features
from data.filter import filter_data
from data.preprocessing import (
    unify_and_drop_part_ids,
    unify_and_drop_process_types,
    unify_and_filter_resources,
)
from models.resnet_bilstm_attn.model import *

if __name__ == "__main__":
    real_data = pd.read_csv(
        r"C:\resnet-bilstm-attention-dt\datasrc\real\real_factorydata_oclog.csv",
        parse_dates=["start_time", "end_time"],
        index_col="process_execution_id",
    )
    sim_data = pd.read_csv(
        r"C:\resnet-bilstm-attention-dt\datasrc\sim\simulated_data_oclog.csv",
        parse_dates=["start_time", "end_time"],
        index_col="process_execution_id",
    )

    sim_data["is_valid"] = 0
    sim_data["duration"] = (
        sim_data["end_time"] - sim_data["start_time"]
    ).dt.total_seconds()

    real_data = unify_and_drop_part_ids(real_data)
    real_data = unify_and_drop_process_types(real_data)
    real_data = unify_and_filter_resources(real_data)

    final_data = filter_data(pd.concat([real_data, sim_data]).fillna(0))
    final_data = (
        add_kpi_features(final_data)
        .sort_values(
            by=["end_time", "order_id", "sequence_number"],
            key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
        )
        .reset_index(drop=True)
    )
