import logging
import os
import sys

# Add the src folder to Python path so that modules under 'data' and 'models' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import pandas as pd
import torch

# Imports for the BiLSTM pipeline.
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.featureeng import add_kpi_features
from data.filter import filter_data
from data.preprocessing import (
    unify_and_drop_part_ids,
    unify_and_drop_process_types,
    unify_and_filter_resources,
)

# Import the baseline pipeline function.
from src.models.decisiontree.baseline import run_decision_tree_pipeline
from src.models.resnet_bilstm_attn.dataset import BiLSTMDataset
from src.models.resnet_bilstm_attn.model import (
    BiLSTM,
    collate_fn,
    diagnose_model,
    evaluate_model,
    evaluate_model_with_preds,
    train_model,
)
from src.utils.reporting import generate_report


def run_baseline_pipeline(final_data):
    logging.info("Running Baseline (Decision Tree) Pipeline...")
    # run_decision_tree_pipeline is expected to internally do the train/test split.
    run_decision_tree_pipeline(final_data)
    logging.info("Baseline model finished.")


def run_bilstm_pipeline(final_data):
    logging.info("Running BiLSTM Pipeline...")
    # Create features X and target y (adjust feature columns as required)
    X = final_data.drop(columns=["is_valid", "start_time", "end_time"], errors="ignore")
    y = final_data["is_valid"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df_train_mod = pd.concat([X_train, y_train], axis=1)
    df_test_mod = pd.concat([X_test, y_test], axis=1)

    train_dataset = BiLSTMDataset(df_train_mod, sequence_length=19)
    test_dataset = BiLSTMDataset(df_test_mod, sequence_length=19)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    logging.info("CUDA IS AVAILABLE? %s", torch.cuda.is_available())
    logging.info("Train dataset size: %d", len(train_dataset))
    logging.info("df_train_mod shape: %s", df_train_mod.shape)

    model = BiLSTM(input_size=12, hidden_size=512, num_layers=1, attention_heads=4)
    loss_history = train_model(model, train_loader, num_epochs=10)
    metrics = evaluate_model(model, test_loader)
    logging.info("Test Accuracy: %s", metrics["accuracy"])
    logging.info("Test ROC AUC: %s", metrics["roc_auc"])

    diagnose_model(loss_history, metrics["accuracy"])
    all_labels, all_preds, all_probs = evaluate_model_with_preds(model, test_loader)
    generate_report(all_labels, all_preds, all_probs)

    torch.save(model.state_dict(), "BiLSTM_model.pth")
    logging.info("BiLSTM model saved.")


def main():
    try:
        real_data = pd.read_csv(
            r"D:\resnet-bilstm-attention-dt\datasrc\real\real_factorydata_oclog.csv",
            parse_dates=["start_time", "end_time"],
            index_col="process_execution_id",
        )
        sim_data = pd.read_csv(
            r"D:\resnet-bilstm-attention-dt\datasrc\sim\simulated_data_oclog.csv",
            parse_dates=["start_time", "end_time"],
            index_col="process_execution_id",
        )
        real_data["is_valid"] = 1
        real_data["duration"] = (
            real_data["end_time"] - real_data["start_time"]
        ).dt.total_seconds()
        logging.info("Real data shape: %s", real_data.shape)
        sim_data["is_valid"] = 0
        sim_data["duration"] = (
            sim_data["end_time"] - sim_data["start_time"]
        ).dt.total_seconds()
        logging.info("Simulated data shape: %s", sim_data.shape)
        real_data = unify_and_drop_part_ids(real_data)
        real_data = unify_and_drop_process_types(real_data)
        real_data = unify_and_filter_resources(real_data)
        real_data = filter_data(real_data)
        final_data = pd.concat([real_data, sim_data]).fillna(0)

        # Ensure time columns are datetime
        final_data["start_time"] = pd.to_datetime(
            final_data["start_time"], utc=True, errors="coerce"
        )
        final_data["end_time"] = pd.to_datetime(
            final_data["end_time"], utc=True, errors="coerce"
        )

        final_data = (
            add_kpi_features(final_data)
            .sort_values(
                by=["end_time", "order_id", "sequence_number"],
                key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
            )
            .reset_index(drop=True)
        )
        logging.info(
            "is_valid counts: %s", final_data["is_valid"].value_counts().to_dict()
        )
        # Run each pipeline separately.
        run_baseline_pipeline(final_data)
        run_bilstm_pipeline(final_data)
    except Exception as e:
        logging.error("An error occurred: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
