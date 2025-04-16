import os
import random
import sys
import time
import warnings
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add the parent directory to sys.path to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data.featureeng import add_kpi_features
from src.data.filter import filter_data
from src.data.preprocessing import (
    unify_and_drop_part_ids,
    unify_and_drop_process_types,
    unify_and_filter_resources,
)
from src.models.resnet_bilstm_attn.dataset_hypothesistesting import (
    BiLSTMDatasetHypothesis,
)
from src.models.resnet_bilstm_attn.model import (
    BiLSTM,
    collate_fn,
    evaluate_model_with_preds,
    train_model,
)


def load_and_preprocess_data(
    only_real=False, only_sim=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess real and simulated data separately with consistent features.

    Args:
        only_real: If True, only load and process real data
        only_sim: If True, only load and process simulated data

    Returns:
        Tuple of (real_data, sim_data) with identical feature sets, or a single DataFrame if only one type is requested
    """
    # Load data based on parameters
    real_data = None
    sim_data = None

    if not only_sim:
        print("\n----- Loading Real Data -----")
        real_data = pd.read_csv(
            r"D:\resnet-bilstm-attention-dt\datasrc\real\real_factorydata_oclog.csv",
            parse_dates=["start_time", "end_time"],
            index_col="process_execution_id",
        )
        real_data["is_valid"] = 1
        real_data["duration"] = (
            real_data["end_time"] - real_data["start_time"]
        ).dt.total_seconds()
        print(f"Raw real data shape: {real_data.shape}")

    if not only_real:
        print("\n----- Loading Simulated Data -----")
        sim_data = pd.read_csv(
            r"D:\resnet-bilstm-attention-dt\datasrc\sim\simulated_data_oclog.csv",
            parse_dates=["start_time", "end_time"],
            index_col="process_execution_id",
        )
        sim_data["is_valid"] = 0
        sim_data["duration"] = (
            sim_data["end_time"] - sim_data["start_time"]
        ).dt.total_seconds()
        print(f"Raw simulated data shape: {sim_data.shape}")

    # Process each dataset individually
    if real_data is not None:
        print("\n----- Processing Real Data -----")
        real_data = preprocess_dataset(real_data, "real")

    if sim_data is not None:
        print("\n----- Processing Simulated Data -----")
        sim_data = preprocess_dataset(sim_data, "simulated")

    # Align feature sets between datasets
    if real_data is not None and sim_data is not None:
        real_data, sim_data = align_features(real_data, sim_data)

    # Return based on requested data
    if only_real:
        return real_data
    elif only_sim:
        return sim_data
    else:
        return real_data, sim_data


def preprocess_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply all preprocessing steps to a dataset."""
    print(f"Initial {dataset_name} data shape: {df.shape}")

    try:
        # Apply standard preprocessing
        df = unify_and_drop_part_ids(df)
        df = unify_and_drop_process_types(df)
        df = unify_and_filter_resources(df)

        # For simulated data, apply a more lenient filtering or manually add the time features
        if dataset_name == "real":
            # Apply normal filtering for real data
            df = filter_data(df)
        else:
            # For simulated data
            try:
                # Save a copy in case filtering removes everything
                df_before_filter = df.copy()

                # Try using filter_data (which adds time features)
                df = filter_data(df)

                # If filtering removed all rows, manually add time features to the unfiltered data
                if len(df) == 0:
                    print(
                        "Warning: filter_data() removed all simulated data. Adding time features manually."
                    )
                    df = df_before_filter

                    # Manually add the time-based features that filter_data would have added
                    # Convert datetime to hour of day (0-23)
                    df["hour_of_day"] = df["start_time"].dt.hour

                    # Add cyclical encoding of hour of day (sin and cos)
                    df["hour_of_day_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
                    df["hour_of_day_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

                    # Get day of week (0=Monday, 6=Sunday)
                    df["day_of_week"] = df["start_time"].dt.dayofweek

                    # Add cyclical encoding of day of week (sin and cos)
                    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
                    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

                    # Add is_break indicator (time between 11:30-13:30 or after 16:30)
                    hour = df["start_time"].dt.hour
                    minute = df["start_time"].dt.minute
                    time_value = hour + minute / 60
                    df["is_break"] = ((time_value >= 11.5) & (time_value <= 13.5)) | (
                        time_value >= 16.5
                    )

                    # Add is_not_weekday indicator (Saturday or Sunday)
                    df["is_not_weekday"] = df["day_of_week"] >= 5

                    # Add unix timestamps
                    df["start_time_unix"] = df["start_time"].astype(np.int64) // 10**9
                    df["end_time_unix"] = df["end_time"].astype(np.int64) // 10**9

                    # Add sequence_number if it doesn't exist
                    if "sequence_number" not in df.columns:
                        print(
                            "Adding sequence_number to simulated data (matching filter_data logic)"
                        )
                        # Use EXACTLY the same logic as in filter_data()
                        df["sequence_number"] = (
                            df.sort_values(by=["start_time"])
                            .groupby("process_execution_id")
                            .cumcount()
                            + 1
                        )
            except Exception as e:
                print(f"Error during simulated data processing: {e}")
                # If anything goes wrong, make sure we at least add sequence_number
                if "sequence_number" not in df.columns:
                    print("Adding sequence_number to simulated data after error")
                    df = df.sort_values("start_time")
                    df["sequence_number"] = range(1, len(df) + 1)

        # Generate KPI features for both types of data
        df = add_kpi_features(df)

        print(f"Preprocessed {dataset_name} data shape: {df.shape}")
        print(f"Features: {sorted(df.columns.tolist())}")

    except Exception as e:
        print(f"Error preprocessing {dataset_name} data: {e}")
        print(f"Exception details: {str(e)}")
        print("Continuing with partially processed data")

        # Last resort: ensure sequence_number exists
        if "sequence_number" not in df.columns:
            print("Adding sequence_number as last resort")
            df = df.sort_values("start_time")
            df["sequence_number"] = range(1, len(df) + 1)

    return df


def align_features(
    real_df: pd.DataFrame, sim_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure both datasets have identical feature sets."""
    print("\n----- Aligning Features -----")

    # Get all unique columns from both datasets
    all_columns = set(real_df.columns) | set(sim_df.columns)
    print(f"Total unique columns across datasets: {len(all_columns)}")

    # Add missing columns to each dataset
    for col in all_columns:
        # Add to real data if missing
        if col not in real_df.columns:
            print(f"Adding missing column '{col}' to real data")
            if col in [
                "hour_of_day_cos",
                "hour_of_day_sin",
                "day_of_week_cos",
                "day_of_week_sin",
            ]:
                real_df[col] = 0.0  # Default for cyclical features
            elif col in ["is_break", "is_not_weekday"]:
                real_df[col] = False  # Default for boolean features
            elif pd.api.types.is_numeric_dtype(sim_df[col]):
                real_df[col] = 0  # Default for numeric features
            else:
                real_df[col] = sim_df[col].iloc[0]  # Use first value from sim data

        # Add to sim data if missing
        if col not in sim_df.columns:
            print(f"Adding missing column '{col}' to simulated data")
            if col in [
                "hour_of_day_cos",
                "hour_of_day_sin",
                "day_of_week_cos",
                "day_of_week_sin",
            ]:
                sim_df[col] = 0.0  # Default for cyclical features
            elif col in ["is_break", "is_not_weekday"]:
                sim_df[col] = False  # Default for boolean features
            elif pd.api.types.is_numeric_dtype(real_df[col]):
                sim_df[col] = 0  # Default for numeric features
            else:
                sim_df[col] = real_df[col].iloc[0]  # Use first value from real data

    # Sort columns to ensure identical order
    sorted_columns = sorted(all_columns)
    real_df = real_df[sorted_columns]
    sim_df = sim_df[sorted_columns]

    print(f"Aligned real data shape: {real_df.shape}")
    print(f"Aligned simulated data shape: {sim_df.shape}")
    print(f"Features match: {set(real_df.columns) == set(sim_df.columns)}")

    return real_df, sim_df


def perform_identical_real_data_test() -> pd.DataFrame:
    """
    Performs the 'Identical Data Test' using only real data to validate the framework.
    """
    print("\n===== PERFORMING IDENTICAL REAL DATA TEST =====")

    # Load and preprocess only real data
    real_data = load_and_preprocess_data(only_real=True)
    print(f"Original real data shape: {real_data.shape}")

    # Verify we have data
    if len(real_data) == 0:
        raise ValueError("No real data available after preprocessing!")

    # Create a copy of the real data
    real_data_copy = real_data.copy()

    # Change the label of the copy to 0 (as if it were simulated data)
    real_data_copy["is_valid"] = 0

    # Combine the original and the copy
    combined_data = pd.concat([real_data, real_data_copy]).fillna(0)

    # Reset index to avoid ambiguity between index and columns
    combined_data = combined_data.reset_index(drop=False)

    # Sort if needed
    sort_columns = [
        "end_time",
        "sequence_number",
    ]  # Removed order_id to avoid ambiguity
    valid_sort_columns = [col for col in sort_columns if col in combined_data.columns]

    if valid_sort_columns:
        combined_data = combined_data.sort_values(
            by=valid_sort_columns,
            key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
        ).reset_index(drop=True)
    else:
        combined_data = combined_data.reset_index(drop=True)

    print(f"Combined identical real data shape: {combined_data.shape}")
    print(f"is_valid counts: {combined_data['is_valid'].value_counts().to_dict()}")

    return combined_data


def perform_identical_sim_data_test() -> pd.DataFrame:
    """
    Performs the 'Identical Data Test' using only simulated data to validate the framework.
    """
    print("\n===== PERFORMING IDENTICAL SIMULATED DATA TEST =====")

    # Load and preprocess only simulated data
    sim_data = load_and_preprocess_data(only_sim=True)
    print(f"Original simulated data shape: {sim_data.shape}")

    # Verify we have data
    if len(sim_data) == 0:
        raise ValueError("No simulated data available after preprocessing!")

    # Create a copy of the sim data
    sim_data_copy = sim_data.copy()

    # Change the label of the copy to 1 (as if it were real data)
    sim_data_copy["is_valid"] = 1

    # Combine the original and the copy
    combined_data = pd.concat([sim_data, sim_data_copy]).fillna(0)

    # Reset index to avoid ambiguity between index and columns
    combined_data = combined_data.reset_index(drop=False)

    # Sort if needed
    sort_columns = [
        "end_time",
        "sequence_number",
    ]  # Removed order_id to avoid ambiguity

    # Make sure the sort columns exist
    valid_sort_columns = [col for col in sort_columns if col in combined_data.columns]

    if valid_sort_columns:
        combined_data = combined_data.sort_values(
            by=valid_sort_columns,
            key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
        ).reset_index(drop=True)
    else:
        combined_data = combined_data.reset_index(drop=True)

    print(f"Combined identical sim data shape: {combined_data.shape}")
    print(f"is_valid counts: {combined_data['is_valid'].value_counts().to_dict()}")

    return combined_data


# Add global feature cache
feature_cache = {}


def prepare_train_test_data(
    df: pd.DataFrame,
    feature_subset: List[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and testing data with optional feature subset selection.
    Ensures both classes are represented in train and test sets.
    """
    # Create cache key based on features and random state
    cache_key = (frozenset(feature_subset) if feature_subset else None, random_state)

    # Check if we've already processed this combination
    if cache_key in feature_cache:
        return feature_cache[cache_key]

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check class distribution before doing anything
    class_counts = df_copy["is_valid"].value_counts()
    print(f"Original class distribution: {dict(class_counts)}")

    # Check if we have both classes in the data with sufficient samples
    if len(class_counts) < 2:
        raise ValueError("Dataset contains only one class. Cannot split properly.")

    # Ensure minimum samples for stratification - at least 2 samples per class needed
    for cls, count in class_counts.items():
        if count < 2:
            raise ValueError(
                f"Class {cls} has only {count} samples, which is too few for stratification."
            )

    # Extract target variable
    y = df_copy["is_valid"]

    # Remove non-feature columns
    non_feature_cols = ["is_valid"]
    X = df_copy.drop(
        columns=[col for col in non_feature_cols if col in df_copy.columns]
    )

    # Filter features if a subset is specified
    if feature_subset:
        available_features = [col for col in feature_subset if col in X.columns]
        if not available_features:
            raise ValueError(
                f"None of the specified features {feature_subset} are available in the dataset"
            )
        X = X[available_features]

    # Convert datetime columns to numeric features to avoid TypeError
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            # Extract useful numeric features from datetime
            X[f"{col}_timestamp"] = (
                X[col].astype(np.int64) // 10**9
            )  # Unix timestamp in seconds
            X = X.drop(columns=[col])  # Remove the original datetime column

    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Verify that both classes appear in both train and test sets
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())

    if len(train_classes) < 2 or len(test_classes) < 2:
        raise ValueError(
            f"Split resulted in imbalanced classes. Train classes: {train_classes}, Test classes: {test_classes}"
        )

    print(f"Train class distribution: {dict(y_train.value_counts())}")
    print(f"Test class distribution: {dict(y_test.value_counts())}")

    # Cache the result before returning
    feature_cache[cache_key] = (X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    random_state: int = 683,
) -> DecisionTreeClassifier:
    """
    Train a decision tree classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        max_depth: Maximum depth of the decision tree
        random_state: Random seed for reproducibility

    Returns:
        Trained decision tree model
    """
    dt = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    dt.fit(X_train, y_train)
    return dt


def evaluate_classifier(
    model: Union[DecisionTreeClassifier, BiLSTM],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "dt",
) -> Dict[str, float]:
    """
    Evaluate a classifier on test data.
    """
    result_metrics = {}

    # Verify we have at least two classes in test set
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Test set contains only classes: {unique_classes}. Need both classes for proper evaluation."
        )

    if model_type.lower() == "dt":
        y_pred = model.predict(X_test)

        # Get probabilities - handle potential shape issues
        y_proba_all = model.predict_proba(X_test)

        # Make sure we're getting the probability for the positive class
        if y_proba_all.shape[1] >= 2:  # Normal case with both classes
            y_proba = y_proba_all[:, 1]
        else:  # Unusual case with only one class in prediction
            y_proba = y_proba_all[:, 0]

    elif model_type.lower() == "lstm":
        # Create a test dataset and dataloader for BiLSTM using the hypothesis testing dataset class
        test_data = pd.concat([X_test, y_test], axis=1)
        feature_columns = list(X_test.columns)
        try:
            test_dataset = BiLSTMDatasetHypothesis(
                test_data, sequence_length=19, feature_columns=feature_columns
            )
            test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

            # Get predictions
            all_labels, all_preds, all_probs = evaluate_model_with_preds(
                model, test_loader
            )
            y_pred = np.array(all_preds)
            y_proba = np.array(all_probs).clip(0, 1)  # Clip to [0,1] range
            y_test = np.array(all_labels)

            # Check for any NaN values in probabilities
            if np.isnan(y_proba).any():
                print(f"Warning: {np.isnan(y_proba).sum()} NaN values in probabilities")
                y_proba = np.nan_to_num(y_proba, nan=0.5)  # Replace NaNs with 0.5

        except Exception as e:
            print(f"Error in LSTM evaluation: {e}")
            print("Falling back to random predictions")
            # Generate random predictions as fallback
            y_pred = np.random.randint(0, 2, size=len(y_test))
            y_proba = np.random.random(size=len(y_test))

        # Check again after processing through the LSTM
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            raise ValueError(
                f"LSTM predictions resulted in only classes: {unique_classes}. Cannot compute metrics properly."
            )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Calculate metrics
    result_metrics["accuracy"] = accuracy_score(y_test, y_pred)

    # Use try/except for ROC AUC to handle potential errors
    try:
        result_metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        result_metrics["roc_auc"] = 0.5  # Default to random chance

    return result_metrics


def permutation_test(
    model_train_func: Callable,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metric: str = "accuracy",
    n_permutations: int = 1000,
    model_type: str = "dt",
    verbose: bool = True,
    **model_kwargs,
) -> Tuple[float, float, List[float]]:
    """
    Perform permutation test with improved efficiency for BiLSTM models.
    """
    # Ensure both classes are in train and test sets
    if len(set(y_train.unique())) < 2 or len(set(y_test.unique())) < 2:
        raise ValueError("Both training and test sets must contain both classes")

    # Extract model training parameters based on model type
    train_kwargs = {}
    if model_type.lower() == "dt":
        # Only pass parameters that train_decision_tree accepts
        dt_params = ["max_depth", "random_state"]
        train_kwargs = {k: v for k, v in model_kwargs.items() if k in dt_params}
    else:
        # LSTM model can receive all parameters
        train_kwargs = model_kwargs

    # Train model on original data with appropriate parameters
    model = model_train_func(X_train, y_train, **train_kwargs)

    # Generate predictions once
    if model_type.lower() == "dt":
        # Decision tree case remains unchanged
        metrics = evaluate_classifier(model, X_test, y_test, model_type=model_type)
        observed_stat = metrics[metric]

        # For permutations, simply use the trained model with shuffled labels
        permutation_stats = []
        iterator = (
            tqdm(range(n_permutations), desc="Permutation testing")
            if verbose
            else range(n_permutations)
        )

        for _ in iterator:
            y_test_perm = y_test.sample(frac=1.0, replace=False).reset_index(drop=True)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics on permuted data
            if metric == "accuracy":
                perm_stat = accuracy_score(y_test_perm, y_pred)
            else:  # roc_auc
                perm_stat = roc_auc_score(y_test_perm, y_proba)

            permutation_stats.append(perm_stat)
    else:  # BiLSTM model
        # Create dataset and get predictions only once
        test_data = pd.concat([X_test, y_test], axis=1)
        feature_columns = list(X_test.columns)
        test_dataset = BiLSTMDatasetHypothesis(
            test_data, sequence_length=19, feature_columns=feature_columns
        )
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

        # Get predictions once
        all_labels, all_preds, all_probs = evaluate_model_with_preds(model, test_loader)
        y_pred = np.array(all_preds)

        # NEW: Ensure probabilities are properly normalized
        y_proba = np.array(all_probs).clip(0, 1)  # Clip to [0,1] range
        y_test_array = np.array(all_labels)

        # Check for any NaN values in probabilities
        if np.isnan(y_proba).any():
            print(f"Warning: {np.isnan(y_proba).sum()} NaN values in probabilities")
            y_proba = np.nan_to_num(y_proba, nan=0.5)  # Replace NaNs with 0.5

        # Calculate observed statistic
        if metric == "accuracy":
            observed_stat = accuracy_score(y_test_array, y_pred)
        else:  # roc_auc
            try:
                observed_stat = roc_auc_score(y_test_array, y_proba)
            except Exception as e:
                print(f"ROC AUC calculation error: {e}")
                print(f"y_test_array unique values: {np.unique(y_test_array)}")
                print(f"y_proba range: [{y_proba.min()}, {y_proba.max()}]")
                if len(np.unique(y_test_array)) < 2:
                    print("Not enough unique classes, falling back to accuracy")
                    observed_stat = accuracy_score(y_test_array, y_pred)
                else:
                    raise

        # Free up memory
        torch.cuda.empty_cache()

        # Use smaller batch size for evaluation to avoid OOM
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,  # Smaller batch size
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=2,
        )

        # For permutations, use the fixed predictions with shuffled labels
        permutation_stats = []
        iterator = (
            tqdm(range(n_permutations), desc="Permutation testing")
            if verbose
            else range(n_permutations)
        )

        for _ in iterator:
            # Create a permuted version of the test labels (instead of reshuffling through the dataloader)
            y_test_perm = np.random.permutation(y_test_array)

            # Calculate metric on permuted data using the SAME predictions
            if metric == "accuracy":
                perm_stat = accuracy_score(y_test_perm, y_pred)
            else:  # roc_auc
                try:
                    perm_stat = roc_auc_score(y_test_perm, y_proba)
                except Exception as e:
                    print(f"Permutation ROC AUC error: {e}")
                    # Default to 0.5 (random chance) if calculation fails
                    perm_stat = 0.5

            permutation_stats.append(perm_stat)

    # Calculate p-value
    p_value = sum(stat >= observed_stat for stat in permutation_stats) / n_permutations

    # Save null distribution for visualization/analysis
    if verbose and model_type.lower() == "lstm":
        component = model_kwargs.get("component", "unknown_component")
        run_id = model_kwargs.get("run_id", 1)
        save_dir = model_kwargs.get("output_dir", "hypothesis_results")
        os.makedirs(save_dir, exist_ok=True)

        df_null = pd.DataFrame(
            {
                "permutation_stat": permutation_stats,
                "observed_stat": observed_stat,
                "p_value": p_value,
            }
        )
        df_null.to_csv(f"{save_dir}/{component}_run{run_id}_null_dist.csv", index=False)

    return observed_stat, p_value, permutation_stats


def multiple_runs_hypothesis_test(
    df: pd.DataFrame,
    feature_subsets: Dict[str, List[str]],
    n_runs: int = 10,
    n_permutations: int = 1000,
    test_size: float = 0.2,
    alpha: float = 0.01,
    model_type: str = "dt",
    metric: str = "accuracy",
    output_dir: str = "hypothesis_results",
    **model_kwargs,
) -> Dict[str, Dict[str, Union[List[float], float]]]:
    """
    Perform hypothesis testing over multiple runs with different random seeds
    for various feature subsets.

    Args:
        df: Input dataframe
        feature_subsets: Dictionary mapping component names to feature lists
        n_runs: Number of runs with different random seeds
        n_permutations: Number of permutations for each test
        test_size: Proportion of data to use for testing
        alpha: Significance level
        model_type: Type of model ('dt' for decision tree, 'lstm' for BiLSTM)
        metric: Metric to use ('accuracy' or 'roc_auc')
        output_dir: Directory to save results
        **model_kwargs: Additional arguments to pass to the model training function

    Returns:
        Dictionary containing results for each feature subset and run
    """
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_type.lower())
    os.makedirs(model_output_dir, exist_ok=True)

    results = {}

    # Select model training function based on model_type
    if model_type.lower() == "dt":
        model_train_func = train_decision_tree
    elif model_type.lower() == "lstm":
        # Create a function that wraps the BiLSTM training for compatibility
        def train_bilstm(X_train, y_train, **kwargs):
            # Ensure data is properly formatted for BiLSTM
            train_data = pd.concat([X_train, y_train], axis=1)

            # Create dataset and dataloader
            sequence_length = kwargs.get(
                "sequence_length", 19
            )  # Default sequence length
            batch_size = kwargs.get("batch_size", 32)  # Default batch size

            train_dataset = BiLSTMDatasetHypothesis(
                train_data, sequence_length=sequence_length
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

            # Define BiLSTM model - Updated to match notebook's xLSTM architecture
            input_size = X_train.shape[1]
            hidden_size = kwargs.get("hidden_size", 512)  # Increased from 64 to 512
            num_layers = kwargs.get("num_layers", 1)  # Changed from 2 to 1
            attention_heads = kwargs.get("attention_heads", 4)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BiLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                attention_heads=attention_heads,
            ).to(device)

            # Train BiLSTM model
            n_epochs = kwargs.get("n_epochs", 10)  # Default number of epochs
            train_model(
                model,
                train_loader,
                num_epochs=n_epochs,
                learning_rate=1e-3,
                device=device,
            )

            return model

        model_train_func = train_bilstm
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Run hypothesis tests for each feature subset
    for component, features in feature_subsets.items():
        print(f"\nTesting component: {component}")

        # Initialize results for this component
        results[component] = {
            "observed_stats": [],
            "p_values": [],
            "rejections": [],
        }

        # Run multiple tests with different random seeds
        for run in range(n_runs):
            print(f"Run {run+1}/{n_runs}")
            random_seed = random.randint(1, 10000)  # Generate a random seed

            try:
                # Prepare train/test data with the current feature subset
                X_train, X_test, y_train, y_test = prepare_train_test_data(
                    df, features, test_size=test_size, random_state=random_seed
                )

                # Create a complete kwargs dictionary with both model and tracking parameters
                run_kwargs = model_kwargs.copy()  # Start with the model parameters
                run_kwargs.update(
                    {  # Add tracking parameters
                        "component": component,
                        "run_id": run + 1,
                        "output_dir": model_output_dir,
                    }
                )

                # Pass all parameters through a single kwargs dictionary
                observed_stat, p_value, _ = permutation_test(
                    model_train_func,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    metric=metric,
                    n_permutations=n_permutations,
                    model_type=model_type,
                    verbose=True,
                    **run_kwargs,
                )

                # Store results
                results[component]["observed_stats"].append(observed_stat)
                results[component]["p_values"].append(p_value)
                results[component]["rejections"].append(1 if p_value < alpha else 0)

                print(
                    f"  {metric}: {observed_stat:.4f}, p-value: {p_value:.4f}, reject: {p_value < alpha}"
                )

            except Exception as e:
                print(f"Error in run {run+1}: {str(e)}")
                # Add placeholder values to maintain run count
                results[component]["observed_stats"].append(float("nan"))
                results[component]["p_values"].append(float("nan"))
                results[component]["rejections"].append(0)

        # Calculate summary statistics
        valid_observations = [
            stat for stat in results[component]["observed_stats"] if not np.isnan(stat)
        ]
        valid_p_values = [p for p in results[component]["p_values"] if not np.isnan(p)]
        valid_rejections = [
            r
            for i, r in enumerate(results[component]["rejections"])
            if not np.isnan(results[component]["observed_stats"][i])
        ]

        # Only calculate stats if we have valid results
        if valid_observations:
            results[component]["mean_observed_stat"] = np.mean(valid_observations)
            results[component]["std_observed_stat"] = np.std(valid_observations)
            results[component]["mean_p_value"] = np.mean(valid_p_values)
            results[component]["rejection_rate"] = (
                np.mean(valid_rejections) if valid_rejections else 0
            )

            print(f"Summary for {component}:")
            print(f"  Mean {metric}: {results[component]['mean_observed_stat']:.4f}")
            print(f"  Mean p-value: {results[component]['mean_p_value']:.4f}")
            print(f"  Rejection rate: {results[component]['rejection_rate']:.2f}")
        else:
            results[component]["mean_observed_stat"] = float("nan")
            results[component]["std_observed_stat"] = float("nan")
            results[component]["mean_p_value"] = float("nan")
            results[component]["rejection_rate"] = 0
            print(f"No valid results for {component}")

    # Save results to file using the model-specific directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(model_output_dir, f"hypothesis_results_{timestamp}.txt")

    with open(result_file, "w") as f:
        f.write("Hypothesis Testing Results\n")
        f.write("========================\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Runs: {n_runs}\n")
        f.write(f"Permutations: {n_permutations}\n")
        f.write(f"Alpha: {alpha}\n\n")

        for component, res in results.items():
            f.write(f"Component: {component}\n")
            f.write(
                f"  Mean {metric}: {res.get('mean_observed_stat', float('nan')):.4f}\n"
            )
            f.write(
                f"  Std {metric}: {res.get('std_observed_stat', float('nan')):.4f}\n"
            )
            f.write(f"  Mean p-value: {res.get('mean_p_value', float('nan')):.4f}\n")
            f.write(f"  Rejection rate: {res.get('rejection_rate', 0):.2f}\n")
            f.write("\n")

    print(f"\nResults saved to: {result_file}")

    # Generate plots for each component
    print("\nGenerating plots...")
    for component in feature_subsets.keys():
        try:
            generate_hypothesis_test_plots(model_output_dir, component)
        except Exception as e:
            print(f"Error generating plots for {component}: {e}")

    # Generate summary plots
    try:
        generate_summary_plots(results, model_output_dir, metric, model_type)
    except Exception as e:
        print(f"Error generating summary plots: {e}")

    return results


def define_feature_subsets() -> Dict[str, List[str]]:
    """
    Define feature subsets for different SBDT components.

    Returns:
        Dictionary mapping component names to feature lists
    """
    # TODO: Create feature subsets for the SBDT components
    return {
        "time_model": [
            "duration",
            "sequence_number",
            "hour_of_day_cos",
            "hour_of_day_sin",
            "day_of_week_cos",
            "day_of_week_sin",
            "is_break",
            "is_not_weekday",
        ],
        "resource_model": [
            "resource_id",
            "part_id",
            "process_id",
        ],
        "transformation_model": [
            "part_id",
            "process_id",
            "sequence_number",
        ],
        "transition_model": [
            "part_id",
            "resource_id",
            "sequence_number",
            "duration",
        ],
        "process_model": [
            "process_id",
            "duration",
            "sequence_number",
        ],
        "kpi_based": [
            "throughput",
            "cycle_time_sec",
            "lead_time_sec",
            "setup_time_sec",
        ],
        "all_features": [],  # Will be populated with all available features
    }


def generate_hypothesis_test_plots(
    results_dir: str, component_name: str, run_id: int = None
):
    """
    Generate plots for hypothesis testing results.

    Args:
        results_dir: Directory containing the null distribution CSV files
        component_name: Name of the SBDT component to plot
        run_id: If provided, plot only this specific run; otherwise plot all runs
    """
    # Set up plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    if run_id is not None:
        # Plot specific run
        run_files = [f"{component_name}_run{run_id}_null_dist.csv"]
    else:
        # Find all files for this component
        import glob

        run_files = glob.glob(f"{results_dir}/{component_name}_run*_null_dist.csv")
        run_files = [os.path.basename(f) for f in run_files]

    for file_name in run_files:
        try:
            # Load the null distribution
            file_path = os.path.join(results_dir, file_name)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            null_dist_df = pd.read_csv(file_path)

            if "observed_stat" in null_dist_df.columns:
                # New format with observed_stat column
                observed_stat = null_dist_df["observed_stat"].iloc[0]
                p_value = (
                    null_dist_df["p_value"].iloc[0]
                    if "p_value" in null_dist_df.columns
                    else None
                )
                null_dist = null_dist_df["permutation_stat"].values
            else:
                # Old format with just permutation_stat
                null_dist = null_dist_df["permutation_stat"].values
                # Calculate observed_stat and p_value (if needed)
                observed_stat = null_dist.mean() + null_dist.std()  # Placeholder
                p_value = None

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot histogram of null distribution
            sns.histplot(
                null_dist,
                kde=True,
                ax=ax,
                color="skyblue",
                stat="density",
                label="Null Distribution",
            )

            # Add vertical line for observed statistic
            ax.axvline(
                x=observed_stat,
                color="red",
                linestyle="--",
                label=f"Observed Statistic: {observed_stat:.4f}",
            )

            # Add title and labels
            current_run = file_name.split("_run")[1].split("_")[0]
            ax.set_title(f"Permutation Test for {component_name} (Run {current_run})")
            ax.set_xlabel("Test Statistic")
            ax.set_ylabel("Density")

            # Add legend
            legend_text = f"Observed: {observed_stat:.4f}"
            if p_value is not None:
                legend_text += f", p-value: {p_value:.4f}"
            ax.text(
                0.02,
                0.95,
                legend_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.legend()
            plt.tight_layout()

            # Save the plot
            output_file = os.path.join(
                results_dir, f"{component_name}_run{current_run}_permutation_test.png"
            )
            plt.savefig(output_file, dpi=300)
            print(f"Plot saved to: {output_file}")
            plt.close()

        except Exception as e:
            print(f"Error generating plot for {file_name}: {e}")


def generate_summary_plots(
    results: Dict[str, Dict[str, Union[List[float], float]]],
    output_dir: str,
    metric: str,
    model_type: str,
):
    """
    Generate summary plots comparing all SBDT components.

    Args:
        results: Results dictionary from multiple_runs_hypothesis_test
        output_dir: Directory to save plots
        metric: Metric used (accuracy or roc_auc)
        model_type: Type of model (dt or lstm)
    """
    # Set up plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    # Extract data for plotting
    components = []
    means = []
    stds = []
    p_values = []
    rejection_rates = []

    for component, res in results.items():
        components.append(component)
        means.append(res.get("mean_observed_stat", float("nan")))
        stds.append(res.get("std_observed_stat", float("nan")))
        p_values.append(res.get("mean_p_value", float("nan")))
        rejection_rates.append(res.get("rejection_rate", 0))

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "Component": components,
            f"Mean {metric}": means,
            f"Std {metric}": stds,
            "Mean p-value": p_values,
            "Rejection Rate": rejection_rates,
        }
    )

    # 1. Plot Mean Metric Values with Error Bars
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        df["Component"],
        df[f"Mean {metric}"],
        yerr=df[f"Std {metric}"],
        capsize=10,
        color="skyblue",
        alpha=0.8,
    )

    # Add a horizontal line at 0.5 (random chance for classification)
    ax.axhline(
        y=0.5, color="red", linestyle="--", alpha=0.7, label="Random Chance (0.5)"
    )

    # Add text showing the exact values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    ax.set_title(f"Mean {metric.upper()} by SBDT Component ({model_type.upper()})")
    ax.set_ylabel(f"{metric.upper()}")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{model_type}_{metric}_by_component.png"), dpi=300
    )
    plt.close()

    # 2. Plot Rejection Rates
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["Component"], df["Rejection Rate"], color="salmon", alpha=0.8)

    # Add text showing the exact values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    ax.set_title(f"Rejection Rate by SBDT Component ({model_type.upper()})")
    ax.set_ylabel("Rejection Rate")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_rejection_rates.png"), dpi=300)
    plt.close()

    # 3. Plot p-values
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["Component"], df["Mean p-value"], color="lightgreen", alpha=0.8)

    # Add a horizontal line at 0.05 (common significance level)
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="α = 0.05")

    # Add text showing the exact values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    ax.set_title(f"Mean p-value by SBDT Component ({model_type.upper()})")
    ax.set_ylabel("Mean p-value")
    ax.set_ylim(0, max(1.0, df["Mean p-value"].max() + 0.1))
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_p_values.png"), dpi=300)
    plt.close()

    print(f"Summary plots saved to: {output_dir}")


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    real_data, sim_data = load_and_preprocess_data()
    final_data = pd.concat([real_data, sim_data]).fillna(0)
    print(f"Dataset shape: {final_data.shape}")

    # Define feature subsets
    feature_subsets = define_feature_subsets()

    # Populate "all_features" with all available features
    non_feature_cols = [
        "is_valid",
    ]
    all_features = [col for col in final_data.columns if col not in non_feature_cols]
    feature_subsets["all_features"] = all_features

    print(f"Number of feature subsets: {len(feature_subsets)}")
    for component, features in feature_subsets.items():
        print(f"  {component}: {len(features)} features")

    # Set the evaluation metric
    metric = "roc_auc"  # Using roc_auc as the metric instead of accuracy
    model_type = "lstm"  # Change to "lstm" if you want to test with BiLSTM

    """For LSTM: Alpha is set to 0.01, and the number of runs is set to 10., Permutations is 200
    for DT: Alpha is set to 0.05, and the number of runs is set to 10., Permutations is 1000
    """
    # Run the hypothesis tests
    results = multiple_runs_hypothesis_test(
        final_data,
        feature_subsets,
        n_runs=10,  # Number of runs with different random seeds
        n_permutations=200,  # Number of permutations for each test
        test_size=0.2,
        alpha=0.01,
        model_type=model_type,  # dt or lstm
        metric=metric,  # Pass the defined metric
        output_dir="hypothesis_results",
        max_depth=5,  # Additional parameter for the decision tree
    )

    # Print final conclusions
    print("\nFinal conclusions:")
    for component, res in results.items():
        conclusion = "INACCURATE" if res["rejection_rate"] > 0.9 else "ACCURATE"
        print(
            f"SBDT Component '{component}': {conclusion} (Rejection Rate: {res['rejection_rate']:.2f}, Mean {metric.upper()}: {res['mean_observed_stat']:.4f})"
        )

    # Generate plots for existing results (optional - add this if you want to generate plots for already saved data)  # Change as needed to dt or lstm
    results_dir = f"hypothesis_results/{model_type}"

    if os.path.exists(results_dir):
        print("\nGenerating plots for existing results...")
        for component in feature_subsets.keys():
            try:
                generate_hypothesis_test_plots(results_dir, component)
            except Exception as e:
                print(f"Error generating plots for {component}: {e}")
