import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import torch
from torch.utils.data import DataLoader
import os
import sys
import time
from tqdm import tqdm
import random

# Add the parent directory to sys.path to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.resnet_bilstm_attn.model import (
    BiLSTM,
    collate_fn,
    train_model,
    evaluate_model,
    evaluate_model_with_preds,
)
from src.models.resnet_bilstm_attn.dataset_hypothesistesting import (
    BiLSTMDatasetHypothesis,
)
from src.data.preprocessing import (
    unify_and_drop_part_ids,
    unify_and_drop_process_types,
    unify_and_filter_resources,
)
from src.data.filter import filter_data
from src.data.featureeng import add_kpi_features


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load and preprocess both real and simulated data.

    Returns:
        pd.DataFrame: Combined preprocessed dataset
    """
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

    # Set target variable and calculate duration for real data
    real_data["is_valid"] = 1
    real_data["duration"] = (
        real_data["end_time"] - real_data["start_time"]
    ).dt.total_seconds()
    print(f"Real data shape: {real_data.shape}")

    # Set target variable and calculate duration for simulated data
    sim_data["is_valid"] = 0
    sim_data["duration"] = (
        sim_data["end_time"] - sim_data["start_time"]
    ).dt.total_seconds()
    print(f"Simulated data shape: {sim_data.shape}")

    # Apply preprocessing functions only to real data
    real_data = unify_and_drop_part_ids(real_data)
    real_data = unify_and_drop_process_types(real_data)
    real_data = unify_and_filter_resources(real_data)
    real_data = filter_data(real_data)

    # Combine datasets
    final_data = pd.concat([real_data, sim_data]).fillna(0)

    # Ensure time columns are datetime
    final_data["start_time"] = pd.to_datetime(
        final_data["start_time"], utc=True, errors="coerce"
    )
    final_data["end_time"] = pd.to_datetime(
        final_data["end_time"], utc=True, errors="coerce"
    )

    # Apply KPI features and sort
    final_data = (
        add_kpi_features(final_data)
        .sort_values(
            by=["end_time", "order_id", "sequence_number"],
            key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
        )
        .reset_index(drop=True)
    )
    print(f"is_valid counts: {final_data['is_valid'].value_counts().to_dict()}")

    return final_data


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
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check class distribution before doing anything
    class_counts = df_copy["is_valid"].value_counts()
    print(f"Original class distribution: {dict(class_counts)}")

    # Check if we have both classes in the data with sufficient samples
    if len(class_counts) < 2:
        raise ValueError(f"Dataset contains only one class. Cannot split properly.")

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
        test_dataset = BiLSTMDatasetHypothesis(
            test_data, sequence_length=19, feature_columns=feature_columns
        )
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

        # Get predictions
        all_labels, all_preds, all_probs = evaluate_model_with_preds(model, test_loader)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)
        y_test = np.array(all_labels)

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
    result_metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

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
    Perform permutation test to determine if a classifier can distinguish between
    real and simulated data with statistical significance.
    """
    # Ensure both classes are in train and test sets
    if len(set(y_train.unique())) < 2:
        raise ValueError(
            f"Training set doesn't contain both classes. Cannot train model properly."
        )

    if len(set(y_test.unique())) < 2:
        raise ValueError(
            f"Test set doesn't contain both classes. Cannot evaluate properly."
        )

    # Train model on original data
    model = model_train_func(X_train, y_train, **model_kwargs)

    # Evaluate on original data
    try:
        metrics = evaluate_classifier(model, X_test, y_test, model_type=model_type)
        observed_stat = metrics[metric]
    except Exception as e:
        raise ValueError(f"Error evaluating model: {str(e)}")

    # Permutation test
    permutation_stats = []

    # Set up progress bar if verbose
    iterator = (
        tqdm(range(n_permutations), desc="Permutation testing")
        if verbose
        else range(n_permutations)
    )

    for _ in iterator:
        # Create a permuted version of the test labels
        y_test_perm = y_test.sample(frac=1.0, replace=False).reset_index(drop=True)

        # Evaluate on permuted data
        if model_type.lower() == "dt":
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:  # LSTM model
            test_data = pd.concat([X_test, y_test_perm], axis=1)
            test_dataset = BiLSTMDatasetHypothesis(test_data, sequence_length=19)
            test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

            # Get predictions
            all_labels, all_preds, all_probs = evaluate_model_with_preds(
                model, test_loader
            )
            y_pred = np.array(all_preds)
            y_proba = np.array(all_probs)
            y_test_array = np.array(all_labels)

        # Calculate metric on permuted data
        if metric == "accuracy":
            perm_stat = accuracy_score(y_test_perm, y_pred)
        else:  # roc_auc
            perm_stat = roc_auc_score(y_test_perm, y_proba)

        permutation_stats.append(perm_stat)

    # Calculate p-value
    p_value = sum(stat >= observed_stat for stat in permutation_stats) / n_permutations

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

                # Run permutation test
                observed_stat, p_value, _ = permutation_test(
                    model_train_func,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    metric=metric,
                    n_permutations=n_permutations,
                    model_type=model_type,
                    **model_kwargs,
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
        f.write(f"Hypothesis Testing Results\n")
        f.write(f"========================\n")
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


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    final_data = load_and_preprocess_data()
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

    # Run the hypothesis tests
    results = multiple_runs_hypothesis_test(
        final_data,
        feature_subsets,
        n_runs=10,  # Number of runs with different random seeds
        n_permutations=1000,  # Number of permutations for each test
        test_size=0.2,
        alpha=0.05,
        model_type="lstm",  # dt or lstm
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
