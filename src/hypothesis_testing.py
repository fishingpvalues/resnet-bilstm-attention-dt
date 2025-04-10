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
    BiLSTMDataset,
    collate_fn,
    train_model,
    evaluate_model,
    evaluate_model_with_preds,
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

    # Set the target variable: 0 for simulated, 1 for real data
    sim_data["is_valid"] = 0
    real_data["is_valid"] = 1

    # Calculate duration for simulated data
    sim_data["duration"] = (
        sim_data["end_time"] - sim_data["start_time"]
    ).dt.total_seconds()

    # Apply preprocessing functions to real data
    real_data = unify_and_drop_part_ids(real_data)
    real_data = unify_and_drop_process_types(real_data)
    real_data = unify_and_filter_resources(real_data)

    # Combine datasets and apply additional preprocessing
    final_data = filter_data(pd.concat([real_data, sim_data]).fillna(0))
    final_data = (
        add_kpi_features(final_data)
        .sort_values(
            by=["end_time", "order_id", "sequence_number"],
            key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
        )
        .reset_index(drop=True)
    )

    return final_data


def prepare_train_test_data(
    df: pd.DataFrame,
    feature_subset: List[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and testing data with optional feature subset selection.

    Args:
        df: Input dataframe
        feature_subset: Optional list of feature names to use
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing X_train, X_test, y_train, y_test
    """
    # Drop non-feature columns
    non_feature_cols = [
        "is_valid",
        "start_time",
        "end_time",
        "order_id",
        "start_time_unix",
        "end_time_unix",
    ]

    X = df.drop(columns=[col for col in non_feature_cols if col in df.columns])

    # Filter features if a subset is specified
    if feature_subset:
        available_features = [col for col in feature_subset if col in X.columns]
        if not available_features:
            raise ValueError(
                f"None of the specified features {feature_subset} are available in the dataset"
            )
        X = X[available_features]

    y = df["is_valid"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

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

    Args:
        model: Trained model (either DecisionTreeClassifier or BiLSTM)
        X_test: Test features
        y_test: Test labels
        model_type: Type of model ('dt' for decision tree, 'lstm' for BiLSTM)

    Returns:
        Dictionary containing accuracy and AUC metrics
    """
    if model_type.lower() == "dt":
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type.lower() == "lstm":
        # Create a test dataset and dataloader for BiLSTM
        test_dataset = BiLSTMDataset(
            pd.concat([X_test, y_test], axis=1), sequence_length=19
        )
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

        # Get predictions
        all_labels, all_preds, all_probs = evaluate_model_with_preds(model, test_loader)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)
        y_test = np.array(all_labels)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return {"accuracy": accuracy, "roc_auc": auc}


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

    Args:
        model_train_func: Function to train the model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        metric: Metric to use ('accuracy' or 'roc_auc')
        n_permutations: Number of permutations
        model_type: Type of model ('dt' for decision tree, 'lstm' for BiLSTM)
        verbose: Whether to display progress bar
        **model_kwargs: Additional arguments to pass to the model training function

    Returns:
        Tuple containing observed statistic, p-value, and null distribution
    """
    # Train model on original data
    model = model_train_func(X_train, y_train, **model_kwargs)

    # Evaluate on original data
    metrics = evaluate_classifier(model, X_test, y_test, model_type=model_type)
    observed_stat = metrics[metric]

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
            test_dataset = BiLSTMDataset(
                pd.concat([X_test, y_test_perm], axis=1), sequence_length=19
            )
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
    alpha: float = 0.05,
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Select model training function based on model_type
    if model_type.lower() == "dt":
        model_train_func = train_decision_tree
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    for component, features in feature_subsets.items():
        print(f"\nTesting component: {component} with {len(features)} features")

        component_results = {
            "observed_stats": [],
            "p_values": [],
            "reject_h0": [],
            "feature_count": len(features),
            "features": features,
        }

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            random_state = 42 + run

            # Prepare data with the current feature subset
            X_train, X_test, y_train, y_test = prepare_train_test_data(
                df,
                feature_subset=features,
                test_size=test_size,
                random_state=random_state,
            )

            # Perform permutation test
            observed_stat, p_value, null_dist = permutation_test(
                model_train_func,
                X_train,
                X_test,
                y_train,
                y_test,
                metric=metric,
                n_permutations=n_permutations,
                model_type=model_type,
                verbose=True,
                random_state=random_state,
                **model_kwargs,
            )

            component_results["observed_stats"].append(observed_stat)
            component_results["p_values"].append(p_value)
            component_results["reject_h0"].append(p_value < alpha)

            # Save the null distribution for this run
            null_dist_df = pd.DataFrame({"permutation_stat": null_dist})
            null_dist_df.to_csv(
                f"{output_dir}/{component}_run{run+1}_null_dist.csv", index=False
            )

            # Plot histogram of null distribution with observed statistic
            plt.figure(figsize=(10, 6))
            plt.hist(
                null_dist,
                bins=30,
                alpha=0.7,
                label=f"Null Distribution (p={p_value:.4f})",
            )
            plt.axvline(
                x=observed_stat,
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Observed {metric}={observed_stat:.4f}",
            )
            plt.title(f"Component: {component}, Run {run+1}, Permutation Test")
            plt.xlabel(f"{metric.upper()}")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{component}_run{run+1}_permutation_test.png")
            plt.close()

        # Calculate aggregate statistics
        component_results["mean_observed_stat"] = np.mean(
            component_results["observed_stats"]
        )
        component_results["std_observed_stat"] = np.std(
            component_results["observed_stats"]
        )
        component_results["mean_p_value"] = np.mean(component_results["p_values"])
        component_results["rejection_rate"] = (
            sum(component_results["reject_h0"]) / n_runs
        )

        results[component] = component_results

        # Save component results
        pd.DataFrame(
            {
                "run": list(range(1, n_runs + 1)),
                "observed_stat": component_results["observed_stats"],
                "p_value": component_results["p_values"],
                "reject_h0": component_results["reject_h0"],
            }
        ).to_csv(f"{output_dir}/{component}_results.csv", index=False)

    # Generate summary report
    summary = []
    for component, res in results.items():
        summary.append(
            {
                "Component": component,
                "Feature_Count": res["feature_count"],
                "Mean_Observed_Stat": res["mean_observed_stat"],
                "Std_Observed_Stat": res["std_observed_stat"],
                "Mean_P_Value": res["mean_p_value"],
                "H0_Rejection_Rate": res["rejection_rate"],
                "Conclusion": (
                    "SBDT Component Inaccurate"
                    if res["rejection_rate"] > 0.5
                    else "SBDT Component Accurate"
                ),
            }
        )

    pd.DataFrame(summary).to_csv(f"{output_dir}/summary_results.csv", index=False)

    # Create a summary plot
    plt.figure(figsize=(12, 8))
    components = [s["Component"] for s in summary]
    rejection_rates = [s["H0_Rejection_Rate"] for s in summary]
    mean_stats = [s["Mean_Observed_Stat"] for s in summary]

    x = np.arange(len(components))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 8))

    bars1 = ax1.bar(
        x - width / 2,
        rejection_rates,
        width,
        label="H0 Rejection Rate",
        color="darkblue",
    )
    ax1.set_ylabel("H0 Rejection Rate", fontsize=12)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2, mean_stats, width, label=f"Mean {metric.upper()}", color="orange"
    )
    ax2.set_ylabel(f"Mean {metric.upper()}", fontsize=12)
    ax2.set_ylim(0.4, 1.05)  # Adjusted for metric values

    ax1.set_xlabel("SBDT Component", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, rotation=45, ha="right")

    # Add horizontal line for alpha level
    ax1.axhline(y=0.5, color="red", linestyle="--", label="Decision Threshold (0.5)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Hypothesis Test Results by SBDT Component", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_plot.png")
    plt.close()

    return results


def define_feature_subsets() -> Dict[str, List[str]]:
    """
    Define feature subsets for different SBDT components.

    Returns:
        Dictionary mapping component names to feature lists
    """
    # TODO: Create feature subsets for the SBDT components
    return {
        "process_timing": [
            "duration",
            "time_since_last_process",
            "hour_of_day",
            "day_of_week",
            "process_count_in_order",
        ],
        "resource_allocation": [
            "resource_id",
            "resource_utilization",
            "resource_efficiency",
            "resource_availability",
        ],
        "process_sequence": [
            "sequence_number",
            "is_first_process",
            "is_last_process",
            "prev_process_duration",
            "next_process_duration",
        ],
        "quality_metrics": [
            "quality_score",
            "defect_rate",
            "process_accuracy",
            "first_pass_yield",
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
        "start_time",
        "end_time",
        "order_id",
        "start_time_unix",
        "end_time_unix",
    ]
    all_features = [col for col in final_data.columns if col not in non_feature_cols]
    feature_subsets["all_features"] = all_features

    print(f"Number of feature subsets: {len(feature_subsets)}")
    for component, features in feature_subsets.items():
        print(f"  {component}: {len(features)} features")

    # Run the hypothesis tests
    results = multiple_runs_hypothesis_test(
        final_data,
        feature_subsets,
        n_runs=10,  # Number of runs with different random seeds
        n_permutations=1000,  # Number of permutations for each test
        test_size=0.2,
        alpha=0.05,
        model_type="dt",  # Using decision tree classifier
        metric="accuracy",  # Using accuracy as the metric
        output_dir="hypothesis_results",
        max_depth=5,  # Additional parameter for the decision tree
    )

    # Print final conclusions
    print("\nFinal conclusions:")
    for component, res in results.items():
        conclusion = "INACCURATE" if res["rejection_rate"] > 0.5 else "ACCURATE"
        print(
            f"SBDT Component '{component}': {conclusion} (Rejection Rate: {res['rejection_rate']:.2f}, Mean {metric}: {res['mean_observed_stat']:.4f})"
        )
