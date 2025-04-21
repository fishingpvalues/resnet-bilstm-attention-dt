"""
This script runs the Cauchy Combination Test (CCT) for hypothesis testing
on the dataset. It serves as an entry point to the hypothesis testing
functionality available in hypothesis_testing_cct.py.
"""

import os
import sys
import time
import colorama
import argparse

# Initialize colorama for colored terminal output
colorama.init()

# Define terminal colors for better logging
GREEN = colorama.Fore.GREEN
YELLOW = colorama.Fore.YELLOW
RED = colorama.Fore.RED
BLUE = colorama.Fore.BLUE
CYAN = colorama.Fore.CYAN
RESET = colorama.Fore.RESET

# Add the parent directory to sys.path to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hypothesis_testing_cct import (
    load_and_preprocess_data,
    define_feature_subsets,
    multiple_runs_hypothesis_test,
    generate_hypothesis_test_plots,
    generate_summary_plots,
)
from src.utils.config import get_output_dir


def run_hypothesis_tests(
    model_type="dt", n_runs=10, n_permutations=1000, output_dir=None
):
    """
    Run hypothesis tests using the specified model type.

    Args:
        model_type: Type of model to use ('dt' or 'lstm')
        n_runs: Number of runs with different random seeds
        n_permutations: Number of permutations for each test
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = "hypothesis_results_cct"

    os.makedirs(output_dir, exist_ok=True)

    print(f"{CYAN}Loading and preprocessing data...{RESET}")
    final_data = load_and_preprocess_data()
    print(f"{GREEN}Dataset shape: {final_data.shape}{RESET}")

    # Define feature subsets
    feature_subsets = define_feature_subsets()

    # Populate "all_features" with all available features
    non_feature_cols = ["is_valid"]
    all_features = [col for col in final_data.columns if col not in non_feature_cols]
    feature_subsets["all_features"] = all_features

    print(f"Number of feature subsets: {len(feature_subsets)}")
    for component, features in feature_subsets.items():
        print(f"  {component}: {len(features)} features")

    # Set the evaluation metric
    metric = "roc_auc"  # Using ROC AUC as the metric

    # Set parameters based on model type
    if model_type.lower() == "dt":
        alpha = 0.05
        additional_params = {"max_depth": 5}
    else:  # lstm
        alpha = 0.01
        additional_params = {"n_epochs": 10, "batch_size": 32}

    print(
        f"\n{YELLOW}===== Running hypothesis tests with {model_type.upper()} model ====={RESET}"
    )
    model_output_dir = os.path.join(output_dir, model_type.lower())

    # Run the hypothesis tests
    results = multiple_runs_hypothesis_test(
        final_data,
        feature_subsets,
        n_runs=n_runs,
        n_permutations=n_permutations,
        test_size=0.2,
        alpha=alpha,
        model_type=model_type,
        metric=metric,
        output_dir=model_output_dir,
        **additional_params,
    )

    # Print final conclusions
    print(f"\n{CYAN}Final conclusions:{RESET}")
    for component, res in results.items():
        if res["rejection_rate"] > 0.5:
            conclusion = f"{RED}INACCURATE{RESET}"
        else:
            conclusion = f"{GREEN}ACCURATE{RESET}"

        print(
            f"SBDT Component '{component}': {conclusion} (Rejection Rate: {res['rejection_rate']:.2f}, "
            f"Mean {metric.upper()}: {res['mean_observed_stat']:.4f}, "
            f"CCT p-value: {res.get('cct_p_value', float('nan')):.4f})"
        )

    # Generate additional comparative plots
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"\n{CYAN}Generating additional summary visualizations...{RESET}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hypothesis tests using CCT")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["dt", "lstm", "both"],
        help="Model type to use (dt, lstm, or both)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs with different random seeds",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of permutations for each test",
    )

    args = parser.parse_args()

    output_dir = "hypothesis_results_cct"
    os.makedirs(output_dir, exist_ok=True)

    if args.model.lower() == "both":
        print(f"{CYAN}Running tests for both Decision Tree and BiLSTM models{RESET}")
        dt_results = run_hypothesis_tests(
            "dt", args.runs, args.permutations, output_dir
        )
        lstm_results = run_hypothesis_tests(
            "lstm", args.runs, args.permutations, output_dir
        )

        # TODO: Add comparative visualization between DT and LSTM results if needed
    else:
        run_hypothesis_tests(args.model, args.runs, args.permutations, output_dir)

    print(f"\n{CYAN}All tests completed!{RESET}")
