"""
This script runs the identical data tests to validate the model's classification behavior.
It tests whether the model is truly distinguishing between real and simulated data
based on meaningful differences or just learning artifacts in the data.
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hypothesis_testing_twosampletest import (
    perform_identical_real_data_test,
    perform_identical_sim_data_test,
    define_feature_subsets,
    multiple_runs_hypothesis_test,
)

# Create output directory
output_dir = "identical_data_test_results"
os.makedirs(output_dir, exist_ok=True)

print("Starting identical data tests...")

# Run the identical real data test
real_test_data = perform_identical_real_data_test()
print(f"Real test data shape: {real_test_data.shape}")

# Run the identical simulated data test
sim_test_data = perform_identical_sim_data_test()
print(f"Simulated test data shape: {sim_test_data.shape}")

# Define feature subsets for testing
feature_subsets = define_feature_subsets()

# Run both decision tree and BiLSTM models on both test datasets
for model_type in ["dt", "lstm"]:
    print(f"\n===== Testing with {model_type.upper()} model =====")

    # Settings for the model type
    if model_type == "dt":
        n_runs = 10
        n_permutations = 1000
        alpha = 0.05
        additional_params = {"max_depth": 5}
    else:  # lstm
        n_runs = 10  # Fewer runs for LSTM due to computational intensity
        n_permutations = 200
        alpha = 0.01
        additional_params = {"n_epochs": 10, "batch_size": 32}

    # Set the evaluation metric
    metric = "roc_auc"

    # Test on identical real data
    print("\n-- Testing on identical real data --")
    real_output_dir = os.path.join(output_dir, f"real_{model_type}")
    real_results = multiple_runs_hypothesis_test(
        real_test_data,
        feature_subsets,
        n_runs=n_runs,
        n_permutations=n_permutations,
        test_size=0.2,
        alpha=alpha,
        model_type=model_type,
        metric=metric,
        output_dir=real_output_dir,
        **additional_params,
    )

    # Test on identical simulated data
    print("\n-- Testing on identical simulated data --")
    sim_output_dir = os.path.join(output_dir, f"sim_{model_type}")
    sim_results = multiple_runs_hypothesis_test(
        sim_test_data,
        feature_subsets,
        n_runs=n_runs,
        n_permutations=n_permutations,
        test_size=0.2,
        alpha=alpha,
        model_type=model_type,
        metric=metric,
        output_dir=sim_output_dir,
        **additional_params,
    )

    # Save summary results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_file = os.path.join(output_dir, f"{model_type}_summary_{timestamp}.txt")

    with open(summary_file, "w") as f:
        f.write(f"Summary Results for {model_type.upper()} Model\n")
        f.write("=" * 50 + "\n\n")

        f.write("Test on Identical Real Data:\n")
        f.write("-" * 30 + "\n")
        for component, res in real_results.items():
            f.write(f"{component}:\n")
            f.write(
                f"  Mean {metric}: {res.get('mean_observed_stat', float('nan')):.4f}\n"
            )
            f.write(f"  Mean p-value: {res.get('mean_p_value', float('nan')):.4f}\n")
            f.write(f"  Rejection rate: {res.get('rejection_rate', 0):.2f}\n\n")

        f.write("\nTest on Identical Simulated Data:\n")
        f.write("-" * 30 + "\n")
        for component, res in sim_results.items():
            f.write(f"{component}:\n")
            f.write(
                f"  Mean {metric}: {res.get('mean_observed_stat', float('nan')):.4f}\n"
            )
            f.write(f"  Mean p-value: {res.get('mean_p_value', float('nan')):.4f}\n")
            f.write(f"  Rejection rate: {res.get('rejection_rate', 0):.2f}\n\n")

    print(f"Summary saved to: {summary_file}")

    # Create comparison plots
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    # Extract data for plotting
    components = list(feature_subsets.keys())
    real_means = [real_results[c].get("mean_observed_stat", 0.5) for c in components]
    sim_means = [sim_results[c].get("mean_observed_stat", 0.5) for c in components]

    # Plot comparison of ROC AUC scores
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    index = range(len(components))

    plt.bar(
        [i - bar_width / 2 for i in index],
        real_means,
        bar_width,
        label="Identical Real Data",
        color="skyblue",
        alpha=0.8,
    )
    plt.bar(
        [i + bar_width / 2 for i in index],
        sim_means,
        bar_width,
        label="Identical Sim Data",
        color="salmon",
        alpha=0.8,
    )

    # Add horizontal line at 0.5 (random chance)
    plt.axhline(
        y=0.5, color="red", linestyle="--", alpha=0.7, label="Random Chance (0.5)"
    )

    plt.xlabel("Component")
    plt.ylabel(f"{metric.upper()}")
    plt.title(
        f"Comparison of {metric.upper()} on Identical Data Tests ({model_type.upper()})"
    )
    plt.xticks([i for i in index], components, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    comparison_plot_path = os.path.join(
        output_dir, f"{model_type}_{metric}_comparison.png"
    )
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()

    print(f"Comparison plot saved to: {comparison_plot_path}")

print("\nAll tests completed!")
