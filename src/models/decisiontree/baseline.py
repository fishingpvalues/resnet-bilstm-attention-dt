import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def run_decision_tree_pipeline(
    final_data,
    max_depth=5,
    random_state=683,
    test_size=0.2,
    report_fn=None,
    output_fn="decision_tree",
):
    """
    Runs a decision tree training and evaluation pipeline on the given final_data.

    Parameters:
        final_data (DataFrame): The preprocessed data including KPI features.
        max_depth (int): Maximum depth for the decision tree.
        random_state (int): Seed for reproducibility.
        test_size (float): Proportion of data to use as the test set.
        report_fn (callable): Optional function to generate a report (accepting y_test, y_pred, y_proba).
        output_fn (str): Filename (without extension) to save the exported decision tree.

    Returns:
        model: Trained DecisionTreeClassifier.
        splits: A tuple (X_train, y_train, X_test, y_test).
        predictions: A tuple (y_pred, y_proba).
    """
    # Ensure final_data is sorted for reproducibility.
    final_data = final_data.sort_values(
        by=["end_time", "order_id", "sequence_number"],
        key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
    ).reset_index(drop=True)

    # Generate features and target.
    X = final_data.drop(
        columns=[
            "is_valid",
            "start_time",
            "end_time",
            "order_id",
            "start_time_unix",
            "end_time_unix",
        ],
        errors="ignore",  # In case some columns are not present.
    )
    y = final_data["is_valid"]

    # Train Test Split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train Decision Tree.
    plt.figure(figsize=(20, 10), dpi=300)
    dt = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    dt.fit(X_train, y_train)

    # Make predictions.
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)[:, 1]

    # Generate a report if a reporting function is provided.
    if report_fn is not None:
        report_fn(y_test, y_pred, y_proba)

    # Export the decision tree using Graphviz.
    dot_data = export_graphviz(
        dt,
        out_file=None,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(output_fn)  # Saves to a file (e.g., decision_tree.pdf)
    graph.view()  # Opens the generated file in a viewer

    return dt, (X_train, y_train, X_test, y_test), (y_pred, y_proba)


if __name__ == "__main__":
    # For testing the pipeline directly.
    import pandas as pd

    # Replace the following line with actual data loading as needed.
    final_data = pd.read_csv(
        r"C:\resnet-bilstm-attention-dt\datasrc\final_data.csv",
        parse_dates=["start_time", "end_time"],
    )

    # Example simple report function.
    def generate_report(y_true, y_pred, y_proba):
        from sklearn.metrics import classification_report, roc_auc_score

        print(classification_report(y_true, y_pred))
        print("ROC-AUC:", roc_auc_score(y_true, y_proba))

    run_decision_tree_pipeline(
        final_data, max_depth=5, random_state=683, report_fn=generate_report
    )
