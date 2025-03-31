# REPORTING UTILITY

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def generate_report(y_true, y_pred, y_proba):
    """
    Generates a detailed report of binary classifier performance.

    Args:
        y_true (array-like): True target labels.
        y_pred (array-like): Predicted class labels.
        y_proba (array-like): Probabilities for the positive class.
    """
    # Print classification report and key metrics
    report = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Classification Report:")
    print(report)
    print("Accuracy: {:.2f}".format(accuracy))
    print("ROC AUC: {:.2f}".format(auc))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))

    # Compute confusion matrix and ROC curve
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # Create subplots for Confusion Matrix and ROC Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot confusion matrix
    im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    # Loop over data dimensions and add text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Plot ROC curve
    ax2.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = {:.2f})".format(auc),
    )
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Model")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Receiver Operating Characteristic (ROC)")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
