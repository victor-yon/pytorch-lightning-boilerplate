import logging

import wandb
from torchmetrics import Metric, MetricCollection


def plot_test_results(metrics: MetricCollection, confusion_matrix: Metric) -> None:
    """
    Plot the results of the test stage.

    Args:
        metrics: The collection of metrics.
        confusion_matrix: The test confusion matrix.
    """
    # Surround the plotting with a try-except block to avoid stopping the run if an error occurs
    try:
        # TODO add/edit plot test results functions
        plot_confusion_matrix(confusion_matrix)
    except Exception as e:
        logging.error(f'An error occurred while plotting the test results: {e}', exc_info=True)


def plot_confusion_matrix(confusion_matrix: Metric) -> None:
    """
    Plot the confusion matrix of the test stage.

    Args:
        confusion_matrix: The confusion matrix.
    """
    cm_data = confusion_matrix.compute()
    nb_classes = len(cm_data)

    flat_data = []
    for i in range(nb_classes):
        for j in range(nb_classes):
            flat_data.append([str(i), str(j), cm_data[i, j]])

    fields = {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"}
    wandb.log({'confusion_matrix': wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=flat_data),
        fields,
        {"title": 'Confusion Matrix'},
    )})
