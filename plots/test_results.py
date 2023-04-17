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
        logging.error(f'An error occurred while plotting the test results: {e}')


def plot_confusion_matrix(confusion_matrix: Metric) -> None:
    """
    Plot the confusion matrix of the test stage.

    Args:
        confusion_matrix: The confusion matrix.
    """
    cm_data = confusion_matrix.compute()
    # Upload the confusion matrix directly to wandb and let it plot it.
    # Multi-run confusion matrix using predefined plot scheme
    wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=cm_data,
                                                               y_true=list(range(10)),
                                                               class_names=[str(i) for i in range(10)])})
