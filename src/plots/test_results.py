import seaborn as sns
import wandb
from loguru import logger
from torchmetrics import Metric, MetricCollection

from utils.output_manager import OutputManager


def plot_test_results(output_manager: OutputManager, metrics: MetricCollection, confusion_matrix: Metric) -> None:
    """
    Plot the results of the test stage.

    Args:
        output_manager: The output manager to handle the plots.
        metrics: The collection of metrics.
        confusion_matrix: The test confusion matrix.
    """

    # Check if it is necessary to create the plots
    if not output_manager.is_plot_enabled():
        return

    # Surround the plotting with a try-except block to avoid stopping the run if an error occurs
    # noinspection PyBroadException
    try:
        # Plot the confusion matrix using matplotlib and Weight and Biases.
        plot_confusion_matrix(output_manager, metrics, confusion_matrix)
        if output_manager.enable_wandb:
            plot_confusion_matrix_wandb(confusion_matrix)
        # TODO add/edit plot test results functions
    except Exception:
        logger.exception(f'An error occurred while plotting the test results')


def plot_confusion_matrix(output_manager: OutputManager, metrics: MetricCollection, confusion_matrix: Metric,
                          annotations: bool = True) -> None:
    """
    Plot the confusion matrix using matplotlib.

    Args:
        output_manager: The output manager to handle the plots.
        metrics: The classification metrics.
        confusion_matrix: The confusion matrix.
        annotations: Whether to show the values or not in the squares of the confusion matrix. Not recommended for large
            confusion matrices.
    """

    cm_data = confusion_matrix.compute()
    nb_class = len(cm_data)
    ax = sns.heatmap(cm_data,
                     vmin=0,
                     vmax=1,
                     square=True,
                     fmt='.0%' if nb_class > 4 else '.1%',
                     cmap='Blues',
                     xticklabels=[str(i) for i in range(nb_class)],
                     yticklabels=[str(i) for i in range(nb_class)],
                     annot=annotations,
                     cbar=(not annotations))
    fig = ax.get_figure()
    ax.set_title(f'Confusion matrix of {len(cm_data)} classes\nAccuracy: {metrics.acc.compute():.2%}')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Labels')

    # Save, show and/or upload the plot, depending on the output manager configuration
    output_manager.process_plot(fig, 'confusion_matrix')


def plot_confusion_matrix_wandb(confusion_matrix: Metric) -> None:
    """
    Plot the confusion matrix using Weight and Biases.

    Args:
        confusion_matrix: The confusion matrix.
    """
    cm_data = confusion_matrix.compute()
    nb_classes = len(cm_data)

    # Convert the confusion matrix to a flat list
    flat_data = []
    for i in range(nb_classes):
        for j in range(nb_classes):
            flat_data.append([str(i), str(j), cm_data[i, j]])

    wandb.log({'confusion_matrix': wandb.plot_table(
        # Define Vega plot scheme
        "wandb/confusion_matrix/v1",
        # Log the table with data
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=flat_data),
        # Define the mapping between the table columns and the plot scheme
        {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"},
        # Set the plot title
        {"title": 'Confusion Matrix'},
    )})
