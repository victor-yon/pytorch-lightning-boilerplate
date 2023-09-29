import seaborn as sns

from utils.output_manager import OutputManager


def setup_environment():
    """
    Set up the environment and context for the project.
    """
    OutputManager.init_default_console_logger()
    setup_plot_style()


def setup_plot_style():
    """
    Defined the default matplotlib style to use.
    """
    sns.set_theme(rc={
        'axes.titlesize': 15,
        'figure.titlesize': 18,
        'axes.labelsize': 13,
        'figure.autolayout': True,
        'figure.constrained_layout.use': True,  # Use constrained layout by default
        'svg.fonttype': 'none'  # Assume fonts are installed on the machine where the SVG will be viewed
    })
