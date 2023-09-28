import logging
import re
from argparse import ArgumentError
from pathlib import Path
from typing import Iterable, Optional

import wandb
from lightning.pytorch.loggers import Logger, MLFlowLogger, WandbLogger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

PROJECT_NAME = 'Demo'  # TODO Set project name


class OutputManager:
    """ Output manager """

    def __init__(self,
                 run_name: str = 'tmp',
                 log_level_console: str = 'info',
                 log_level_files: str = 'debug',
                 save_plots: bool = True,
                 show_plots: bool = False,
                 upload_plots: bool = True,
                 image_latex_format: bool = False,
                 enable_mlflow: bool = False,
                 mlflow_tracking_uri: Optional[str] = None,
                 enable_wandb: bool = False,
                 wandb_api_key: Optional[str] = None,
                 save_trained_model: bool = True):
        """
        Output manager to handle any logging, plot or model parameter extraction.

        Args:
            run_name: The name of the run. It will be used to name the log directory.
            log_level_console: The minimum logging level to show in the standard output.
                Possible values: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
            log_level_files: The minimum logging level to write in the log file.
                Possible values: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
            save_plots: If True, the plots will be locally saved as a file.
            show_plots: If True, the plots will show when ready.
            upload_plots: If True, the plots will be uploaded to the logger service.
            image_latex_format: If True, the plots will be saved as vectoriel image for a better integration in a latex
                document. No effect if save_plots is False. It does not affect the uploaded version.
            enable_mlflow: If True, the MLFlow logger will be setup and used.
            mlflow_tracking_uri: The URI of the MLFlow server to use.
            enable_wandb: If True, the Weights & Biases logger will be setup and used.
            wandb_api_key: The API key to use to log in Weights & Biases.
            save_trained_model: If True, the trained model parameters will be saved in the logger service.
        """
        self.run_name = run_name

        # Logging configuration
        self.log_level_console = OutputManager._parse_log_level(log_level_console)
        self.log_level_files = OutputManager._parse_log_level(log_level_files)

        # Plot configuration
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.upload_plots = upload_plots
        self.image_latex_format = image_latex_format

        # MLFlow configuration
        self.enable_mlflow = enable_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri

        # Weights & Biases configuration
        self.enable_wandb = enable_wandb
        self.wandb_api_key = wandb_api_key

        # Other configuration
        self.save_trained_model = save_trained_model

        self.loggers = self._init_loggers()

    def _init_loggers(self) -> Iterable[Logger]:
        loggers = []

        if self.enable_mlflow:
            loggers.append(MLFlowLogger(experiment_name=PROJECT_NAME, run_name=self.run_name,
                                        tracking_uri=self.mlflow_tracking_uri, artifact_location='./out',
                                        log_model=self.save_trained_model))

        if self.enable_wandb:
            if self.wandb_api_key:
                # Log user with an API key
                if wandb.login(key=self.wandb_api_key):
                    logging.info('Weights & Biases login successful.')
                else:
                    logging.error('Weights & Biases login failed.')

            loggers.append(WandbLogger(project=PROJECT_NAME, name=self.run_name))

        return loggers

    @staticmethod
    def _parse_log_level(log_level: str) -> int:
        """
        Parse a log level from a string to an integer.

        Args:
            log_level: The log level name to parse.

        Returns:
            An integer that corresponds to a valid log level.
        """
        try:
            # Try to parse the log level as an integer
            return logging.getLevelNamesMapping()[log_level.strip().upper()]
        except KeyError:
            raise ArgumentError(None, f'Invalid log level "{log_level}". Possible values: ' +
                                ', '.join(logging.getLevelNamesMapping().keys()))

    def log_config(self, config: dict) -> None:
        """
        Log the configuration of the current run.

        Args:
            config: The configuration to log as a dictionary.
        """
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                # Log run configuration with Weights & Biases
                logger.experiment.config.update(config)
            if isinstance(logger, MLFlowLogger):
                # Log run configuration with MLFlow
                logger.log_hyperparams(config)

    def is_plot_enabled(self) -> bool:
        """
        Check if it is necessary to run the plot code.

        Returns:
            True at least on of the following option is enabled: show_plots, save_plots, upload_plots.
        """
        return self.show_plots or self.save_plots or self.upload_plots

    def process_plot(self, fig: Figure, file_name: str) -> Path | None:
        """
        Process a matplotlib figure.
        The image will be saved locally as a file, uploaded to the logger service and/or shown depending on the
        configuration.

        Args:
            fig: The figure to save.
            file_name: The output file name (no extension). Underscores will replace spaces and slashes.

        Returns: The path where the plot is saved as a local file, or None if not saved.
        """

        # Clean the file name
        file_name = re.sub(r'[\s\\/]+', '_', file_name.strip())

        save_path = None
        if self.save_plots:
            out_format = 'svg' if self.image_latex_format else 'png'
            # TODO [template] Find a way to save the plot locally
            # save_path = get_save_path(Path(OUT_DIR, self.run_name, 'img'), file_name, out_format, allow_overwrite)
            save_path = Path('out', 'tmp', 'img', f'{file_name}.{out_format}')
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(save_path, dpi=200, transparent=self.image_latex_format)
            logging.debug(f'Plot saved in {save_path}')

        # Upload the plot to the logger service
        if self.upload_plots:
            if self.enable_wandb:
                wandb.log({file_name: wandb.Image(fig)})

            for logger in self.loggers:
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_figure(logger.run_id, fig, file_name)

        # Plot image or close it
        fig.show() if self.show_plots else plt.close(fig)

        return save_path
