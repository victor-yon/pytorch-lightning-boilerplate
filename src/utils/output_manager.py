from __future__ import annotations

import logging
import re
import sys
from argparse import ArgumentError
from pathlib import Path
from typing import Iterable, Optional

import loguru
import wandb
import yaml
from lightning.pytorch.loggers import Logger, MLFlowLogger, WandbLogger
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from utils.local_logger import LocalLogger

PROJECT_NAME = 'demo'  # TODO Set project name

# Log format
LOG_FORMAT_CONSOLE = '{time:HH:mm:ss.SSS} |<level>{level:>8}</level>| <level>{message}</level>'
LOG_FORMAT_FILE = '{time:YYYY-MM-DD HH:mm:ss.SSSS} {level:>8} ({file}:{line}) {message}'

# Reduce the logging level of info and debug messages since logging from external libraries is less important
EXTERNAL_LEVEL_MAP = {'CRITICAL': 'CRITICAL', 'ERROR': 'ERROR', 'WARNING': 'WARNING', 'INFO': 'DEBUG', 'DEBUG': 'TRACE'}


class OutputManager:
    """ Output manager """

    def __init__(self,
                 run_name: str = 'tmp',
                 log_level_console: Optional[str] = 'info',
                 log_level_files: Optional[str] = 'debug',
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
            log_level_console: The minimum logging level to show in the standard output. Disabled if None or empty.
                Possible values: CRITICAL, ERROR, WARNING, SUCCESS, INFO, DEBUG, TRACE
            log_level_files: The minimum logging level to write in the log file. Disabled if None or empty.
                Possible values: CRITICAL, ERROR, WARNING, SUCCESS, INFO, DEBUG, TRACE
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
        """
        Initialize every logger (console, local file, Weights & Biases, MLFlow), depending on the current configuration.
        Returns: The list of remote loggers.
        """

        remote_loggers = []
        local_logger = LocalLogger(name=self.run_name)
        remote_loggers.append(local_logger)

        # Setup local loggers (console and file)
        local_loggers = []
        if self.log_level_console is not None:
            local_loggers.append(dict(sink=sys.stdout, colorize=True, level=self.log_level_console.name,
                                      format=LOG_FORMAT_CONSOLE))
        if self.log_level_files is not None:
            log_file = Path(local_logger.log_dir) / 'run.log'
            local_loggers.append(dict(sink=log_file, level=self.log_level_files.name, format=LOG_FORMAT_FILE))

        logger.configure(handlers=local_loggers)

        # Setup remote loggers (MLFlow and Weights & Biases)
        if self.enable_mlflow:
            remote_loggers.append(MLFlowLogger(experiment_name=PROJECT_NAME, run_name=self.run_name,
                                               tracking_uri=self.mlflow_tracking_uri, artifact_location='./out',
                                               log_model=self.save_trained_model))

        if self.enable_wandb:
            if self.wandb_api_key:
                # Log user with an API key
                if wandb.login(key=self.wandb_api_key):
                    logger.info('Weights & Biases login successful.')
                else:
                    logger.error('Weights & Biases login failed.')

            remote_loggers.append(WandbLogger(project=PROJECT_NAME, name=self.run_name, save_dir='./out'))

        return remote_loggers

    def log_config(self, config: dict) -> None:
        """
        Log the configuration of the current run.

        Args:
            config: The configuration to log as a dictionary.
        """

        logger.debug(f'Run configuration:\n{yaml.dump(config, allow_unicode=True)}')

        for log in self.loggers:
            if isinstance(log, WandbLogger):
                # Log run configuration with Weights & Biases (do not use default log_hyperparams)
                log.experiment.config.update(config)
            else:
                # Log run configuration
                log.log_hyperparams(config)

    def is_plot_enabled(self) -> bool:
        """
        Check if it is necessary to run the plot code.

        Returns:
            True at least on of the following option is enabled: show_plots, save_plots, upload_plots.
        """
        return self.show_plots or self.save_plots or self.upload_plots

    def process_plot(self, fig: Figure, file_name: str) -> None:
        """
        Process a matplotlib figure.
        The image will be saved locally as a file, uploaded to the logger service and/or shown depending on the
        configuration.

        Args:
            fig: The figure to save.
            file_name: The output file name (no extension). Underscores will replace spaces and slashes.

        Returns: The path where the plot is saved as a local file, or None if not saved.
        """
        # Upload/save the plot with the logger service
        for log in self.loggers:
            if isinstance(log, MLFlowLogger) and self.upload_plots:
                log.experiment.log_figure(log.run_id, fig, file_name)
            if isinstance(log, WandbLogger) and self.upload_plots:
                log.experiment.log({file_name: wandb.Image(fig)})
            if isinstance(log, LocalLogger) and self.save_plots:
                log.log_image(fig, file_name, self.image_latex_format)

        # Plot image or close it
        fig.show() if self.show_plots else plt.close(fig)

    @staticmethod
    def _parse_log_level(log_level: Optional[str]) -> Optional[loguru.Level]:
        """
        Parse a log level from a string.

        Args:
            log_level: The log level name to parse.

        Returns:
            A tuple that represents a valid log level, or None if no logging.
        """

        if log_level is None or log_level.strip() == '':
            return None

        try:
            # Try to parse the log level as an integer
            return logger.level(log_level.strip().upper())
        except ValueError:
            raise ArgumentError(None, f'Invalid log level "{log_level}". Possible values: ' +
                                ', '.join(['CRITICAL', 'ERROR', 'WARNING', 'SUCCESS', 'INFO', 'DEBUG', 'TRACE']))

    @staticmethod
    def init_default_console_logger() -> None:
        """
        Initialize a default console logger configuration.
        It should be called as early as possible in the program to make sure it catches all the logs.
        Once the user configuration is loaded, the handler will be overridden.
        """
        logging.captureWarnings(True)  # Capture warnings with the logging system

        logger.configure(
            # Temporary handler that will be overridden after the user configuration is loaded
            handlers=[dict(sink=sys.stdout, colorize=True, level='INFO', format=LOG_FORMAT_CONSOLE)],
            # Update levels colors
            levels=[
                dict(name='CRITICAL', color='<red><bold>'),
                dict(name='ERROR', color='<red>'),
                dict(name='WARNING', color='<yellow>'),
                dict(name='SUCCESS', color='<green><bold>'),
                dict(name='INFO', color=''),
                dict(name='DEBUG', color='<fg #8C8C8C>'),  # light gray
                dict(name='TRACE', color='<fg #8C8C8C><italic>')  # light gray
            ]
        )

        class InterceptHandler(logging.Handler):
            """ Intercept standard logging messages and redirect them to Loguru. """

            def emit(self, record: logging.LogRecord) -> None:
                """ Override the emit method. """
                # Get the corresponding Loguru level if it exists
                level: str | int
                try:
                    level = EXTERNAL_LEVEL_MAP[logger.level(record.levelname).name]
                except ValueError:
                    level = record.levelno

                message = record.getMessage()

                # Clean up user warnings
                if level == 'WARNING':
                    message = re.sub(r'.*UserWarning: ', '', message)
                    message = re.sub(r'rank_zero_warn\(.*', '', message)

                logger.opt(exception=record.exc_info).log(level, message.strip())

        # Redirect default logging messages to Loguru
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # Redirect pytorch-lightning logging messages to Loguru
        for logger_name in ['lightning.pytorch', 'lightning.fabric', 'pytorch_lightning']:
            logging.getLogger(logger_name).handlers.clear()
            logging.getLogger(logger_name).addHandler(InterceptHandler())

        # Ignore matplotlib info and debug messages (too verbose)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
