import logging
from argparse import ArgumentError
from typing import Iterable, Optional

import wandb
from lightning.pytorch.loggers import Logger, WandbLogger

PROJECT_NAME = 'demo'  # TODO Set project name


class OutputManager:
    """ Output manager """

    def __init__(self,
                 run_name: str = 'tmp',
                 log_level_console: str = 'info',
                 log_level_file: str = 'debug',
                 skip_plot: bool = False,
                 save_plot: bool = True,
                 save_trained_model: bool = True,
                 enable_wandb: bool = True,
                 wandb_api_key: Optional[str] = None):
        """
        Output manager to handle any logging, plot or model parameter extraction.

        Args:
            run_name: The name of the run. It will be used to name the log directory.
            log_level_console: The minimum logging level to show in the standard output.
                Possible values: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
            log_level_file: The minimum logging level to write in the log file.
                Possible values: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
            skip_plot: If True, the plots will not be processed.
            save_plot: If True, the plots will be saved in the logger service.
            save_trained_model: If True, the trained model parameters will be saved in the logger service.
            enable_wandb: If True, the Weights & Biases logger will be setup and used.
            wandb_api_key: The API key to use to log in Weights & Biases.
        """
        self.run_name = run_name
        self.log_level_console = OutputManager._parse_log_level(log_level_console)
        self.log_level_file = OutputManager._parse_log_level(log_level_file)
        self.skip_plot = skip_plot
        self.save_plot = save_plot
        self.save_trained_model = save_trained_model
        self.enable_wandb = enable_wandb
        self.wandb_api_key = wandb_api_key

        self.loggers = self._init_loggers()

    def _init_loggers(self) -> Iterable[Logger]:
        loggers = []

        if self.enable_wandb:
            if self.wandb_api_key:
                # Log user with API key
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
            The integer correspond to a valid log level.
        """
        try:
            # Try to parse the log level as an integer
            return logging.getLevelNamesMapping()[log_level.strip().upper()]
        except KeyError:
            raise ArgumentError(None, f'Invalid log level "{log_level}". Possible values: ' +
                                ', '.join(logging.getLevelNamesMapping().keys()))
