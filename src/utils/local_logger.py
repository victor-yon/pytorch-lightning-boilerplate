import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Union, Dict, Any, Optional, List

import yaml
from lightning.fabric.loggers.csv_logs import _ExperimentWriter
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from matplotlib.figure import Figure
from pytorch_lightning.loggers import CSVLogger


class LocalLogger(CSVLogger):

    def __init__(self, name: str = "tmp", version: Optional[Union[int, str]] = None,
                 flush_logs_every_n_steps: int = 100):
        super().__init__('./out', name, version, '', flush_logs_every_n_steps)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        config_file = Path(self.log_dir) / "config.yaml"
        if config_file.exists():
            logger.warning(f'Overwrite existing config file "{config_file}".')

        with open(config_file, "w") as f:
            f.write(yaml.dump(params, allow_unicode=True))

        logger.trace(f'Plot configuration in "{config_file}"')

    @rank_zero_only
    def log_image(self, fig: Figure, file_name: str, image_latex_format: bool):
        """
        Save a matplotlib figure in the log directory.

        Args:
            fig: The figure to save.
            file_name: The name of the file to save (special characters will be replaced).
            image_latex_format: If True, the image will be saved in a format compatible with LaTeX.
        """
        img_dir = Path(self.log_dir) / "img"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Clean the file name
        file_name = re.sub(r'[\s\\/]+', '_', file_name.strip())

        # Construct the save path
        out_format = 'svg' if image_latex_format else 'png'
        save_path = img_dir / f'{file_name}.{out_format}'

        # Save the image
        fig.savefig(save_path, dpi=200, transparent=image_latex_format)
        logger.trace(f'Plot saved in "{save_path}"')

    def _get_next_version(self) -> int:
        versions_root = Path(self._root_dir, self.name)
        versions_root.mkdir(parents=True, exist_ok=True)

        existing_versions = []
        for d in versions_root.iterdir():
            if d.is_dir() and d.name.startswith("version_"):
                existing_versions.append(int(d.name.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _LocalExperimentWriter(log_dir=self.log_dir)
        return self._experiment


class _LocalExperimentWriter(_ExperimentWriter):
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_METRICS_FILE = "metrics.csv"

    def __init__(self, log_dir: str) -> None:
        """ Override to remove already existing warning. """
        self.metrics: List[Dict[str, float]] = []

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self._fs.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)
