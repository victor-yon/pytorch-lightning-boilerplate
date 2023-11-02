from pathlib import Path
from typing import Optional, Type

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar, ModelSummary
from loguru import logger

from models.base_model import BaseModel
from utils.output_manager import OutputManager


class ModelTrainer(Trainer):
    """ Model trainer """

    def __init__(self,
                 max_epochs: Optional[int] = 50,
                 max_steps: Optional[int] = None,
                 load_model_path: Optional[str] = None,
                 val_check_interval: Optional[float] = 1,
                 accelerator: str = 'auto',
                 output_manager: OutputManager = None):
        """
        Trainer with default values override for this project.

        Args:
            max_epochs: The number of training epochs.
            max_steps: The number of training steps.
                If set, it will override the max_epochs parameter.
                None, 0 or -1 will disable this parameter.
            load_model_path: A path to a pickle file containing the parameter values to load.
                If this path is set and valid, the model will be loaded from this file instead of training a new one.
                The model architecture must be the same as the one defined in the model class.
            val_check_interval: The number of epochs between each validation. Set to None to disable validation.
            accelerator: The type of hardware accelerator to use for training and testing. Can be any from the following
                list: "cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto" (see pytorch-lightning Trainer class for more
                information).
            output_manager: A reference to the output manager of this project.
        """
        super().__init__(max_epochs=max_epochs,
                         max_steps=max_steps or -1,
                         val_check_interval=val_check_interval,
                         accelerator=accelerator,
                         logger=output_manager.loggers if output_manager else False,
                         callbacks=[RichProgressBar(), ModelSummary()])

        self.output_manager = output_manager

        if max_epochs and max_epochs > 0 and max_steps and max_steps > 0:
            logger.warning(f'Both "max_epochs" and "max_steps" are set. "max_steps" will be used ({max_steps:,d}).')

        # If a load path defined, check it exists
        if load_model_path is not None and load_model_path.strip() != '':
            self.load_model_path = Path(load_model_path)

            if not self.load_model_path.is_file():
                raise FileNotFoundError(f'Cannot find the model file "{self.load_model_path.resolve()}"')

        else:
            self.load_model_path = None

    def load_model_from_file(self, model_class: Type[BaseModel]) -> BaseModel:
        """
        Load the parameters of the model from a file.

        Args:
            model_class: The class of the model to load.

        Returns:
            The model with the parameters loaded from the file.
        """
        model = model_class.load_from_checkpoint(self.load_model_path)
        # FIXME: In the MNIST example the test set will be different than the one used for training.
        #   This will lead to artificially better accuracy.
        #   An elegant way yo solve this would be to save and load the _split_seed attribute of the dataset as a state.
        # Link the current output manager to the model
        model.output_manager = self.output_manager

        logger.info(f'Model parameters loaded from "{self.load_model_path}"')

        return model
