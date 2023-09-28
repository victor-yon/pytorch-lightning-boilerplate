from typing import Optional

from lightning.pytorch import Trainer

from utils.output_manager import OutputManager


class ModelTrainer(Trainer):
    """ Model trainer """

    def __init__(self,
                 max_epochs: int = 50,
                 val_check_interval: Optional[float] = 1,
                 accelerator: str = 'auto',
                 output_manager: OutputManager = None):
        """
        Trainer with default values override for this project.

        Args:
            max_epochs: The number of training epochs.
            val_check_interval: The number of epochs between each validation. Set to None to disable validation.
            accelerator: The type of hardware accelerator to use for the training. Can be any from the following
                list: "cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto"
                (see pytorch-lightning Trainer class for more information).
            output_manager: A reference to the output manager of this project.
        """
        super().__init__(max_epochs=max_epochs,
                         val_check_interval=val_check_interval,
                         accelerator=accelerator,
                         logger=output_manager.loggers)

        self.output_manager = output_manager

        # TODO [template] override fit method to make to training optional or load model parameters for file
