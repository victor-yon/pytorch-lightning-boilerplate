from typing import Optional

from lightning.pytorch import Trainer


class ModelTrainer(Trainer):
    """ Model trainer """
    def __init__(self,
                 run_name: str = 'tmp',
                 out_path: str = '../out',
                 max_epochs: int = 50,
                 val_check_interval: Optional[float] = 1,
                 accelerator: str = 'auto'):
        """
        Trainer with default values set for this project.

        Args:
            run_name: The name of the run. It will be used to name the log directory.
            out_path: The path to the output directory. It will be created if it doesn't exist.
            max_epochs: The number of training epochs.
            val_check_interval: The number of epochs between each validation. Set to None to disable validation.
            accelerator: The type of hardware accelerator to use for the training. Can be any from the following
                list: "cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto"
                (see pytorch-lightning Trainer class for more information).
        """
        super().__init__(max_epochs=max_epochs,
                         val_check_interval=val_check_interval,
                         logger=False,  # TODO [template] add logger
                         accelerator=accelerator)

        # TODO [template] override fit method to make to training optional or load model parameters for file
