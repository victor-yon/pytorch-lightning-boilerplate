from typing import Tuple

from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from datasets.project_dataset import ProjectDataModule
from models.base_model import BaseModel
from models.simple_classifier import SimpleClassifier
from utils.trainer import ModelTrainer


class CLIConfig(LightningCLI):
    def __init__(self):
        """
        Custom Command Line Interface with default values set for the project.
        By default, override configurations with user file: "./config.yaml".
        The automatic run is disabled to have more control over the run process.
        """
        super().__init__(model_class=BaseModel,
                         datamodule_class=ProjectDataModule,
                         trainer_class=ModelTrainer,
                         run=False,
                         parser_kwargs={'default_config_files': ['config.yaml']},
                         save_config_kwargs={'overwrite': True},
                         subclass_mode_model=True)

    def get_run_objects(self) -> Tuple[BaseModel, ProjectDataModule, ModelTrainer]:
        """
        Returns:
            All the objects initialized according to the configuration file and the program arguments.
            They are ready to be used for the training and testing.
        """
        return self.model, self.datamodule, self.trainer

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """
        Add some CLI arguments to the parser if they are not related to a specific object.
        """
        # Define a model as the default one
        parser.groups['model'].set_defaults(model=SimpleClassifier.__name__)
        # parser.set_defaults(optimizer='Adam')
        # TODO Add global arguments here if they are not related to a specific object.
        # For example:
        # arg_group = parser.add_argument_group('Group name')
        # arg_group.add_argument('--group_name.arg_name', type=int, default=42, help='Documentation here')
        pass
