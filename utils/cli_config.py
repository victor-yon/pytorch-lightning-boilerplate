from typing import Any, Tuple

import yaml
from lightning import Trainer
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from datasets.project_dataset import ProjectDataModule
from models.base_model import BaseModel
from models.simple_classifier import SimpleClassifier
from utils.output_manager import OutputManager
from utils.trainer import ModelTrainer

# Argument names for which we want to hide the values
_SECRET_ARGS = ('wandb_api_key',)


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

        self.output_manager = None

    def get_run_objects(self) -> Tuple[BaseModel, ProjectDataModule, ModelTrainer]:
        """
        Returns:
            All the objects initialized according to the configuration file and the program arguments.
            They are ready to be used for the training and testing.
        """
        return self.model, self.datamodule, self.trainer

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_core_arguments_to_parser(parser)

        # Add the arguments for the output manager
        parser.add_class_arguments(OutputManager, 'output', fail_untyped=True)

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """
        Add some CLI arguments to the parser if they are not related to a specific object.
        """
        # Define a model as the default one
        parser.groups['model'].set_defaults(model=SimpleClassifier.__name__)

        # TODO Add global arguments here if they are not related to a specific object.
        # For example:
        # arg_group = parser.add_argument_group('Group name')
        # arg_group.add_argument('--group_name.arg_name', type=int, default=42, help='Documentation here')

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # Create the output manager just before the trainer initialization, because we need to liked it to the trainer.
        self.output_manager = self._get(self.config_init, 'output')
        kwargs['output_manager'] = self.output_manager
        # TODO [template] remove output_manager from trainer CLI arguments
        return super().instantiate_trainer(**kwargs)

    def __str__(self):
        # default_dict = self.parser.get_defaults().as_dict()

        config_dict = dict()
        for key, value in self.config.as_dict().items():
            # Skip spacial arguments
            if key.startswith('__'):
                continue

            # Hide secret argument values
            config_dict[key] = value if key not in _SECRET_ARGS else '***'

        # Remove entry which have the default value
        # FIXME [template] this is not working because the values loaded from the config file are considered as default
        #  values
        # config_dict = {key: value for key, value in config_dict.items() if value != default_dict[key]}

        return 'Current configuration:\n' + yaml.dump(config_dict, allow_unicode=True)
