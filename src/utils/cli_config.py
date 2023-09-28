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
            They are ready to be used for training and testing.
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
        """
        Override the default trainer instantiation to instantiation the output manager before the trainer.
        """
        # Create the output manager just before the trainer initialization, because we need to link it to the trainer.
        self.output_manager = self._get(self.config_init, 'output')
        # Log the configuration
        self.output_manager.log_config(self.get_config_dict())
        # Link the output manager to the model
        self.model.output_manager = self.output_manager
        # Link the output manager to the trainer
        kwargs['output_manager'] = self.output_manager
        # TODO [template] remove output_manager from trainer CLI arguments
        return super().instantiate_trainer(**kwargs)

    def get_config_dict(self) -> dict:
        """
        :return: The current configuration as a dictionary with the secret arguments hidden and
            special arguments removed.
        """

        def recursive_process_dict(ref_d: dict, d: dict) -> None:
            """
            Recursively copy the config dictionary to hide secret arguments and remove special arguments.
            We assume there is no circular reference.

            :param ref_d: The reference config dictionary
            :param d: The dictionary to fill
            """
            for key, value in ref_d.items():
                if isinstance(value, dict):
                    d[key] = dict()
                    recursive_process_dict(value, d[key])
                else:
                    # Skip spacial arguments
                    if key.startswith('__'):
                        continue
                    # Hide secret argument values
                    d[key] = value if key not in _SECRET_ARGS or not value else '<secret>'

        processed_dict = dict()
        recursive_process_dict(self.config.as_dict(), processed_dict)

        return processed_dict

    def __str__(self):
        # default_dict = self.parser.get_defaults().as_dict()

        config_dict = self.get_config_dict()

        # Remove entry which have the default value
        # FIXME [template] this is not working because the values loaded from the config file are considered as default
        #  values
        # config_dict = {key: value for key, value in config_dict.items() if value != default_dict[key]}

        return 'Current configuration:\n' + yaml.dump(config_dict, allow_unicode=True)
