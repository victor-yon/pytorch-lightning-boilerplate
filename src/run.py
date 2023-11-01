from loguru import logger

from utils.cli_config import CLIConfig
from utils.setup_env import setup_environment


@logger.catch
def run():
    """ Run the program (loading, training, testing). """
    # Set up the environment and context
    setup_environment()

    # Initialize the model, the dataset, and the trainer, according to the configuration file and the program arguments
    model, datamodule, trainer = CLIConfig().get_run_objects()

    # Training or load the model from file
    if trainer.load_model_path is not None:
        model = trainer.load_model_from_file(model.__class__)
    else:
        trainer.fit(model, datamodule)

    # Testing
    trainer.test(model, datamodule)


if __name__ == '__main__':
    run()
