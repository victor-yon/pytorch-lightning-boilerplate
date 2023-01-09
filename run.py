from utils.run_cli import RunCLI
from utils.setup_enviroment import setup_environment

if __name__ == '__main__':
    # Set up the environment and context
    setup_environment()

    # Initialize the model, the dataset and the trainer according to the configuration file and the program arguments
    model, datamodule, trainer = RunCLI().get_run_objects()

    # Training
    trainer.fit(model, datamodule)

    # Testing
    trainer.test(model, datamodule)
