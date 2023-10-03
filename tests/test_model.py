import unittest

from lightning import Trainer, seed_everything

from datasets.project_dataset import ProjectDataModule
from models.simple_classifier import SimpleClassifier


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Fix the seed for the tests
        seed_everything(42)

    def test_model_fit_datamodule(self):
        model = SimpleClassifier()
        datamodule = ProjectDataModule()

        # Use pytorch-lightning default trainer with fast_dev_run to test the model
        trainer = Trainer(fast_dev_run=True)

        # noinspection PyBroadException
        try:
            trainer.fit(model, datamodule)
        except Exception:
            self.fail("Error during model fit with datamodule. It could be a logic or structure issue in the model "
                      "or the dataset.")


if __name__ == '__main__':
    unittest.main()
