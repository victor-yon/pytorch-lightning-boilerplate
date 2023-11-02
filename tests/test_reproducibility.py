import unittest

from lightning import seed_everything

from datasets.project_dataset import ProjectDataModule
from models.simple_classifier import SimpleClassifier
from utils.trainer import ModelTrainer


class TestReproducibility(unittest.TestCase):
    @staticmethod
    def run_with_seed(seed: int):
        seed_everything(seed)

        model = SimpleClassifier()
        datamodule = ProjectDataModule()
        trainer = ModelTrainer(max_epochs=None, max_steps=5, val_check_interval=None)
        trainer.callbacks.clear()  # Remove every callback to speed up the tests
        trainer.fit(model, datamodule)
        return trainer.test(model, datamodule)[0]

    def test_training_reproducibility(self):
        results_a = self.run_with_seed(42)
        results_b = self.run_with_seed(10)
        results_c = self.run_with_seed(42)

        self.assertNotEquals(results_a, results_b,
                             "The same model trained with different seeds should produce different results "
                             "(it is very likely, at least)")
        self.assertDictEqual(results_a, results_c,
                             "The same model trained with the same seed should produce the same results")


if __name__ == '__main__':
    unittest.main()
