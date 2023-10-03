import unittest

from lightning import seed_everything

from datasets.project_dataset import ProjectDataModule


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Fix the seed for the tests
        seed_everything(42)

    def test_loading_train_data(self):
        datamodule = ProjectDataModule()
        datamodule.prepare_data()
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()

        # Check the train dataloader
        self.assertIsNotNone(train_dataloader, "The train dataloader is not loaded correctly")
        self.assertGreater(len(train_dataloader), 0, "The train dataset is empty")
        self.assertIsNotNone(datamodule._dataset_train, "The train dataset is not loaded correctly")
        self.assertIsInstance(datamodule._dataset_train[0], tuple,
                              "The train dataset row is not a tuple. Expected (input_data, label)")
        self.assertEqual(len(datamodule._dataset_train[0]), 2,
                         "The train dataset row is tuple of size 2. Expected (input_data, label)")

        val_dataloader = datamodule.val_dataloader()

        # Check the validation dataloader
        self.assertIsNotNone(val_dataloader, "The validation dataloader is not loaded correctly")
        self.assertGreater(len(val_dataloader), 0, "The validation dataset is empty")
        self.assertIsNotNone(datamodule._dataset_val, "The validation dataset is not loaded correctly")
        self.assertIsInstance(datamodule._dataset_val[0], tuple,
                              "The validation dataset row is not a tuple. Expected (input_data, label)")
        self.assertEqual(len(datamodule._dataset_val[0]), 2,
                         "The validation dataset row is tuple of size 2. Expected (input_data, label)")

    def test_loading_test_data(self):
        datamodule = ProjectDataModule()
        datamodule.prepare_data()
        datamodule.setup("test")
        test_dataloader = datamodule.test_dataloader()

        # Check the test dataloader
        self.assertIsNotNone(test_dataloader, "The test dataloader is not loaded correctly")
        self.assertGreater(len(test_dataloader), 0, "The test dataset is empty")
        self.assertIsNotNone(datamodule._dataset_test, "The test dataset is not loaded correctly")
        self.assertIsInstance(datamodule._dataset_test[0], tuple,
                              "The test dataset row is not a tuple. Expected (input_data, label)")
        self.assertEqual(len(datamodule._dataset_test[0]), 2,
                         "The test dataset row is tuple of size 2. Expected (input_data, label)")

    def test_no_overlap(self):
        datamodule = ProjectDataModule()
        datamodule.prepare_data()
        datamodule.setup("fit")
        datamodule.setup("test")

        train_indices = set(datamodule._dataset_train.indices)
        val_indices = set(datamodule._dataset_val.indices)
        test_indices = set(datamodule._dataset_test.indices)

        common_indices = train_indices.intersection(val_indices)
        common_indices = train_indices.intersection(test_indices).union(common_indices)
        common_indices = test_indices.intersection(val_indices).union(common_indices)

        self.assertEqual(len(common_indices), 0, "The train, validation and test subsets overlap")


if __name__ == '__main__':
    unittest.main()
