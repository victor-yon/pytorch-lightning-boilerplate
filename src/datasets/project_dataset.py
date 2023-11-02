import os
import random
from math import ceil
from typing import List, Optional

import numpy as np
import openml
from lightning.pytorch import LightningDataModule
from loguru import logger
from numpy.typing import NDArray
from tabulate import tabulate
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split, Subset


class ProjectDataset(Dataset):
    def __init__(self, data: NDArray[float], labels: NDArray[int]):
        """
        Create a dataset instance.

        Args:
            data: The data row that will be used as input for the model.
            labels: The labels for each row.
        """
        self.data: NDArray[float] = data
        self.labels: NDArray[float] = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ProjectDataModule(LightningDataModule):
    """ The dataset """

    def __init__(self, dataset_name: str = 'mnist_784', batch_size: int = 256):
        """
        Create a data module instance, that contains the different subsets of the dataset.

        Args:
            dataset_name: The name of the dataset to download from OpenML.
                See https://www.openml.org/search?type=data&status=active for the list of available datasets.
            batch_size: The size of the batch to use for the training.
        """
        super().__init__()
        self.dataset_name: str = dataset_name
        self.classes: Optional[List] = None
        self.batch_size: int = batch_size

        # Objects to store the different datasets instances
        self._dataset_train: Optional[Subset[ProjectDataset]] = None
        self._dataset_test: Optional[Subset[ProjectDataset]] = None
        self._dataset_val: Optional[Subset[ProjectDataset]] = None

        # Generate a seed for the dataset split to make sure that every call of setup() will return the sets, even in
        # distributed scenarios. It should also guarantee the reproducibility if the global seed is fixed.
        self._split_seed = random.randint(0, int(1e12))
        logger.debug(f'Dataset split seed: {self._split_seed}')

        # Can change later depending on the accelerator type
        self._nb_workers: int = 0

    def download_dataset(self) -> ProjectDataset:
        """
        Download the dataset from OpenML and create the dataset.
        Returns:
            An instance of the full dataset.
        """
        # Download dataset or load if from cache
        dataframe, _, _, _ = openml.datasets.get_dataset(
            self.dataset_name,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False
        ).get_data(dataset_format='dataframe')
        # Get pixels values (all columns except the last one). The type should match model parameters.
        data = dataframe.iloc[:, :-1].to_numpy(dtype=np.float32)
        # Get labels (last column)
        labels = dataframe['class'].to_numpy(dtype=int)

        # Create the global dataset
        return ProjectDataset(data, labels)

    def prepare_data(self) -> None:
        """
        Download and prepare the dataset for future use.
        It is important to not change the state of the object here, as it might be called in parallel devices.
        """
        # Download dataset if no already cached
        self.download_dataset()

    def setup(self, stage: Optional[str]) -> None:
        # Get the number of workers to use for the data loading
        self.set_auto_nb_workers()

        # Load the dataset from cache
        dataset = self.download_dataset()

        # Split the dataset sets, with a fixed seed for reproducibility and consistency in case of distributed runs
        train_set, test_set, val_set = random_split(dataset, [0.7, 0.2, 0.1],
                                                    Generator().manual_seed(self._split_seed))

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.log_sets_info(train_set, test_set, val_set)  # Log information about the dataset
            self._dataset_train = train_set
            self._dataset_val = val_set
            logger.debug(f'Train and validation datasets loaded')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self._dataset_test = test_set
            logger.debug(f'Test dataset loaded')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_train, shuffle=True, batch_size=self.batch_size, num_workers=self._nb_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_val, shuffle=False, batch_size=self.batch_size, num_workers=self._nb_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_test, shuffle=False, batch_size=self.batch_size, num_workers=self._nb_workers)

    def set_auto_nb_workers(self) -> None:
        """
        Automatically search for the optimal number of workers (processes that load the data), depending on the
        accelerator type and the number of available CPU.
        """
        if self.trainer is None:
            # The trainer is not initialized yet, so we can't get the accelerator type
            return

        # Try to detect the number of available CPU cores
        try:
            nb_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            nb_workers = os.cpu_count()

        self._nb_workers = ceil(nb_workers / 2)  # The optimal number seems to be half of the cores

        logger.debug(f'Number of workers for data loading: {self._nb_workers}')

    def log_sets_info(self, train_set, test_set, valid_set) -> None:
        """
        Log information about the dataset and the different subsets.
        """
        train_size = len(train_set)
        test_size = len(test_set)
        val_size = len(valid_set)
        total_size = train_size + test_size + val_size

        logger.info(f'Dataset "{self.dataset_name}" loaded\n' +
                    tabulate({
                        "Dataset": ["Train", "Test", "Validation", "Total"],
                        "Size": [train_size, test_size, val_size, total_size],
                        "Percentage": [train_size / total_size, test_size / total_size, val_size / total_size, 1]
                    }, headers="keys", tablefmt="mixed_outline", floatfmt=".1%"))
