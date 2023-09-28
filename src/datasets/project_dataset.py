import logging
import os
from math import ceil
from typing import List, Optional

import numpy as np
import openml
from lightning.pytorch import LightningDataModule
from lightning.pytorch.accelerators import CUDAAccelerator
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, random_split


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
        self._nb_workers: int = 0  # Can change later depending on the accelerator

        # Objects to store the different datasets instances
        self.dataset: Optional[ProjectDataset] = None
        self.dataset_train: Optional[ProjectDataset] = None
        self.dataset_test: Optional[ProjectDataset] = None
        self.dataset_val: Optional[ProjectDataset] = None

    def prepare_data(self):
        # Download dataset if no cached
        openml.datasets.get_dataset(self.dataset_name,
                                    download_data=False,
                                    download_qualities=False,
                                    download_features_meta_data=False)

    def setup(self, stage: Optional[str]) -> None:
        # Get the number of workers to use for the data loading
        self.set_auto_nb_workers()

        # Load the dataset from cache as a pandas dataframe
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
        self.dataset = ProjectDataset(data, labels)

        # Split into subsets
        nb = len(self.dataset)
        # FIXME more robust splitting
        # TODO [template] splitting ratio from setting
        self.dataset_train, self.dataset_test, self.dataset_val = random_split(
            self.dataset, [int(0.7 * nb), int(0.2 * nb), int(0.1 * nb)]
        )

        # TODO [template] log the dataset sizes

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size, num_workers=self._nb_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, shuffle=False, batch_size=self.batch_size, num_workers=self._nb_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, shuffle=False, batch_size=self.batch_size, num_workers=self._nb_workers)

    def set_auto_nb_workers(self) -> None:
        """
        Automatically search for the optimal number of workers (processes that load the data), depending on the
        accelerator type and the number of available CPU.
        """
        if self.trainer is None:
            raise ValueError('The trainer must be linked to the data module before setting the number of workers.')

        accelerator = self.trainer.accelerator

        if isinstance(accelerator, CUDAAccelerator):
            # CUDA doesn't support multithreading for data loading
            self._nb_workers = 0
        else:
            # Try to detect the number of available CPU
            # noinspection PyBroadException
            try:
                nb_workers = len(os.sched_getaffinity(0))
            except Exception:
                nb_workers = os.cpu_count()

            self._nb_workers = ceil(nb_workers / 2)  # The optimal number seems to be half of the cores

        logging.debug(f'Number of workers for data loading: {self._nb_workers}')
