from typing import List, Optional

import openml
from lightning.pytorch import LightningDataModule
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
        self.dataset: Optional[ProjectDataset] = None
        self.dataset_train: Optional[ProjectDataset] = None
        self.dataset_test: Optional[ProjectDataset] = None
        self.dataset_val: Optional[ProjectDataset] = None

    def prepare_data(self):
        # Download dataset if needed
        openml.datasets.get_dataset(self.dataset_name)

    def setup(self, stage: Optional[str]) -> None:
        # Load the dataset from cache as numpy array
        numpy_data, _, _, _ = openml.datasets.get_dataset(self.dataset_name).get_data(dataset_format='array')
        data = numpy_data[:, 0:-1]
        labels = numpy_data[:, -1].astype(int)

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
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, shuffle=False, batch_size=self.batch_size)
