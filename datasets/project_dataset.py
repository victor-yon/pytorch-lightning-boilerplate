from pathlib import Path
from typing import Optional

import tensorflow_datasets as tfds
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class ProjectDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ProjectDataModule(LightningDataModule):
    def __init__(self, data_file_path: str = 'data/', batch_size: int = 256):
        super().__init__()
        self.data_file_path = Path(data_file_path)
        self.batch_size = batch_size

        # Objects to store the different datasets instances
        self.dataset: Optional[ProjectDataset] = None
        self.dataset_train: Optional[ProjectDataset] = None
        self.dataset_test: Optional[ProjectDataset] = None
        self.dataset_val: Optional[ProjectDataset] = None

        raise NotImplementedError  # TODO implement data loader

    def setup(self, stage: Optional[str]) -> None:
        # TODO load the data
        data, labels = None, None

        # Create the global dataset
        self.dataset = ProjectDataset(data, labels)

        # Split into subsets
        nb = len(self.dataset)
        # FIXME more robust splitting
        # TODO [template] splitting ratio from setting
        self.dataset_train, self.dataset_test, self.dataset_val = random_split(
            self.dataset, [int(0.8 * nb), int(0.1 * nb), int(0.1 * nb)]
        )

        # TODO [template] log the dataset sizes

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, shuffle=False, batch_size=self.batch_size)
