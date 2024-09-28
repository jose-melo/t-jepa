from argparse import Namespace

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE
from src.utils.train_utils import MOCK_SIZE


def drop_single_sample_collate_fn(batch):
    if len(batch) == 1:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: BaseDataset,
        test_size_ratio: float = 0.2,
        val_size_ratio: float = 0.2,
        random_state: int = 42,
        device: str = "cpu",
        batch_size: int = 64,
        workers: int = 16,
        pin_memory: bool = False,
        full_dataset_cuda: bool = False,
        preprocessing: nn.Module = None,
        mock: bool = False,
        using_embedding: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.args = {
            "batch_size": batch_size,
            "data_loader_nprocs": workers,
            "pin_memory": pin_memory,
            "full_dataset_cuda": full_dataset_cuda,
            "test_size_ratio": test_size_ratio,
            "val_size_ratio": val_size_ratio,
            "random_state": random_state,
            "mock": mock,
            "using_embedding": using_embedding,
        }
        self.args = Namespace(**self.args)
        self.preprocessing = preprocessing
        self.loaded = False

    def setup(self, stage=None):
        if self.loaded:
            return
        self.dataset.load()

        self.train = TorchDataset(
            dataset=self.dataset,
            mode="train",
            kwargs=self.args,
            device=self.device,
            preprocessing=self.preprocessing,
        )
        self.test = TorchDataset(
            dataset=self.dataset,
            mode="test",
            kwargs=self.args,
            device=self.device,
            preprocessing=self.preprocessing,
        )
        self.val = TorchDataset(
            dataset=self.dataset,
            mode="val",
            kwargs=self.args,
            device=self.device,
            preprocessing=self.preprocessing,
        )
        self.loaded = True

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.data_loader_nprocs,
            pin_memory=self.args.pin_memory,
            collate_fn=drop_single_sample_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.data_loader_nprocs,
            pin_memory=self.args.pin_memory,
            collate_fn=drop_single_sample_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.data_loader_nprocs,
            pin_memory=self.args.pin_memory,
            collate_fn=drop_single_sample_collate_fn,
        )


class TorchDataset(Dataset):

    def __init__(
        self,
        dataset: BaseDataset,
        mode: str,
        kwargs,
        device,
        preprocessing: nn.Module = None,
    ):

        X = dataset.X
        Y = dataset.y
        self.task_type = dataset.task_type
        self.D = dataset.D

        X = preprocessing(X, dataset, kwargs) if preprocessing is not None else X

        if self.task_type == TASK_TYPE.REGRESSION:
            std_scaler = StandardScaler()
            std_scaler.fit(Y.reshape(-1, 1))
            Y = std_scaler.transform(Y.reshape(-1, 1)).reshape(-1)

        if kwargs.mock:
            X = X[:MOCK_SIZE]
            Y = Y[:MOCK_SIZE]

        if kwargs.test_size_ratio != 0:
            test_size = int(X.shape[0] * kwargs.test_size_ratio)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                Y,
                test_size=test_size,
                random_state=kwargs.random_state,
            )
            if kwargs.val_size_ratio != 0:
                val_size = int(X.shape[0] * kwargs.val_size_ratio)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=val_size,
                    random_state=kwargs.random_state,
                )
            else:
                X_val = X_test
                y_val = y_test
        else:
            X_train, X_test, X_val = X, X, X
            y_train, y_test, y_val = Y, Y, Y

        self.n_features = dataset.D
        self.n_encoded_features = (
            sum([x[1] - 1 for x in dataset.cardinalities]) + dataset.D
        )
        self.dataset_name = dataset.name
        self.cardinalities = dataset.cardinalities
        self.num_or_cat = dataset.num_or_cat

        self.device = device
        self._mode = mode
        self.full_dataset_cuda = kwargs.full_dataset_cuda
        self.batch_learning = kwargs.batch_size != -1 or kwargs.val_batch_size != -1

        if self._mode == "train":
            self.X_train = X_train
            if (
                dataset.task_type == TASK_TYPE.BINARY_CLASS
                or dataset.task_type == TASK_TYPE.REGRESSION
            ):
                self.Y_train = y_train.astype("float32").reshape(-1, 1)
            elif dataset.task_type == TASK_TYPE.MULTI_CLASS:
                self.Y_train = y_train.astype("int")
            else:
                raise ValueError(f"Task type {dataset.task_type} not supported.")

        elif self._mode == "test":
            self.X_test = X_test
            if (
                dataset.task_type == TASK_TYPE.BINARY_CLASS
                or dataset.task_type == TASK_TYPE.REGRESSION
            ):
                self.Y_test = y_test.astype("float32").reshape(-1, 1)
            elif dataset.task_type == TASK_TYPE.MULTI_CLASS:
                self.Y_test = y_test.astype("int")
            else:
                raise ValueError(f"Task type {dataset.task_type} not supported.")
        elif self._mode == "val":
            self.X_val = X_val
            if (
                dataset.task_type == TASK_TYPE.BINARY_CLASS
                or dataset.task_type == TASK_TYPE.REGRESSION
            ):
                self.Y_val = y_val.astype("float32").reshape(-1, 1)
            elif dataset.task_type == TASK_TYPE.MULTI_CLASS:
                self.Y_val = y_val.astype("int")
            else:
                raise ValueError(f"Task type {dataset.task_type} not supported.")

    def __len__(
        self,
    ):
        if self._mode == "train":
            return self.X_train.shape[0]
        elif self._mode == "test":
            return self.X_test.shape[0]
        elif self._mode == "val":
            return self.X_val.shape[0]

    def __getitem__(self, index):

        if self._mode == "train":
            return self.X_train[index], self.Y_train[index]
        elif self._mode == "test":
            return self.X_test[index], self.Y_test[index]
        elif self._mode == "val":
            return self.X_val[index], self.Y_val[index]

    def set_mode(self, mode):
        assert mode in ["train", "test", "val"]
        self._mode = mode
        if self.full_dataset_cuda or not self.batch_learning:
            self.put_on_device(mode)

    def put_on_device(self, mode):
        if self._mode == "train":
            self.X_train = [x.to(self.device) for x in self.X_train]
        elif self._mode == "test":
            self.X_test = [x.to(self.device) for x in self.X_test]
        elif self._mode == "val":
            self.X_val = [x.to(self.device) for x in self.X_val]
