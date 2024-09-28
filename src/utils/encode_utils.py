from typing import OrderedDict
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.datasets.base import BaseDataset


def encode_data(
    train_data: np.ndarray,
    dataset: BaseDataset,
    args: OrderedDict,
    **kwargs,
):

    num_features = dataset.D
    cardinalities = dataset.cardinalities
    train_encoded_dataset = []
    categorical_idx = [card[0] for card in cardinalities]
    for col_index in range(num_features):
        train_col = train_data[:, col_index].reshape(-1, 1)
        if col_index not in categorical_idx:
            fitted_encoder = MinMaxScaler().fit(train_col)
            encoded_train_col = fitted_encoder.transform(train_col).astype(np.float32)
        else:
            encoded_train_col = train_col
        train_encoded_dataset.append(np.array(encoded_train_col).astype(np.float32))

    train_encoded_dataset = [torch.from_numpy(x) for x in train_encoded_dataset]

    train_encoded_dataset = torch.cat(train_encoded_dataset, dim=1).float()
    return train_encoded_dataset


def torch_cast_to_dtype(obj, dtype_name):
    if dtype_name == "float32":
        obj = obj.float()
    elif dtype_name == "float64":
        obj = obj.double()
    elif dtype_name == "long":
        obj = obj.long()
    else:
        raise NotImplementedError

    return obj


def get_numpy_dtype(dtype_name):
    if dtype_name == "float32":
        dtype = np.float32
    elif dtype_name == "float64":
        dtype = np.float64
    else:
        raise NotImplementedError

    return dtype


def get_torch_dtype(dtype_name):
    if dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "float64":
        dtype = torch.float64
    else:
        raise NotImplementedError

    return dtype
