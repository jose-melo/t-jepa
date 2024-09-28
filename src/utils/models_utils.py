from abc import abstractmethod
from dataclasses import dataclass
from typing import OrderedDict

import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tabulate
import torch
from scipy.special import softmax
from sklearn.calibration import expit
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
from torch import nn
from numpy import floor

from src.datasets.base import BaseDataset


@dataclass
class TASK_TYPE:
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"
    BINARY_CLASS = "binary_class"


class BaseModel(pl.LightningModule):

    def __init__(
        self,
        head_dimension: int,
        out_dim: int,
        loss: nn.Module = torch.nn.MSELoss(),
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        T_max: int = 10,
        eta_min: float = 0.0,
        dataset_name: str = None,
    ):
        super(BaseModel, self).__init__()
        self.encoder = None
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.eta_min = eta_min
        self.dataset_name = dataset_name

        self.head = nn.Linear(head_dimension, out_dim)
        self.encoder = self.build_encoder()

        self.task_type = self.get_task_type()

    def print_params(self):
        print("Model Parameters")

        param_dict = {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, (int, float, str, tuple, list))
        }
        print(
            tabulate.tabulate(
                param_dict.items(), headers=["Param", "Value"], tablefmt="fancy_grid"
            )
        )

    @abstractmethod
    def build_encoder(self):
        pass

    def forward(self, x):
        x = self.encoder(x)
        out = self.head(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(f"{self.dataset_name}_train_loss", loss)
        self.calculate_metrics(y, y_hat, mode="train")
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(f"{self.dataset_name}_val_loss", loss)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.calculate_metrics(y, y_hat, mode="val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(f"{self.dataset_name}_test_loss", loss)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.calculate_metrics(y, y_hat, mode="test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "monitor": "val_loss",
            },
        }

    def get_task_type(self):
        if type(self.loss) == nn.MSELoss:
            return TASK_TYPE.REGRESSION
        if type(self.loss) == nn.BCEWithLogitsLoss:
            return TASK_TYPE.BINARY_CLASS
        if type(self.loss) == nn.CrossEntropyLoss:
            return TASK_TYPE.MULTI_CLASS

    def calculate_metrics(
        self,
        y: torch.Tensor,
        prediction: torch.Tensor,
        classification_mode: str = "logits",
        mode: str = "train",
    ) -> dict[str, float]:
        y = y.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()

        if self.task_type == TASK_TYPE.REGRESSION:
            rmse = mean_squared_error(y, prediction) ** 0.5
            self.log(f"{self.dataset_name}_{mode}_rmse", rmse)
            self.log(f"{self.dataset_name}_{mode}_score", -rmse)
            return {"rmse": rmse, "score": -rmse}
        else:
            labels = None
            if classification_mode == "probs":
                probs = prediction
            elif classification_mode == "logits":
                probs = (
                    expit(prediction)
                    if self.task_type == TASK_TYPE.BINARY_CLASS
                    else softmax(prediction, axis=1)
                )
            else:
                assert classification_mode == "labels"
                probs = None
                labels = prediction
            if labels is None:
                labels = (
                    np.round(probs).astype("int64")
                    if self.task_type == TASK_TYPE.BINARY_CLASS
                    else probs.argmax(axis=1)
                )

            if mode == "train":
                result = {
                    "accuracy": (labels == y).mean(),
                }

            else:
                result = classification_report(
                    y, labels, output_dict=True, zero_division=0
                )
                if self.task_type == TASK_TYPE.BINARY_CLASS:
                    if len(np.unique(y)) == 1:
                        result["roc_auc"] = 0
                    else:
                        result["roc_auc"] = roc_auc_score(y, probs)

        result["score"] = result["accuracy"]
        for label in result:
            if type(result[label]) == float:
                self.log(f"{self.dataset_name}_{mode}_{label}", result[label])

            if type(result[label]) == dict:
                if label in ["macro avg", "micro avg"]:
                    continue

                if label in ["weighted avg"]:
                    for k, v in result[label].items():
                        self.log(f"{self.dataset_name}_{mode}_{k}", v)

        return result

    @staticmethod
    def preprocessing(
        data: np.array,
        dataset: BaseDataset,
        args: OrderedDict,
        **kwargs,
    ):
        train_encoded_dataset = data
        if not args.using_embedding:
            train_encoded_dataset = []
            categorical_idx = [card[0] for card in dataset.cardinalities]
            for col_index in range(dataset.D):
                train_col = data[:, col_index].reshape(-1, 1)
                if col_index in categorical_idx:
                    fitted_encoder = OneHotEncoder(sparse_output=False).fit(train_col)
                else:
                    fitted_encoder = MinMaxScaler().fit(train_col)
                encoded_train_col = fitted_encoder.transform(train_col).astype(
                    np.float32
                )
                train_encoded_dataset.append(
                    np.array(encoded_train_col).astype(np.float32)
                )

            train_encoded_dataset = [torch.from_numpy(x) for x in train_encoded_dataset]

            train_encoded_dataset = torch.cat(train_encoded_dataset, dim=1).float()

        return train_encoded_dataset


class EncodeEmbeddingFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
    ):
        super(EncodeEmbeddingFeatures, self).__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim

        if self.encoder_type == "conv":
            final_h = int(floor(floor(input_dim / 2 - 1) / 2 - 1))
            final_w = int(floor(floor(input_embed_dim / 2 - 1) / 2 - 1))

            if final_h < 1 or final_w < 1:
                print(
                    "Warning: input_dim too small for conv encoder. Using linear flatten instead"
                )
                self.encoder_type = "linear_flatten"
                self.encode = nn.Linear(
                    self.input_embed_dim * self.input_dim, self.emb_dim
                )
            else:
                self.encode = nn.Sequential(
                    nn.Conv2d(1, 3, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.BatchNorm2d(3),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Conv2d(3, 1, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.BatchNorm2d(1),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Flatten(),
                    nn.Linear(final_w * final_h, self.emb_dim),
                )
        elif self.encoder_type == "linear_flatten":
            self.encode = nn.Linear(self.input_embed_dim * self.input_dim, self.emb_dim)
        elif self.encoder_type == "linear_per_feature":
            self.feature_embs = nn.ModuleList()
            for _ in range(self.emb_dim):
                self.feature_embs.append(nn.Linear(self.input_embed_dim, 1))
        elif self.encoder_type == "max_pool":
            self.encode = nn.MaxPool1d(self.input_embed_dim)
        elif self.encoder_type == "mean_pool":
            self.encode = nn.AvgPool1d(self.input_embed_dim)
        else:
            raise ValueError(
                f"Encoder type {self.encoder_type} not supported. Use 'conv', 'linear_flatten' or 'linear_per_feature'"
            )

    def forward(self, x: torch.Tensor):
        if self.encoder_type == "linear_per_feature":
            x_new = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
            for i in range(x.shape[1]):
                x_embedded = self.feature_embs[i](x[:, i, :]).squeeze()
                x_new[:, i] = x_embedded
            return x_new
        elif self.encoder_type == "linear_flatten":
            x = x.view(x.shape[0], -1)
        elif self.encoder_type == "conv":
            x = x.unsqueeze(1)

        x = self.encode(x)

        if self.encoder_type in ["max_pool", "mean_pool"]:
            x = x.squeeze()

        return x
