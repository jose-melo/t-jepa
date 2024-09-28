from typing import OrderedDict

import pytorch_lightning as pl
import torch
from torch import nn

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel, EncodeEmbeddingFeatures


class MLP(BaseModel):
    def __init__(
        self,
        input_dim: int = 128,
        n_hidden: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        loss: nn.Module = torch.nn.MSELoss(),
        out_dim: int = 1,
        exp_lr: float = 1e-3,
        exp_weight_decay: float = 0.01,
        exp_eta_min: float = 0.0,
        dataset_name: str = None,
        using_embedding: bool = False,
        iterations_per_epoch: int = None,
        num_epochs: int = None,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
        **kwargs,
    ) -> None:
        self.emb_dim = input_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.out_dim = out_dim
        self.using_embedding = using_embedding
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim
        T_max = num_epochs * iterations_per_epoch
        super(MLP, self).__init__(
            head_dimension=self.hidden_dim,
            out_dim=self.out_dim,
            loss=loss,
            lr=exp_lr,
            weight_decay=exp_weight_decay,
            T_max=T_max,
            eta_min=exp_eta_min,
            dataset_name=dataset_name,
        )
        self.print_params()

    def build_encoder(self):

        hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden - 1):
            hidden_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.BatchNorm1d(self.hidden_dim),
                )
            )
        return nn.Sequential(
            (
                EncodeEmbeddingFeatures(
                    input_dim=self.emb_dim,
                    emb_dim=self.emb_dim,
                    encoder_type=self.encoder_type,
                    input_embed_dim=self.input_embed_dim,
                )
                if self.using_embedding
                else nn.Identity()
            ),
            nn.Linear(self.emb_dim, self.hidden_dim),
            *hidden_layers,
        )

    @staticmethod
    def get_model_args(
        datamodule: pl.LightningDataModule,
        args: OrderedDict,
        model_args: OrderedDict,
        dataset: BaseDataset = None,
        **kwargs,
    ):
        if dataset is None:
            dataset = datamodule.dataset

        if hasattr(dataset, "H"):
            model_args.input_embed_dim = dataset.H
        else:
            model_args.input_embed_dim = None

        extra_cls = 1 if args.using_embedding else 0
        model_args.input_dim = extra_cls + datamodule.dataset.D

        if not args.using_embedding:
            model_args.input_dim += sum(
                [x[1] - 1 for x in datamodule.dataset.cardinalities]
            )

        if args.task_type == TASK_TYPE.MULTI_CLASS:
            model_args.out_dim = len(set(datamodule.dataset.y))
        else:
            model_args.out_dim = 1

        if args.using_embedding:
            model_args.summary_input = (
                args.batch_size,
                model_args.input_dim,
                model_args.input_embed_dim,
            )
        else:
            model_args.summary_input = (args.batch_size, model_args.input_dim)

        if not hasattr(model_args, "dataset_name"):
            model_args.dataset_name = datamodule.dataset.name
        datamodule.setup("train")
        model_args.iterations_per_epoch = len(datamodule.train_dataloader())
        model_args.num_epochs = args.exp_train_total_epochs
        return model_args
