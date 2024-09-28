"""
    [1] Klambauer, Günter, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. “Self-Normalizing Neural Networks.” arXiv, September 7, 2017. https://doi.org/10.48550/arXiv.1706.02515.
"""

import math
import typing as ty

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules import Module, MSELoss

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel, EncodeEmbeddingFeatures


class CrossLayer(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, x):
        return self.dropout(x0 * self.linear(x)) + x


class DCNv2Base(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        n_hidden_layers: int,
        n_cross_layers: int,
        hidden_dropout: float,
        cross_dropout: float,
        d_out: int,
        stacked: bool,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.first_linear = nn.Linear(d_in, d)
        self.last_linear = nn.Linear(d if stacked else 2 * d, d_out)

        deep_layers = sum(
            [
                [nn.Linear(d, d), nn.ReLU(True), nn.Dropout(hidden_dropout)]
                for _ in range(n_hidden_layers)
            ],
            [],
        )
        cross_layers = [CrossLayer(d, cross_dropout) for _ in range(n_cross_layers)]

        self.deep_layers = nn.Sequential(*deep_layers)
        self.cross_layers = nn.ModuleList(cross_layers)
        self.stacked = stacked

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_linear(x)

        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x, x_cross)

        if self.stacked:
            return self.last_linear(self.deep_layers(x_cross)).squeeze(1)
        else:
            return self.last_linear(
                torch.cat([x_cross, self.deep_layers(x)], dim=1)
            ).squeeze(1)


class DCNv2(BaseModel):

    def __init__(
        self,
        out_dim: int,
        input_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_cross_layers: int,
        hidden_dropout: float,
        cross_dropout: float,
        stacked: bool,
        cardinalities: list[tuple[int, int]],
        d_embedding: int,
        loss: Module = MSELoss(),
        lr: float = 0.001,
        weight_decay: float = 0.01,
        dataset_name: str = None,
        iterations_per_epoch: int = None,
        num_epochs: int = None,
        eta_min: float = 0.0,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
        using_embedding: bool = False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.d = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_cross_layers = n_cross_layers
        self.hidden_dropout = hidden_dropout
        self.cross_dropout = cross_dropout
        self.d_out = hidden_dim
        self.stacked = stacked
        self.categories = [x[1] for x in cardinalities]
        self.d_embedding = d_embedding
        self.cardinalities = cardinalities
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim
        self.using_embedding = using_embedding

        T_max = num_epochs * iterations_per_epoch
        super().__init__(
            head_dimension=hidden_dim,
            out_dim=out_dim,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
            T_max=T_max,
            eta_min=eta_min,
            dataset_name=dataset_name,
        )
        self.print_params()

    def build_encoder(self):
        class EncoderDCNv2(nn.Module):
            def __init__(
                self,
                n_features: int,
                cardinalities: list[tuple[int, int]],
                dcnv2: nn.Module,
                *args,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.n_features = n_features
                self.cardinalities = cardinalities
                self.dcnv2 = dcnv2

            def forward(self, x: torch.Tensor):
                categorical_idx_colums = [x[0] for x in self.cardinalities]
                numerical_idx_colums = [
                    idx
                    for idx in range(self.n_features)
                    if idx not in categorical_idx_colums
                ]

                cat_x = x[:, categorical_idx_colums]
                cat_x = cat_x.long()
                num_x = x[:, numerical_idx_colums]

                return self.dcnv2(num_x, cat_x)

        dcnv2 = DCNv2Base(
            d_in=self.input_dim - len(self.categories),
            d=self.d,
            n_hidden_layers=self.n_hidden_layers,
            n_cross_layers=self.n_cross_layers,
            hidden_dropout=self.hidden_dropout,
            cross_dropout=self.cross_dropout,
            d_out=self.d_out,
            stacked=self.stacked,
            categories=self.categories,
            d_embedding=self.d_embedding,
        )
        encoder = EncoderDCNv2(
            self.input_dim,
            self.cardinalities,
            dcnv2,
        )

        return nn.Sequential(
            (
                EncodeEmbeddingFeatures(
                    input_dim=self.input_dim,
                    emb_dim=self.input_dim,
                    encoder_type=self.encoder_type,
                    input_embed_dim=self.input_embed_dim,
                )
                if self.using_embedding
                else nn.Identity()
            ),
            encoder,
        )

    @staticmethod
    def get_model_args(
        datamodule: pl.LightningDataModule,
        args: ty.OrderedDict,
        model_args: ty.OrderedDict,
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
        model_args.input_dim = datamodule.dataset.D + extra_cls

        model_args.n_cont_features = datamodule.dataset.D

        if not args.using_embedding:
            model_args.n_cont_features -= len(datamodule.dataset.cardinalities)

        if args.using_embedding:
            model_args.cardinalities = []
        else:
            model_args.cardinalities = datamodule.dataset.cardinalities

        model_args.out_dim = (
            len(set(datamodule.dataset.y))
            if args.task_type == TASK_TYPE.MULTI_CLASS
            else 1
        )

        if not args.using_embedding:
            model_args.summary_input = (args.batch_size, model_args.input_dim)
        else:
            model_args.summary_input = (
                args.batch_size,
                model_args.input_dim,
                model_args.input_embed_dim,
            )

        if not hasattr(model_args, "dataset_name"):
            model_args.dataset_name = datamodule.dataset.name

        if not hasattr(model_args, "lr"):
            model_args.lr = args.lr

        if not hasattr(model_args, "weight_decay"):
            model_args.weight_decay = args.weight_decay

        datamodule.setup("train")
        model_args.iterations_per_epoch = len(datamodule.train_dataloader())
        model_args.num_epochs = args.exp_train_total_epochs
        return model_args
