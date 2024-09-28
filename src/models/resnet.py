"""
    ResNet for tabular data

    [1] Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. “Revisiting Deep Learning Models for Tabular Data.” arXiv, October 26, 2023. https://doi.org/10.48550/arXiv.2106.11959.

    [2] https://github.com/yandex-research/rtdl-revisiting-models 

"""

from typing import OrderedDict

import pytorch_lightning as pl
import torch
from rtdl_revisiting_models import ResNet as ResNetBase
from torch.nn.modules import Module

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel, EncodeEmbeddingFeatures


class ResNet(BaseModel):

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        loss: Module,
        lr: float,
        weight_decay: float,
        n_blocks: int,
        d_out: int,
        d_block: int,
        d_hidden: int,
        d_hidden_multiplier: float,
        dropout1: float,
        dropout2: float,
        dataset_name: str = None,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
        using_embedding: bool = False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.d_out = d_out
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.d_hidden = d_hidden
        self.d_hidden_multiplier = d_hidden_multiplier
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim
        self.using_embedding = using_embedding
        super().__init__(
            head_dimension=d_out,
            out_dim=out_dim,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
            dataset_name=dataset_name,
        )

    def build_encoder(self):
        encoder = ResNetBase(
            d_in=self.input_dim,
            d_out=self.d_out,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            d_hidden=self.d_hidden,
            d_hidden_multiplier=self.d_hidden_multiplier,
            dropout1=self.dropout1,
            dropout2=self.dropout2,
        )
        print("using embedding: ", self.using_embedding)

        return torch.nn.Sequential(
            (
                EncodeEmbeddingFeatures(
                    input_dim=self.input_dim,
                    emb_dim=self.input_dim,
                    encoder_type=self.encoder_type,
                    input_embed_dim=self.input_embed_dim,
                )
                if self.using_embedding
                else torch.nn.Identity()
            ),
            encoder,
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

        model_args.cardinalities = dataset.cardinalities
        model_args.n_cont_features = (
            dataset.D - sum([x[1] - 1 for x in dataset.cardinalities]) + extra_cls
        )

        return model_args
