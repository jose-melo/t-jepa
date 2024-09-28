"""
    FT-Transformer for tabular data

    [1] Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. “Revisiting Deep Learning Models for Tabular Data.” arXiv, October 26, 2023. https://doi.org/10.48550/arXiv.2106.11959.

    [2] https://github.com/yandex-research/rtdl-revisiting-models 

"""

from typing import OrderedDict
from rtdl_revisiting_models import FTTransformer as FTTransformerBase
from torch.nn.modules import Module
from torch import nn
import pytorch_lightning as pl

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel, EncodeEmbeddingFeatures


class FTTransformer(BaseModel):

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        n_cont_features: int,
        cardinalities: list[tuple[int, int]],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_d_hidden_multiplier: float,
        ffn_dropout: float,
        residual_dropout: float,
        loss: Module,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        dataset_name: str = None,
        num_epochs: int = None,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
        iterations_per_epoch: int = None,
        using_embedding: bool = False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.n_cont_features = n_cont_features
        cat_cardinalities = [x[1] for x in cardinalities]
        self.cardinalities = cardinalities
        self.cat_cardinalities = cat_cardinalities
        self.d_out = d_block
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim
        self.using_embedding = using_embedding

        T_max = num_epochs * iterations_per_epoch

        super().__init__(
            head_dimension=d_block,
            out_dim=out_dim,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
            T_max=T_max,
            dataset_name=dataset_name,
        )
        self.print_params()

    def build_encoder(self):

        class EncoderFTTransformer(nn.Module):
            def __init__(
                self,
                n_features: int,
                cardinalities: list[tuple[int, int]],
                fttransformer: nn.Module,
                *args,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.fftransformer = fttransformer
                self.n_features = n_features
                self.cardinalities = cardinalities

            def forward(self, x):

                categorical_idx_colums = [x[0] for x in self.cardinalities]
                numerical_idx_colums = [
                    idx
                    for idx in range(self.n_features)
                    if idx not in categorical_idx_colums
                ]

                cat_x = x[:, categorical_idx_colums]
                cat_x = cat_x.long()
                num_x = x[:, numerical_idx_colums]

                if cat_x.shape[1] == 0:
                    cat_x = None

                return self.fftransformer(num_x, cat_x)

        fttransformer = FTTransformerBase(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=self.d_out,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )
        encoder = EncoderFTTransformer(
            n_features=self.input_dim
            - sum(self.cat_cardinalities)
            + len(self.cat_cardinalities),
            cardinalities=self.cardinalities,
            fttransformer=fttransformer,
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

        if args.using_embedding:
            model_args.cardinalities = []
        else:
            model_args.cardinalities = datamodule.dataset.cardinalities

        model_args.n_cont_features = datamodule.dataset.D
        model_args.n_cont_features += extra_cls

        if not args.using_embedding:
            model_args.n_cont_features -= len(datamodule.dataset.cardinalities)

        return model_args
