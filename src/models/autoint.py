"""
    [1] Song, Weiping, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 
        “AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.” 
        In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 
        1161–70, 2019. https://doi.org/10.1145/3357384.3357925.
    
    [2] Code inspired by: https://github.com/yandex-research/rtdl-revisiting-models  


"""

import math
import typing as ty

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel, EncodeEmbeddingFeatures


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_latent_tokens: int,
        d_token: int,
    ) -> None:
        super().__init__()
        assert n_latent_tokens == 0
        self.n_latent_tokens = n_latent_tokens
        if d_numerical:
            self.weight = nn.Parameter(Tensor(d_numerical + n_latent_tokens, d_token))
            # The initialization is inspired by nn.Linear
            nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = None
            assert categories is not None
        if categories is None:
            self.category_offsets = None
            self.category_embeddings = None
        else:
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

    @property
    def n_tokens(self) -> int:
        return (0 if self.weight is None else len(self.weight)) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        if x_num is None:
            return self.category_embeddings(x_cat + self.category_offsets[None])  # type: ignore[code]
        x_num = torch.cat(
            [
                torch.ones(len(x_num), self.n_latent_tokens, device=x_num.device),
                x_num,
            ],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]  # type: ignore[code]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],  # type: ignore[code]
                dim=1,
            )
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ["xavier", "kaiming"]

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == "xavier" and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class AutoIntBase(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
    ) -> None:
        assert not prenormalization
        assert activation == "relu"
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        if len(categories) == 0:
            categories = None

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, 0, d_token)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == "xavier":
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == "layerwise"
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    "linear": nn.Linear(d_token, d_token, bias=False),
                }
            )
            if not prenormalization or layer_idx:
                layer["norm0"] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()
                if kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert kv_compression_sharing == "key-value"
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token * n_tokens, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (
                (layer["key_compression"], layer["value_compression"])
                if "key_compression" in layer and "value_compression" in layer
                else (
                    (layer["key_compression"], layer["key_compression"])
                    if "key_compression" in layer
                    else (None, None)
                )
            )
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = layer["linear"](x)
            x = self._end_residual(x, x_residual, layer, 0)
            x = self.activation(x)

        x = x.flatten(1, 2)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


class AutoInt(BaseModel):

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        cardinalities: list[tuple[int, int]],
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        loss: nn.Module = nn.MSELoss(),
        lr: float = 0.001,
        weight_decay: float = 0.01,
        dataset_name: str = None,
        eta_min: float = 0.0,
        iterations_per_epoch: int = None,
        num_epochs: int = None,
        kv_compression: ty.Optional[float] = None,
        kv_compression_sharing: ty.Optional[str] = None,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
        using_embedding: bool = False,
        **kwargs,
    ) -> None:

        self.categories = [x[1] for x in cardinalities]
        self.n_layers = n_layers
        self.d_token = d_token
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.activation = activation
        self.prenormalization = prenormalization
        self.initialization = initialization
        self.kv_compression = kv_compression
        self.kv_compression_sharing = kv_compression_sharing
        self.d_out = d_token
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.cardinalities = cardinalities
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim
        self.using_embedding = using_embedding
        T_max = num_epochs * iterations_per_epoch

        super().__init__(
            head_dimension=self.d_token,
            out_dim=self.out_dim,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
            T_max=T_max,
            eta_min=eta_min,
            dataset_name=dataset_name,
        )
        self.print_params()

    def build_encoder(self):

        class EncoderAutoIntBase(nn.Module):

            def __init__(
                self,
                n_features: int,
                cardinalities: list[tuple[int, int]],
                autoint: nn.Module,
                using_embedding: bool = False,
                *args,
                **kwargs,
            ) -> None:
                super().__init__(*args, **kwargs)
                self.n_features = n_features
                self.cardinalities = cardinalities
                self.autoint = autoint
                self.using_embedding = using_embedding

            def forward(self, x: torch.Tensor):
                if self.using_embedding:
                    return self.autoint(x, None)
                categorical_idx_colums = [x[0] for x in self.cardinalities]
                numerical_idx_colums = [
                    idx
                    for idx in range(self.n_features)
                    if idx not in categorical_idx_colums
                ]
                numerical_idx_colums = torch.tensor(
                    numerical_idx_colums,
                    device=x.device,
                    dtype=int,
                )
                categorical_idx_colums = torch.tensor(
                    categorical_idx_colums,
                    device=x.device,
                    dtype=int,
                )
                cat_x = x[:, categorical_idx_colums]
                cat_x = cat_x.long()
                num_x = x[:, numerical_idx_colums]
                if cat_x.shape[1] == 0:
                    cat_x = None
                return self.autoint(num_x, cat_x)

        d_numerical = self.input_dim - len(self.categories)

        autoint = AutoIntBase(
            d_numerical=d_numerical,
            categories=self.categories,
            n_layers=self.n_layers,
            d_token=self.d_token,
            n_heads=self.n_heads,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            activation=self.activation,
            prenormalization=self.prenormalization,
            initialization=self.initialization,
            kv_compression=self.kv_compression,
            kv_compression_sharing=self.kv_compression_sharing,
            d_out=self.d_out,
        )

        encoder = EncoderAutoIntBase(
            n_features=self.input_dim,
            cardinalities=self.cardinalities,
            autoint=autoint,
            using_embedding=self.using_embedding,
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
