import math
from typing import Optional

from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
import torch
import torch.nn as nn

from src.tjepa_transformer import TransformerEncoder, TransformerEncoderLayer
from src.utils.encode_utils import torch_cast_to_dtype
from src.utils.log_utils import _debug_values
from src.utils.train_utils import (
    PositionalEncoding,
    apply_masks_from_idx,
)


class Tokenizer(nn.Module):
    category_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: Optional[list[int]],
        d_token: int,
        bias: bool,
        n_cls_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.categories = categories
        self.n_cls_tokens = n_cls_tokens
        if categories is None or len(categories) == 0:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            self.category_embeddings = nn.ModuleList(
                [nn.Embedding(num_categories, d_token) for num_categories in categories]
            )
            for i in range(len(self.category_embeddings)):
                torch.nn.init.kaiming_uniform_(
                    self.category_embeddings[i].weight, a=math.sqrt(5)
                )

        # take [CLS] token into account
        self.weight = nn.Parameter(
            torch.Tensor(d_numerical + self.n_cls_tokens, d_token)
        )
        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None

        # The initialization is inspired by nn.Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            torch.nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:

        x_some = x_num if x_cat is None else x_cat

        assert x_some is not None
        if isinstance(x_some, list):
            batch_size = len(x_some[0])
            device = x_some[0].device
        else:
            batch_size = len(x_some)
            device = x_some.device

        x_num = torch.cat(
            [torch.ones(batch_size, self.n_cls_tokens, device=device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x_cat_embedded = [
                self.category_embeddings[i](x_cat[i])
                for i in range(len(self.categories))
            ]
            x_cat_embedded = torch.stack(x_cat_embedded, dim=1)
            x = torch.cat([x, x_cat_embedded], dim=1)

        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(self.n_cls_tokens, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ## embedding params
        ### num
        idx_num_features: list,
        ### categorical embedding
        cardinalities: list,
        ## model parameters
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        p_dropout: int,
        layer_norm_eps: float,
        gradient_clipping: float,
        feature_type_embedding: bool,
        feature_index_embedding: bool,
        dim_feedforward: int,
        device: torch.device,
        args: dict,
    ):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.idx_num_features = idx_num_features
        self.idx_cat_features = [card[0] for card in cardinalities]
        self.cardinalities = cardinalities
        self.n_input_features = len(idx_num_features) + len(cardinalities)
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps
        self.gradient_clipping = gradient_clipping
        self.dim_feedforward = dim_feedforward
        self.n_cls_tokens = args.n_cls_tokens

        self.print_params()

        self.tokenizer = Tokenizer(
            d_numerical=len(idx_num_features),
            categories=[card[1] for card in cardinalities],
            d_token=hidden_dim,
            bias=True,
            n_cls_tokens=self.n_cls_tokens,
        )

        ## Add feature type embedding
        ## Optionally, we construct "feature type" embeddings:
        ## a representation H is learned based on wether a feature
        ## is numerical of categorical.
        self.feature_type_embedding = feature_type_embedding
        if (
            self.feature_type_embedding
            and self.cardinalities
            and len(idx_num_features) > 0
        ):
            self.feature_types = torch_cast_to_dtype(
                torch.empty(self.n_input_features), "long"
            ).to(self.device)
            for feature_index in range(self.n_input_features):
                if feature_index in self.idx_num_features:
                    self.feature_types[feature_index] = 0
                elif feature_index in self.idx_cat_features:
                    self.feature_types[feature_index] = 1
                else:
                    raise Exception
            self.feature_type_embedding = nn.Embedding(2, self.hidden_dim)
            print(
                f"Using feature type embedding (unique embedding for "
                f"categorical and numerical features)."
            )
        else:
            self.feature_type_embedding = None

        # Feature Index Embedding
        # Optionally, learn a representation based on the index of the column.
        # Allows us to explicitly encode column identity, as opposed to
        # producing this indirectly through the per-column feature embeddings.
        self.feature_index_embedding = feature_index_embedding
        if self.feature_index_embedding:
            self.feature_indices = torch_cast_to_dtype(
                torch.arange(self.n_input_features), "long"
            ).to(self.device)
            self.feature_index_embedding = nn.Embedding(
                self.n_input_features, self.hidden_dim
            )
            print(
                f"Using feature index embedding (unique embedding for " f"each column)."
            )
        else:
            self.feature_index_embedding = None

        self.encoder = TabularEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            p_dropout=p_dropout,
            layer_norm_eps=layer_norm_eps,
            activation=args.model_act_func,
            dim_feedforward=dim_feedforward,
        )
        self.pe = PositionalEncoding(self.hidden_dim)
        self.args = args

        # *** Gradient Clipping ***
        if gradient_clipping:
            clip_value = gradient_clipping
            print(f"Clipping gradients to value {clip_value}.")
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def print_params(self):
        print(
            f"{self.__class__.__name__} {self.__class__.__bases__[0].__name__} parameters:"
        )

        print(
            tabulate(
                [
                    ["hidden_dim", self.hidden_dim],
                    ["idx_num_features", self.idx_num_features],
                    ["idx_cat_features", self.idx_cat_features],
                    ["cardinalities", self.cardinalities],
                    ["n_input_features", self.n_input_features],
                    ["device", self.device],
                    ["num_layers", self.num_layers],
                    ["num_heads", self.num_heads],
                    ["p_dropout", self.p_dropout],
                    ["layer_norm_eps", self.layer_norm_eps],
                    ["gradient_clipping", self.gradient_clipping],
                    ["dim_feedforward", self.dim_feedforward],
                ],
                headers=["Parameter", "Value"],
                tablefmt="pretty",
            )
        )

    def in_embbed_sample(self, x, mask=None):

        x_num = x[:, self.idx_num_features]
        x_cat = x[:, self.idx_cat_features] if len(self.idx_cat_features) > 0 else None

        if x_cat is not None:
            x_cat = x_cat.detach().cpu().numpy()
            categories = [list(range(card[1])) for card in self.cardinalities]
            ohe = OneHotEncoder(sparse_output=False, categories=categories).fit(x_cat)
            x_cat = torch.tensor(ohe.transform(x_cat), device=x_num.device)

            cardinalities = [card[1] for card in self.cardinalities]
            split_indices = torch.tensor([0] + cardinalities).cumsum(0)
            one_hot_features = [
                x_cat[:, split_indices[i] : split_indices[i + 1]]
                for i in range(len(split_indices) - 1)
            ]
            cat_indices = [torch.argmax(feature, dim=1) for feature in one_hot_features]
        else:
            cat_indices = None

        _debug_values(x[0].T, title="Before linear embedding")
        out = self.tokenizer(x_num, cat_indices)

        _debug_values(out[0].T, title="After linear embedding")
        out = self.pe(out)
        _debug_values(out[0].T, title="After positional encoding")

        if self.feature_type_embedding is not None:
            feature_type_embeddings = self.feature_type_embedding(self.feature_types)
            feature_type_embeddings = torch.unsqueeze(feature_type_embeddings, 0)
            feature_type_embeddings = feature_type_embeddings.repeat(out.size(0), 1, 1)
            feature_type_embeddings = torch.cat(
                [
                    torch.zeros(out.size(0), 1, self.hidden_dim).to(self.device),
                    feature_type_embeddings,
                ],
                dim=1,
            )
            out = out + feature_type_embeddings

        if mask is not None:
            out = apply_masks_from_idx(out, mask)
            _debug_values(out[0].T, title="After applying masks")

        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                self.feature_indices
            )

            feature_index_embeddings = torch.unsqueeze(feature_index_embeddings, 0)

            feature_index_embeddings = feature_index_embeddings.repeat(
                out.size(0), 1, 1
            )

            out = out + feature_index_embeddings

        return out

    def forward(self, x, mask=None):
        x = self.in_embbed_sample(x, mask)
        out = self.encoder(x)
        return out


class TabularEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
        dim_feedforward: int,
    ):
        super(TabularEncoder, self).__init__()
        self.transformer_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=p_dropout,
            batch_first=True,
            activation=activation,
        )
        self.transformer = TransformerEncoder(
            self.transformer_layers,
            num_layers,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)

    def forward(self, x):
        _debug_values(x[0].T, title="Before transformer")
        x = self.transformer(x)
        x = self.dropout1(x)
        _debug_values(x[0].T, title="Before layer norm")
        x = self.layernorm1(x)
        _debug_values(x[0].T, title="Before FC")
        x = self.fc(x)
        x = self.dropout2(x)

        _debug_values(x[0].T, title="Before layer norm")
        x = self.layernorm2(x)
        _debug_values(x[0].T, title="After transformer")
        return x
