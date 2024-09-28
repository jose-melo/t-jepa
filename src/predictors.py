import numpy as np

from tabulate import tabulate
import torch
import torch.nn as nn

from src.encoder import TabularEncoder

from src.utils.log_utils import _debug_values
from src.utils.train_utils import (
    get_1d_sincos_pos_embed,
    trunc_normal_,
    apply_masks_from_idx,
)


class Predictors(nn.Module):
    def __init__(
        self,
        ## predictor type
        pred_type: str,
        ## model parameters
        hidden_dim: int,
        pred_embed_dim: int,
        num_features: int,
        num_layers: int,
        num_heads: int,
        p_dropout: int,
        layer_norm_eps: float,
        activation: str,
        device: torch.device,
        cardinalities: list,
        pred_dim_feedforward: int = None,
    ):

        super(Predictors, self).__init__()

        self.pred_type = pred_type

        assert self.pred_type in ["mlp", "transformer"], "wrong type of predictor"

        self.hidden_dim = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.pred_embed_dim = pred_embed_dim
        self.num_features = num_features
        self.cardinalities = cardinalities
        self.dim_feedforward = pred_dim_feedforward

        if self.pred_type == "mlp":
            self.predictors = []
            for _ in range(num_features):
                self.predictors.append(
                    MLP(
                        self.hidden_dim * num_features,
                        self.hidden_dim,
                        self.num_layers,
                        self.p_dropout,
                        self.layer_norm_eps,
                        self.activation,
                    ).to(self.device)
                )
        else:
            self.predictors = TransformerPredictor(
                num_features=num_features,
                model_hidden_dim=self.hidden_dim,
                pred_embed_dim=self.pred_embed_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                p_dropout=self.p_dropout,
                layer_norm_eps=self.layer_norm_eps,
                activation=self.activation,
                dim_feedforward=self.dim_feedforward,
            ).to(self.device)

        self.print_params()

    def print_params(self):
        print(
            f"{self.__class__.__name__} {self.__class__.__bases__[0].__name__} parameters:"
        )

        print(
            tabulate(
                [
                    ["hidden_dim", self.hidden_dim],
                    ["num_features", self.num_features],
                    ["num_layers", self.num_layers],
                    ["num_heads", self.num_heads],
                    ["p_dropout", self.p_dropout],
                    ["layer_norm_eps", self.layer_norm_eps],
                    ["activation", self.activation],
                    ["pred_embed_dim", self.pred_embed_dim],
                    ["device", self.device],
                    ["cardinalities", self.cardinalities],
                    ["dim_feedforward", self.dim_feedforward],
                ],
                headers=["Parameter", "Value"],
                tablefmt="pretty",
            )
        )

    def forward(self, x, masks_enc, masks_pred):
        if self.pred_type == "mlp":
            return self.forward_mlp(x, masks_enc, masks_pred)
        else:
            return self.forward_transformer(x, masks_enc, masks_pred)

    def forward_predictor_k(self, x, k):
        return self.predictors[k](x)

    def forward_mlp(self, x, mask_pred):
        out = []
        for mask in mask_pred:
            out_mask = []
            for col_idx in range(self.num_features):
                out_batch = self.forward_predictor_k(x, col_idx)
                out_mask.append(out_batch)
            out_mask = torch.stack(out_mask, dim=1)

            columns_to_keep = torch.ones(mask.shape[-1])
            card_idx = 0
            orig_idx = 0
            idx = 0
            while idx < mask.shape[-1]:
                if orig_idx in [card[0] for card in self.cardinalities]:
                    for _ in range(
                        self.cardinalities[card_idx][1] - 1,
                    ):
                        idx += 1
                        columns_to_keep[idx] = 0
                    card_idx += 1
                idx += 1
                orig_idx += 1
            mask = mask[:, columns_to_keep == 1]

            out_mask = out_mask * mask.unsqueeze(-1)
            out.append(out_mask)
        return out

    def forward_transformer(self, x, masks_enc, masks_pred):
        return self.predictors(x, masks_enc, masks_pred)

    def load_state_dict_(self, list_state_dict):
        if self.pred_type == "mlp":
            for idx, pred in enumerate(self.predictors):
                pred.load_state_dict(list_state_dict[idx])
        else:
            self.predictors.load_state_dict(list_state_dict)

    def state_dict_(self):
        if self.pred_type == "mlp":
            return [pred.state_dict() for pred in self.predictors]
        else:
            return self.predictors.state_dict()


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
    ):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        layers = []
        if self.num_layers > 1:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Dropout(p=p_dropout))
                layers.append(nn.LayerNorm(hidden_dim, eps=layer_norm_eps))
                if activation == "relu":
                    layers.append(nn.ReLU())
                if activation == "gelu":
                    layers.append(nn.GELU())
                if activation == "elu":
                    layers.append(nn.ELU())

            self.fc = nn.Sequential(*layers)
        else:
            self.in_layer = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(p=p_dropout)
            self.layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            if activation == "relu":
                self.activation = nn.ReLU()
            if activation == "gelu":
                self.activation = nn.GELU()
            if activation == "elu":
                self.activation = nn.ELU()

        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        if self.num_layers > 1:
            x = self.fc(x)
        else:
            x = self.in_layer(x)
            x = self.dropout(x)
            x = self.layernorm(x)
            x = self.activation(x)

        out = self.out_layer(x)

        return out


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        num_features: int,
        model_hidden_dim: int,
        pred_embed_dim: int,
        num_layers: int,
        num_heads: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
        init_std: float = 0.02,
        dim_feedforward: int = None,
    ):
        super(TransformerPredictor, self).__init__()

        self.model_hidden_dim = model_hidden_dim
        self.pred_embed_dim = pred_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.num_features = num_features
        self.dim_feedforward = dim_feedforward

        self.predictor_emb = nn.Linear(
            self.model_hidden_dim, self.pred_embed_dim, bias=True
        )
        self.layer_norm = nn.LayerNorm(self.pred_embed_dim)

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_features, self.pred_embed_dim), requires_grad=False
        )
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.pred_embed_dim,
            np.arange(self.num_features),
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.pred_embed_dim))

        self.transformer = TabularEncoder(
            hidden_dim=self.pred_embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            p_dropout=self.p_dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
            dim_feedforward=self.dim_feedforward,
        )

        self.predictor_norm = nn.LayerNorm(self.pred_embed_dim)
        self.predictor_proj = nn.Linear(
            self.pred_embed_dim, self.model_hidden_dim, bias=True
        )

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_enc, masks_pred):

        B = len(x)

        _debug_values(x[0].T, title="Input")
        x = self.predictor_emb(x)

        _debug_values(x[0].T, title="Embedded input")

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)

        _debug_values(x_pos_embed[0].T, title="Positional embedding")

        x_pos_embed = apply_masks_from_idx(x_pos_embed, masks_enc)

        x += x_pos_embed

        _debug_values(x[0].T, title="After adding positional embedding")

        _, N_ctxt, D = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)

        _debug_values(pos_embs[0].T, title="Positional embedding before mask")
        pos_embs = apply_masks_from_idx(pos_embs, masks_pred)

        _debug_values(pos_embs[0].T, title="Positional embedding with mask")

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        _debug_values(pred_tokens[0].T, title="Predictor tokens")

        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        _debug_values(x[0].T, title="Input with predictor tokens")

        x = self.transformer(x)
        x = self.predictor_norm(x)

        _debug_values(x[0].T, title="After transformer")

        x = x[:, N_ctxt:]

        _debug_values(x[0].T, title="After slicing")

        x = self.predictor_proj(x)

        _debug_values(x[0].T, title="After projection")

        return x
