from typing import OrderedDict

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel
import torch
import torch.nn as nn
import pytorch_lightning as pl


class LinearProbe(BaseModel):

    def __init__(
        self,
        loss: nn.Module = torch.nn.MSELoss(),
        input_dim: int = 128,
        out_dim: int = 1,
        exp_lr: float = 1e-3,
        exp_weight_decay: float = 0.01,
        exp_eta_min: float = 0.0,
        dataset_name: str = None,
        iterations_per_epoch: int = None,
        num_epochs: int = None,
        input_embed_dim: int = 64,
        encoder_type: str = "conv",
        using_embedding: bool = False,
        **kwargs,
    ) -> None:
        self.encoder_type = encoder_type
        self.emb_dim = input_dim
        self.using_embedding = using_embedding
        self.input_embed_dim = input_embed_dim
        self.out_dim = out_dim
        self.input_dim = input_dim

        super(LinearProbe, self).__init__(
            head_dimension=input_embed_dim * input_dim,
            out_dim=out_dim,
            loss=loss,
            lr=exp_lr,
            weight_decay=exp_weight_decay,
            T_max=num_epochs * iterations_per_epoch,
            eta_min=exp_eta_min,
            dataset_name=dataset_name,
        )
        self.print_params()

    def build_encoder(self):
        class FlattenedIdentity(nn.Module):
            def forward(self, x):
                return x.view(x.size(0), -1)

        return FlattenedIdentity()

    @staticmethod
    def get_model_args(
        datamodule: pl.LightningDataModule,
        args: OrderedDict,
        model_args: OrderedDict,
        dataset: BaseDataset = None,
        **kwargs,
    ) -> dict:

        if dataset is None:
            dataset = datamodule.dataset

        if hasattr(dataset, "H"):
            model_args.input_embed_dim = dataset.H
        else:
            model_args.input_embed_dim = None

        extra_cls = args.n_cls_tokens if args.using_embedding else 0
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
