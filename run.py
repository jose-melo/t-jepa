import os, copy
import random


import numpy as np
from tabulate import tabulate
import wandb
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.utils.encode_utils import encode_data
import src.utils.idr_torch as idr_torch  # JEAN-ZAY

from src.encoder import Encoder
from src.predictors import Predictors
from src.torch_dataset import TorchDataset
from src.train import Trainer
from src.mask import MaskCollator
from src.configs import build_parser
from src.utils.log_utils import make_job_name
from src.utils.log_utils import print_args
from src.utils.checkpointer import EarlyStopCounter
from src.utils.train_utils import init_weights
from src.utils.optim_utils import init_optim

from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP


def main(args):

    if args.mp_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.size,
            rank=idr_torch.rank,
        )

    ema_start = args.model_ema_start
    ema_end = args.model_ema_end
    num_epochs = args.exp_train_total_epochs
    ipe_scale = args.exp_ipe_scale

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
    args.is_batchlearning = args.batch_size != -1
    args.iteration = 0
    start_epoch = 0
    if args.test:
        args.mock = True

    if (not args.mp_distributed) or (args.mp_distributed and idr_torch.local_rank == 0):
        if args.verbose:
            print_args(args)

    if args.random:
        args.torch_seed = np.random.randint(0, 100000)
        args.np_seed = np.random.randint(0, 100000)

    torch.manual_seed(args.torch_seed)
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)

    jobname = make_job_name(args)

    print(tabulate(vars(args).items(), tablefmt="fancy_grid"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dataset.load()
    args.test_size = 0
    train_torchdataset = TorchDataset(
        dataset=dataset,
        mode="train",
        kwargs=args,
        device=device,
        preprocessing=encode_data,
    )

    context_encoder = Encoder(
        idx_num_features=dataset.num_features,
        cardinalities=dataset.cardinalities,
        hidden_dim=args.model_dim_hidden,
        num_layers=args.model_num_layers,
        num_heads=args.model_num_heads,
        p_dropout=args.model_dropout_prob,
        layer_norm_eps=args.model_layer_norm_eps,
        gradient_clipping=args.exp_gradient_clipping,
        feature_type_embedding=args.model_feature_type_embedding,
        feature_index_embedding=args.model_feature_index_embedding,
        dim_feedforward=args.model_dim_feedforward,
        device=device,
        args=args,
    )

    predictors = Predictors(
        pred_type=args.pred_type,
        hidden_dim=args.model_dim_hidden,
        pred_embed_dim=args.pred_embed_dim,
        num_features=dataset.D,
        num_layers=args.pred_num_layers,
        num_heads=args.pred_num_heads,
        p_dropout=args.pred_p_dropout,
        layer_norm_eps=args.pred_layer_norm_eps,
        activation=args.pred_activation,
        device=device,
        cardinalities=dataset.cardinalities,
        pred_dim_feedforward=args.pred_dim_feedforward,
    )

    for m in context_encoder.modules():
        init_weights(m, init_type=args.init_type)

    if args.pred_type == "mlp":
        for pred in predictors.predictors:
            for m in pred.modules():
                init_weights(m, init_type=args.init_type)

    target_encoder = copy.deepcopy(context_encoder)

    context_encoder.to(device)
    target_encoder.to(device)
    predictors.to(device)

    scaler = GradScaler(enabled=args.model_amp)
    if args.model_amp:
        print(f"Initialized gradient scaler for Automatic Mixed Precision.")

    early_stop_counter = EarlyStopCounter(
        args, jobname, args.data_set, device=device, is_distributed=False
    )

    mask_collator = MaskCollator(
        args.mask_allow_overlap,
        args.mask_min_ctx_share,
        args.mask_max_ctx_share,
        args.mask_min_trgt_share,
        args.mask_max_trgt_share,
        args.mask_num_preds,
        args.mask_num_encs,
        dataset.D,
        dataset.cardinalities,
    )

    dataloader = DataLoader(
        dataset=train_torchdataset,
        batch_size=args.batch_size,
        num_workers=args.data_loader_nprocs,
        collate_fn=mask_collator,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    ipe = len(dataloader)

    (optimizer, scheduler, weightdecay_scheduler) = init_optim(
        context_encoder,
        predictors,
        ipe,
        args.exp_start_lr,
        args.exp_lr,
        args.exp_warmup,
        args.exp_train_total_epochs,
        args.exp_weight_decay,
        args.exp_final_weight_decay,
        args.exp_final_lr,
        args.exp_ipe_scale,
        args.exp_scheduler,
        args.exp_weight_decay_scheduler,
    )

    momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    if args.load_from_checkpoint:
        if os.path.isfile(args.load_path):
            (
                context_encoder,
                predictors,
                target_encoder,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
                start_epoch,
            ) = early_stop_counter.load_model(
                load_pth=args.load_path,
                context_encoder=context_encoder,
                predictor=predictors,
                target_encoder=target_encoder,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                weightdecay_scheduler=weightdecay_scheduler,
            )
            for _ in range(start_epoch * ipe):
                next(momentum_scheduler)
                mask_collator.step()
        else:
            print(
                "Tried loading from checkpoint,"
                " but provided path does not exist."
                " Starting training from scratch."
            )

        for p in target_encoder.parameters():
            p.requires_grad = False

    trainer = Trainer(
        args=args,
        start_epoch=start_epoch,
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        scheduler=scheduler,
        weightdecay_scheduler=weightdecay_scheduler,
        early_stop_counter=early_stop_counter,
        momentum_scheduler=momentum_scheduler,
        optimizer=optimizer,
        scaler=scaler,
        torch_dataset=train_torchdataset,
        dataloader=dataloader,
        distributed_args=None,
        device=device,
        probe_cadence=args.probe_cadence,
        probe_model=args.probe_model,
    )

    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    wandb.init(mode="offline")
    parser = build_parser()
    args = parser.parse_args()
    main(args)
