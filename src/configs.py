"""Define argument parser."""

import argparse

from src.benchmark.utils import MODEL_NAME_TO_MODEL_MAP


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def build_parser():
    """Build parser"""

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", str2bool)

    ###########################################################################
    # #### Data Config ########################################################
    ###########################################################################

    parser.add_argument(
        "--mock",
        type=bool,
        default=False,
        help="If True, use a mock dataset for testing purposes.",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        default="./",
        help="Revelevant for distributed training. Name of directory where "
        " the dictionnaries containing the anomaly scores.",
    )
    parser.add_argument("--data_path", type=str, default="data", help="Path of data")
    parser.add_argument(
        "--data_set",
        type=str,
        default="jannis",
        help="accepted values are currently: adult, helena, jannis,aloi, california, higgs",
    )
    parser.add_argument(
        "--test_size_ratio", type=float, default=0.0, help="Size of test set."
    )
    parser.add_argument(
        "--val_size_ratio", type=float, default=0.0, help="Size of validation set."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for train/test split.",
    )
    parser.add_argument(
        "--full_dataset_cuda",
        type="bool",
        default=False,
        help="Put the whole dataset on cuda to accelerate training."
        "Induces a higher memory cost, only to be used for small"
        " medium sized datasets.",
    )
    parser.add_argument("--verbose", type=str, default=False)

    ###########################################################################
    # #### Experiment Config ##################################################
    ###########################################################################
    parser.add_argument(
        "--test",
        type="bool",
        default=False,
        help="If True, run the test function of the model.",
    )
    parser.add_argument(
        "--exp_n_runs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--exp_device",
        default=None,
        type=str,
        help="If provided, use this (CUDA) device for the run.",
    )
    parser.add_argument(
        "--np_seed",
        type=int,
        default=42,
        help="Random seed for numpy. Set to -1 to choose at random.",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=42,
        help="Random seed for torch. Set to -1 to choose at random.",
    )
    parser.add_argument(
        "--random",
        type="bool",
        default=False,
        help="If True, set the random seed to a random value.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of instances (rows) in each batch "
        "taken as input by the model. -1 corresponds to no "
        "minibatching.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=-1,
        help="Number of instances (rows) in each batch ",
    )
    parser.add_argument(
        "--exp_train_total_epochs", type=int, default=100, help="Number of epochs."
    )

    ###########################################################################
    ######## Caching Config ###################################################
    ###########################################################################
    parser.add_argument(
        "--exp_patience",
        type=int,
        default=10,
        help="Early stopping -- number of epochs that "
        "training may not improve before training stops. "
        "Turned off by default.",
    )
    parser.add_argument(
        "--log_tensorboard",
        type="bool",
        default=False,
        help="Whether to load tensorboard items.",
    )
    parser.add_argument(
        "--exp_cadence_type",
        type=str,
        default="improvement",
        choices=["improvement", "recurrent", "None"],
        help="What type of caching to consider. If 'improvement'"
        ",caching every exp_cache_cadence times that train loss"
        "improved. If 'recurrent' caching every exp_cache_cadence"
        "epochs.",
    )
    parser.add_argument(
        "--exp_cache_cadence",
        type=int,
        default=1,
        help="Checkpointing -- we cache the model every `exp_cache_cadence` "
        "Set this value to -1 to disable caching.",
    )
    parser.add_argument(
        "--exp_val_cache_cadence",
        type=int,
        default=1,
        help="Keep track of val score, cache every `exp_val_cache_cadence` "
        "Set this value to -1 to disable caching.",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="kaiming",
        help="Initialization type for the model. "
        "Options: 'kaiming', 'xavier', 'normal', 'uniform'.",
    )

    ###########################################################################
    # #### Optimization Config ################################################
    ###########################################################################

    parser.add_argument(
        "--exp_scheduler", type="bool", default=True, help="LR scheduler (from I-JEPA)"
    )
    parser.add_argument(
        "--exp_weight_decay_scheduler",
        type="bool",
        default=True,
        help="Weight decay scheduler (from I-JEPA)",
    )
    parser.add_argument(
        "--exp_start_lr", type=float, default=1e-3, help="Starting learning rate."
    )
    parser.add_argument(
        "--exp_lr", type=float, default=1e-4, help="Reference learning rate."
    )
    parser.add_argument(
        "--exp_final_lr", type=float, default=0, help="Reference learning rate."
    )
    parser.add_argument(
        "--exp_warmup", type=int, default=5, help="Number of warm up epochs"
    )
    parser.add_argument(
        "--exp_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay / L2 regularization penalty. Included in this "
        "section because it is set in the optimizer. "
        "HuggingFace default: 1e-5",
    )
    parser.add_argument(
        "--exp_final_weight_decay",
        type=float,
        default=1e-2,
        help="Final Weight decay / L2 regularization penalty.",
    )
    parser.add_argument(
        "--exp_ipe_scale",
        type=float,
        default=1.0,
        help="scheduler scale factor (def: 1.0)",
    )
    parser.add_argument(
        "--exp_gradient_clipping",
        type=float,
        default=1.0,
        help="If > 0, clip gradients.",
    )

    ###########################################################################
    # #### Multiprocess Config ################################################
    ###########################################################################

    parser.add_argument(
        "--mp_distributed",
        dest="mp_distributed",
        default=False,
        type="bool",
        help="If True, run data-parallel distributed training with Torch DDP.",
    )
    parser.add_argument(
        "--mp_nodes",
        dest="mp_nodes",
        default=1,
        type=int,
        help="number of data loading workers",
    )
    parser.add_argument(
        "--mp_gpus", dest="mp_gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "--mp_nr", dest="mp_nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument("--pin_memory", type="bool", default=True)

    ###########################################################################
    # #### General Model Config ###############################################
    ###########################################################################

    parser.add_argument(
        "--using_embedding",
        type="bool",
        default=False,
        help="Whether to use embedding for the model.",
    )
    parser.add_argument(
        "--model_dtype",
        default="float32",
        type=str,
        help="Data type (supported for float32, float64) " "used for model.",
    )
    parser.add_argument(
        "--data_loader_nprocs",
        type=int,
        default=0,
        help="Number of processes to use in data loading. Specify -1 to use "
        "all CPUs for data loading. 0 (default) means only the main  "
        "process is used in data loading. Applies for serial and "
        "distributed training.",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type="bool",
        default=False,
        help="Whether to load from last saved checkpoints.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to model to load. Must not be None if "
        "load_from_checkpoint is True.",
    )
    parser.add_argument(
        "--model_amp",
        default=False,
        type="bool",
        help="If True, use automatic mixed precision (AMP), "
        "which can provide significant speedups on V100/"
        "A100 GPUs.",
    )
    parser.add_argument(
        "--model_feature_type_embedding",
        type="bool",
        default=True,
        help="When True, learn an embedding on whether each feature is "
        'numerical or categorical. Similar to the "token type" '
        "embeddings canonical in NLP. See https://github.com/huggingface/"
        "transformers/blob/master/src/transformers/models/bert/"
        "modeling_bert.py",
    )
    parser.add_argument(
        "--model_feature_index_embedding",
        type="bool",
        default=True,
        help="When True, learn an embedding on the index of each feature "
        "(column). Similar to positional embeddings.",
    )
    parser.add_argument(
        "--model_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout probability for all fully connected layers in the "
        "(in, but not out) embeddings, attention blocks.",
    )
    parser.add_argument(
        "--model_layer_norm_eps", type=float, default=1.0e-5, help="Layer norm param."
    )
    parser.add_argument(
        "--model_dim_hidden",
        type=int,
        default=64,
        help="Intermediate feature dimension.",
    )
    parser.add_argument(
        "--model_dim_feedforward",
        type=int,
        default=256,
        help="Intermediate feature dimension in the feedforward layer.",
    )
    parser.add_argument(
        "--model_num_heads",
        type=int,
        default=8,
        help="Number of attention heads. Must evenly divide model_dim_hidden.",
    )
    parser.add_argument(
        "--model_num_layers",
        type=int,
        default=4,
        help="Number of layers in transformer encoder.",
    )
    parser.add_argument(
        "--model_ema_end",
        type=float,
        default=1,
        help="Argument for moving average of model weights between context and"
        "target encoders.",
    )
    parser.add_argument(
        "--model_ema_start",
        type=float,
        default=0.996,
        help="Argument for moving average of model weights between context and"
        "target encoders.",
    )
    parser.add_argument(
        "--model_act_func",
        type=str,
        default="relu",
        choices=["relu", "gelu", "elu"],
        help="Activation function in the model.",
    )
    parser.add_argument(
        "--probe_cadence",
        type=int,
        default=0,
        help="Probing cadence.",
    )
    parser.add_argument(
        "--probe_model",
        type=str,
        default="mlp",
        choices=list(MODEL_NAME_TO_MODEL_MAP.keys()),
        help="Model to use for probing.",
    )
    parser.add_argument(
        "--n_cls_tokens",
        type=int,
        default=1,
        help="Number of [CLS] tokens.",
    )

    ###########################################################################
    # #### General Pred Config ################################################
    ###########################################################################

    parser.add_argument(
        "--pred_type",
        type=str,
        default="transformer",
        choices=["mlp", "transformer"],
        help="Predictor type.",
    )
    parser.add_argument(
        "--pred_num_layers",
        type=int,
        default=2,
        help="Number of layer in the predictor",
    )
    parser.add_argument(
        "--pred_embed_dim",
        type=int,
        default=64,
        help="Predictor embedding dimension. Shoudl be lower"
        "or equal to model_dim_hidden.",
    )
    parser.add_argument(
        "--pred_num_heads",
        type=int,
        default=4,
        help="Number of attention heads in the transformer predictor",
    )
    parser.add_argument(
        "--pred_p_dropout", type=float, default=0.1, help="Predictor dropout probabilty"
    )
    parser.add_argument(
        "--pred_layer_norm_eps",
        type=float,
        default=1e-5,
        help="Layer norm coefficient for the predictors",
    )
    parser.add_argument(
        "--pred_activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "elu"],
        help="Activation function in the predictor.",
    )
    parser.add_argument(
        "--pred_dim_feedforward",
        type=int,
        default=256,
        help="Intermediate feature dimension in the feedforward layer.",
    )

    ###########################################################################
    # #### Masking Config #####################################################
    ###########################################################################

    parser.add_argument(
        "--mask_allow_overlap",
        type="bool",
        default=False,
        help="Whether to allow overlap between context rep and target.",
    )
    parser.add_argument(
        "--mask_min_ctx_share",
        type=float,
        default=0.2,
        help="Minimum share of features to be unmasked for context.",
    )
    parser.add_argument(
        "--mask_max_ctx_share",
        type=float,
        default=0.4,
        help="Maximum share of features to be unmasked for context.",
    )
    parser.add_argument(
        "--mask_min_trgt_share",
        type=float,
        default=0.2,
        help="Minimum share of features to be unmasked for target.",
    )
    parser.add_argument(
        "--mask_max_trgt_share",
        type=float,
        default=0.4,
        help="Maximum share of features to be unmasked for target.",
    )
    parser.add_argument(
        "--mask_num_preds",
        type=int,
        default=4,
        help="Number of predictions for each context.",
    )
    parser.add_argument(
        "--mask_num_encs",
        type=int,
        default=1,
        help="Number of context mask per sample.",
    )

    return parser
