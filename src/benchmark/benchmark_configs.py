"""Define argument parser."""

import argparse
from typing import Optional


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
        "--test",
        type=bool,
        default=False,
        help="If True, run the test suite.",
    )
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
        default="abalone",
        help="accepted values are currently: " "TBD",
    )
    parser.add_argument(
        "--test_size_ratio", type=float, default=0.2, help="Size of test set."
    )
    parser.add_argument(
        "--val_size_ratio", type=float, default=0.2, help="Size of validation set."
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
        type=bool,
        default=False,
        help="If True, set the random seed to a random value.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.json",
        help="Path to output file.",
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
        "--exp_train_total_epochs", type=int, default=100, help="Number of epochs."
    )

    ###########################################################################
    ######## Caching Config ###################################################
    ###########################################################################
    parser.add_argument(
        "--exp_patience",
        type=int,
        default=-1,
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
        default="recurrent",
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
        "--start_lr", type=float, default=1e-3, help="Starting learning rate."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Reference learning rate."
    )
    parser.add_argument(
        "--final_lr", type=float, default=1e-3, help="Reference learning rate."
    )
    parser.add_argument(
        "--T_max", type=int, default=10, help="T_max for the scheduler."
    )
    parser.add_argument(
        "--eta_min", type=float, default=0.0, help="eta_min for the scheduler."
    )
    parser.add_argument(
        "--exp_warmup", type=int, default=1e-5, help="Number of warm up epochs"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay / L2 regularization penalty. Included in this "
        "section because it is set in the optimizer. "
        "HuggingFace default: 1e-5",
    )
    parser.add_argument(
        "--final_weight_decay",
        type=float,
        default=1e-2,
        help="Final Weight decay / L2 regularization penalty.",
    )
    parser.add_argument(
        "--ipe_scale",
        type=float,
        default=1.0,
        help="scheduler scale factor (def: 1.0)",
    )
    parser.add_argument(
        "--gradient_clipping",
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
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use.",
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

    # Subparsers for the different models
    subparsers = parser.add_subparsers()

    ###########################################################################
    # #### MLP Model Config ###################################################
    ###########################################################################

    mlp_parser = subparsers.add_parser("mlp")
    mlp_parser.add_argument(
        "--encoder_type",
        type=str,
        default="conv",
        help="Type of encoder to use.",
    )
    mlp_parser.add_argument(
        "--n_hidden",
        type=int,
        default=4,
        help="Number of hidden layers.",
    )
    mlp_parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Dimension of hidden layers.",
    )
    mlp_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )

    ###########################################################################
    ###### ResNet Model Config ################################################
    ###########################################################################

    resnet_parser = subparsers.add_parser("resnet")

    resnet_parser.add_argument(
        "--encoder_type",
        type=str,
        default="conv",
        help="Type of encoder to use.",
    )
    resnet_parser.add_argument(
        "--d_out",
        type=int,
        default=128,
        help="Dimension of output data.",
    )
    resnet_parser.add_argument(
        "--n_blocks",
        type=int,
        default=8,
        help="Number of blocks.",
    )
    resnet_parser.add_argument(
        "--d_block",
        type=int,
        default=339,
        help="Dimension of blocks.",
    )
    resnet_parser.add_argument(
        "--d_hidden",
        type=int,
        default=None,
        help="Dimension of hidden layers.",
    )
    resnet_parser.add_argument(
        "--d_hidden_multiplier",
        type=float,
        default=None,
        help="Multiplier for hidden layers.",
    )
    resnet_parser.add_argument(
        "--dropout1",
        type=float,
        default=0.15,
        help="Dropout probability 1.",
    )
    resnet_parser.add_argument(
        "--dropout2",
        type=float,
        default=0.0,
        help="Dropout probability 2.",
    )

    ###########################################################################
    # #### FTTransformer Model Config #########################################
    ###########################################################################

    fttransformer_parser = subparsers.add_parser("fttransformer")
    fttransformer_parser.add_argument(
        "--encoder_type",
        type=str,
        default="conv",
        help="Type of encoder to use.",
    )
    fttransformer_parser.add_argument(
        "--d_out",
        type=int,
        default=128,
        help="Dimension of output data.",
    )
    fttransformer_parser.add_argument(
        "--n_blocks",
        type=int,
        default=3,
        help="Number of blocks.",
    )
    fttransformer_parser.add_argument(
        "--d_block",
        type=int,
        default=192,
        help="Dimension of blocks.",
    )
    fttransformer_parser.add_argument(
        "--attention_n_heads",
        type=int,
        default=8,
        help="Number of attention heads.",
    )
    fttransformer_parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.2,
        help="Attention dropout probability.",
    )
    fttransformer_parser.add_argument(
        "--ffn_d_hidden",
        type=int,
        default=None,
        help="Dimension of hidden layers.",
    )
    fttransformer_parser.add_argument(
        "--ffn_d_hidden_multiplier",
        type=float,
        default=None,
        help="Multiplier for hidden layers.",
    )
    fttransformer_parser.add_argument(
        "--ffn_dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    fttransformer_parser.add_argument(
        "--residual_dropout",
        type=float,
        default=0.0,
        help="Residual dropout probability.",
    )

    ###########################################################################
    # #### DCNv2 ##############################################################
    ###########################################################################

    dcnv2_parser = subparsers.add_parser("dcnv2")
    dcnv2_parser.add_argument(
        "--encoder_type",
        type=str,
        default="conv",
        help="Type of encoder to use.",
    )
    dcnv2_parser.add_argument(
        "--d_out",
        type=int,
        default=128,
        help="Dimension of output data.",
    )
    dcnv2_parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of input data.",
    )
    dcnv2_parser.add_argument(
        "--n_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers.",
    )
    dcnv2_parser.add_argument(
        "--n_cross_layers",
        type=int,
        default=3,
        help="Number of cross layers.",
    )
    dcnv2_parser.add_argument(
        "--hidden_dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    dcnv2_parser.add_argument(
        "--cross_dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    dcnv2_parser.add_argument(
        "--stacked",
        type=bool,
        default=True,
        help="Whether to use stacked layers.",
    )
    dcnv2_parser.add_argument(
        "--d_embedding",
        type=int,
        default=128,
        help="Dimension of embedding.",
    )

    ###########################################################################
    ##### SNN #################################################################
    ###########################################################################

    snn_parser = subparsers.add_parser("snn")
    snn_parser.add_argument(
        "--d_out",
        type=int,
        default=128,
        help="Dimension of output data.",
    )
    snn_parser.add_argument(
        "--d_layers",
        type=int,
        default=[115, 340, 284],
        help="Dimension of hidden layers.",
    )
    snn_parser.add_argument(
        "--dropout",
        type=int,
        default=0.1,
        help="Dropout probability.",
    )
    snn_parser.add_argument(
        "--d_embedding",
        type=int,
        default=128,
        help="Dimension of embedding.",
    )

    ###########################################################################
    ##### AutoInt #############################################################
    ###########################################################################

    auto_int_parser = subparsers.add_parser("auto_int")
    auto_int_parser.add_argument(
        "--encoder_type",
        type=str,
        default="conv",
        help="Type of encoder to use.",
    )
    auto_int_parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Number of layers.",
    )
    auto_int_parser.add_argument(
        "--d_token",
        type=int,
        default=32,
        help="Dimension of token.",
    )
    auto_int_parser.add_argument(
        "--n_heads",
        type=int,
        default=2,
        help="Number of heads.",
    )
    auto_int_parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="Attention dropout probability.",
    )
    auto_int_parser.add_argument(
        "--residual_dropout",
        type=float,
        default=0.0,
        help="Residual dropout probability.",
    )
    auto_int_parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function.",
    )
    auto_int_parser.add_argument(
        "--prenormalization",
        type=bool,
        default=False,
        help="Whether to use prenormalization.",
    )
    auto_int_parser.add_argument(
        "--initialization",
        type=str,
        default="kaiming",
        help="Initialization.",
    )
    auto_int_parser.add_argument(
        "--kv_compression",
        type=Optional[float],
        default=None,
        help="KV compression.",
    )
    auto_int_parser.add_argument(
        "--kv_compression_sharing",
        type=Optional[str],
        default=None,
        help="KV compression sharing.",
    )
    auto_int_parser.add_argument(
        "--d_out",
        type=int,
        default=32,
        help="Dimension of output data.",
    )

    ###########################################################################
    ##### SAINT ###############################################################
    ###########################################################################

    saint_parser = subparsers.add_parser("saint")
    saint_parser.add_argument(
        "--encoder_type",
        type=str,
        default="conv",
        help="Type of encoder to use.",
    )

    saint_parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Dimension of input data.",
    )
    saint_parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Number of layers.",
    )
    saint_parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of heads.",
    )
    saint_parser.add_argument(
        "--dim_head",
        type=int,
        default=16,
        help="Dimension of head.",
    )
    saint_parser.add_argument(
        "--dim_out",
        type=int,
        default=32,
        help="Dimension of output data.",
    )
    saint_parser.add_argument(
        "--mlp_hidden_mults",
        type=tuple,
        default=(4, 2),
        help="Multiplier for hidden layers.",
    )
    saint_parser.add_argument(
        "--mlp_act",
        type=str,
        default=None,
        help="Activation function.",
    )
    saint_parser.add_argument(
        "--num_special_tokens",
        type=int,
        default=0,
        help="Number of special tokens.",
    )
    saint_parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.0,
        help="Attention dropout probability.",
    )
    saint_parser.add_argument(
        "--ff_dropout",
        type=float,
        default=0.0,
        help="FF dropout probability.",
    )
    saint_parser.add_argument(
        "--cont_embeddings",
        type=str,
        default="MLP",
        help="Continuous embeddings.",
    )
    saint_parser.add_argument(
        "--scalingfactor",
        type=int,
        default=10,
        help="Scaling factor.",
    )
    saint_parser.add_argument(
        "--attentiontype",
        type=str,
        default="col",
        help="Attention type.",
    )
    saint_parser.add_argument(
        "--final_mlp_style",
        type=str,
        default="common",
        help="Final MLP style.",
    )

    ###########################################################################
    ##### XGBoost #############################################################
    ###########################################################################

    xgboost_parser = subparsers.add_parser("xgboost")
    xgboost_parser.add_argument(
        "--n_estimators",
        type=int,
        default=1000,
        help="Number of estimators.",
    )
    xgboost_parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth.",
    )
    xgboost_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    xgboost_parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs.",
    )
    xgboost_parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state.",
    )

    model_parsers = {
        "mlp": mlp_parser,
        "resnet": resnet_parser,
        "fttransformer": fttransformer_parser,
        "dcnv2": dcnv2_parser,
        "snn": snn_parser,
        "autoint": auto_int_parser,
        "saint": saint_parser,
        "xgboost": xgboost_parser,
    }

    model_name = parser.parse_known_args()[0].model_name
    model_parser = model_parsers.get(model_name, None)

    if model_parser is None:
        raise ValueError(f"Unknown model name: {model_name}")

    return parser, model_parser
