import datetime

import torch

PLOT_DEBUG_VALUES = False


def make_job_name(args):
    job_name = (
        "{dataset}__model_nlyrs_{n_layers}_nheads_{n_heads}_hdim_{hdim}"
        "__pred_ovrlap_{overlap}_npreds_{n_mask_preds}_"
        "_nlyrs_{n_pred_layers}_activ_{activation}"
        "nenc_{n_mask_ctx}_inter_ctx_{min_ctx_share}_{max_ctx_share}"
        "_inter_trgt_{min_tgrt_share}_{max_tgrt_share}"
        "__lr_{lr}_start_{start_lr}_final_{final_lr}_{datetime}"
    )
    job_name = job_name.format(
        dataset=args.data_set,
        n_layers=args.model_num_layers,
        n_heads=args.model_num_heads,
        hdim=args.model_dim_hidden,
        overlap="T" if args.mask_allow_overlap else "F",
        n_mask_preds=args.mask_num_preds,
        n_pred_layers=args.pred_num_layers,
        activation=args.pred_activation,
        n_mask_ctx=args.mask_num_encs,
        min_ctx_share=args.mask_min_ctx_share,
        max_ctx_share=args.mask_max_ctx_share,
        min_tgrt_share=args.mask_min_trgt_share,
        max_tgrt_share=args.mask_max_trgt_share,
        lr=args.exp_lr,
        start_lr=args.exp_start_lr,
        final_lr=args.exp_final_lr,
        datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return job_name


def print_args(args):
    to_print_model = (
        "Encoder and Target Encoders:\n"
        "\t- Number of layers: {n_layers}\n"
        "\t- Number of att heads: {n_heads}\n"
        "\t- Hidden dimension: {hdim}\n"
        "\t- Dropout: {model_p_dropout}\n"
    )
    print(
        to_print_model.format(
            n_layers=args.model_num_layers,
            n_heads=args.model_num_heads,
            hdim=args.model_dim_hidden,
            model_p_dropout=args.model_dropout_prob,
        )
    )

    to_print_mask = (
        "Masking:\n"
        "\t- Allow overlap: {overlap}\n"
        "\t- Mask share interval for target: [{min_trgt},{max_trgt}]\n"
        "\t- Mask share interval for context: [{min_ctxt},{max_ctxt}]\n"
        "\t- Number of masks for target: {n_preds}\n"
        "\t- Number of masks for context: {n_ctx}\n"
    )
    print(
        to_print_mask.format(
            overlap=("True" if args.mask_allow_overlap else "False"),
            min_trgt=args.mask_min_trgt_share,
            max_trgt=args.mask_max_trgt_share,
            min_ctxt=args.mask_min_ctx_share,
            max_ctxt=args.mask_max_ctx_share,
            n_preds=args.mask_num_preds,
            n_ctx=args.mask_num_encs,
        )
    )

    to_print_pred = (
        "Predictors:\n"
        "\t- Number of layers: {pred_num_layers}\n"
        "\t- Dropout: {pred_p_dropout}\n"
        "\t- Layer norm epsilon: {pred_layer_norm_eps}\n"
        "\t- Activation function: {pred_activation}\n"
    )
    print(
        to_print_pred.format(
            pred_num_layers=args.pred_num_layers,
            pred_p_dropout=args.pred_p_dropout,
            pred_layer_norm_eps=args.pred_layer_norm_eps,
            pred_activation=args.pred_activation,
        )
    )

    to_print_optimization = (
        "Optimization details:\n"
        "\t- Optimizer: AdamW (default)\n"
        "\t- Learning rate scheduler: {scheduler}\n"
        "\t- Reference learning rate: {ref_lr}\n"
        "\t- Start learning rate: {start_lr}\n"
        "\t- Final learning rate: {final_lr}\n"
        "\t- Weight decay scheduler: {wd_scheduler}\n"
        "\t- Weight decay: {wd}\n"
        "\t- Final weight decay: {final_wd}\n"
        "\t- Gradient clipping: {gradient_clipping}\n"
    )

    print(
        to_print_optimization.format(
            scheduler=("True" if args.exp_scheduler else "False"),
            ref_lr=args.exp_lr,
            start_lr=args.exp_start_lr,
            final_lr=args.exp_final_lr,
            wd_scheduler=("True" if args.exp_weight_decay_scheduler else "False"),
            wd=args.exp_weight_decay,
            final_wd=args.exp_final_weight_decay,
            gradient_clipping=args.exp_gradient_clipping,
        )
    )


def _debug_values(data: torch.Tensor, title="Data", skip=False):
    if skip or not PLOT_DEBUG_VALUES:
        return

    if "plt" not in globals():
        import matplotlib.pyplot as plt

    _data = data
    if len(data.shape) == 1:
        _data = data.unsqueeze(0)

    plt.imshow(_data.cpu().detach().numpy(), aspect="auto")

    if len(data.shape) == 1:
        for i in range(data.shape[0]):
            # make the value in scientific notation if it is less than 0.01
            value = (
                "{:.2f}".format(data[i].item())
                if abs(data[i].item()) > 0.01
                else "{:.1e}".format(data[i].item())
            )
            plt.text(
                i,
                0,
                s=value,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = (
                    "{:.2f}".format(data[i, j].item())
                    if abs(data[i, j].item()) > 0.01
                    else "{:.1e}".format(data[i, j].item())
                )
                plt.text(
                    j,
                    i,
                    s=value,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )
    plt.title(title)
    plt.show()
