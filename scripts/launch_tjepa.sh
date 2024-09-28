#!/bin/bash

if ! command -v python &> /dev/null
then
    echo "Python could not be found. Please install Python to proceed."
    exit
fi

BATCH_SIZE=512
DATA_PATH="./datasets"
DATA_SET="jannis"
EXP_CACHE_CADENCE=20
EXP_FINAL_LR=0
EXP_FINAL_WEIGHT_DECAY=0
EXP_LR=0.0003658682841082736
EXP_PATIENCE=100
EXP_START_LR=0
EXP_TRAIN_TOTAL_EPOCHS=300
EXP_WARMUP=10
EXP_WEIGHT_DECAY=0
INIT_TYPE="trunc_normal"
LOG_TENSORBOARD=False
MASK_MAX_CTX_SHARE=0.36819012681604135
MASK_MAX_TRGT_SHARE=0.6222278244105446
MASK_MIN_CTX_SHARE=0.13628349247854785
MASK_MIN_TRGT_SHARE=0.1556321308543076
MASK_NUM_PREDS=4
MODEL_DIM_FEEDFORWARD=64
MODEL_DIM_HIDDEN=64
MODEL_DROPOUT_PROB=0.0018542257314766695
MODEL_EMA_END=1
MODEL_EMA_START=0.996
MODEL_FEATURE_INDEX_EMBEDDING=False
MODEL_FEATURE_TYPE_EMBEDDING=False
MODEL_NUM_HEADS=2
MODEL_NUM_LAYERS=16
PIN_MEMORY=False
PRED_EMBED_DIM=16
PRED_NUM_HEADS=4
PRED_NUM_LAYERS=16
PRED_P_DROPOUT=0.0019307456645321797
PRED_TYPE="transformer"
PROBE_CADENCE=20
PROBE_MODEL="linear_probe"
TJEPA_RANDOM=false
N_CLS_TOKENS=1

function show_help() {
    echo "Usage: ./script.sh [options]"
    echo ""
    echo "Options:"
    echo "  --batch_size            Batch size (default: $BATCH_SIZE)"
    echo "  --data_path             Path to the datasets (default: $DATA_PATH)"
    echo "  --data_set              Dataset name (default: $DATA_SET)"
    echo "  --exp_cache_cadence     Experiment cache cadence (default: $EXP_CACHE_CADENCE)"
    echo "  --exp_final_lr          Experiment final learning rate (default: $EXP_FINAL_LR)"
    echo "  --exp_final_weight_decay Experiment final weight decay (default: $EXP_FINAL_WEIGHT_DECAY)"
    echo "  --exp_lr                Experiment learning rate (default: $EXP_LR)"
    echo "  --exp_patience          Experiment patience (default: $EXP_PATIENCE)"
    echo "  --exp_start_lr          Experiment start learning rate (default: $EXP_START_LR)"
    echo "  --exp_train_total_epochs Experiment total epochs (default: $EXP_TRAIN_TOTAL_EPOCHS)"
    echo "  --exp_warmup            Experiment warmup (default: $EXP_WARMUP)"
    echo "  --exp_weight_decay      Experiment weight decay (default: $EXP_WEIGHT_DECAY)"
    echo "  --init_type             Initialization type (default: $INIT_TYPE)"
    echo "  --log_tensorboard       Log to TensorBoard (default: $LOG_TENSORBOARD)"
    echo "  --mask_max_ctx_share    Mask max context share (default: $MASK_MAX_CTX_SHARE)"
    echo "  --mask_max_trgt_share   Mask max target share (default: $MASK_MAX_TRGT_SHARE)"
    echo "  --mask_min_ctx_share    Mask min context share (default: $MASK_MIN_CTX_SHARE)"
    echo "  --mask_min_trgt_share   Mask min target share (default: $MASK_MIN_TRGT_SHARE)"
    echo "  --mask_num_preds        Mask number of predictions (default: $MASK_NUM_PREDS)"
    echo "  --model_dim_feedforward Model dimension feedforward (default: $MODEL_DIM_FEEDFORWARD)"
    echo "  --model_dim_hidden      Model dimension hidden (default: $MODEL_DIM_HIDDEN)"
    echo "  --model_dropout_prob    Model dropout probability (default: $MODEL_DROPOUT_PROB)"
    echo "  --model_ema_end         Model EMA end (default: $MODEL_EMA_END)"
    echo "  --model_ema_start       Model EMA start (default: $MODEL_EMA_START)"
    echo "  --model_feature_index_embedding Model feature index embedding (default: $MODEL_FEATURE_INDEX_EMBEDDING)"
    echo "  --model_feature_type_embedding Model feature type embedding (default: $MODEL_FEATURE_TYPE_EMBEDDING)"
    echo "  --model_num_heads       Model number of heads (default: $MODEL_NUM_HEADS)"
    echo "  --model_num_layers      Model number of layers (default: $MODEL_NUM_LAYERS)"
    echo "  --pin_memory            Pin memory (default: $PIN_MEMORY)"
    echo "  --pred_embed_dim        Prediction embed dimension (default: $PRED_EMBED_DIM)"
    echo "  --pred_num_heads        Prediction number of heads (default: $PRED_NUM_HEADS)"
    echo "  --pred_num_layers       Prediction number of layers (default: $PRED_NUM_LAYERS)"
    echo "  --pred_p_dropout        Prediction dropout probability (default: $PRED_P_DROPOUT)"
    echo "  --pred_type             Prediction type (default: $PRED_TYPE)"
    echo "  --probe_cadence         Probe cadence (default: $PROBE_CADENCE)"
    echo "  --probe_model           Probe model (default: $PROBE_MODEL)"
    echo "  --n_cls_tokens          Number of classification tokens (default: $N_CLS_TOKENS)"
    echo "  --random          Random seed (default: $TJEPA_RANDOM)"
    echo "  -h, --help              Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --data_path)
            DATA_PATH="$2"
            shift
            shift
            ;;
        --data_set)
            DATA_SET="$2"
            shift
            shift
            ;;
        --exp_cache_cadence)
            EXP_CACHE_CADENCE="$2"
            shift
            shift
            ;;
        --exp_final_lr)
            EXP_FINAL_LR="$2"
            shift
            shift
            ;;
        --exp_final_weight_decay)
            EXP_FINAL_WEIGHT_DECAY="$2"
            shift
            shift
            ;;
        --exp_lr)
            EXP_LR="$2"
            shift
            shift
            ;;
        --exp_patience)
            EXP_PATIENCE="$2"
            shift
            shift
            ;;
        --exp_start_lr)
            EXP_START_LR="$2"
            shift
            shift
            ;;
        --exp_train_total_epochs)
            EXP_TRAIN_TOTAL_EPOCHS="$2"
            shift
            shift
            ;;
        --exp_warmup)
            EXP_WARMUP="$2"
            shift
            shift
            ;;
        --exp_weight_decay)
            EXP_WEIGHT_DECAY="$2"
            shift
            shift
            ;;
        --init_type)
            INIT_TYPE="$2"
            shift
            shift
            ;;
        --log_tensorboard)
            LOG_TENSORBOARD="$2"
            shift
            shift
            ;;
        --mask_max_ctx_share)
            MASK_MAX_CTX_SHARE="$2"
            shift
            shift
            ;;
        --mask_max_trgt_share)
            MASK_MAX_TRGT_SHARE="$2"
            shift
            shift
            ;;
        --mask_min_ctx_share)
            MASK_MIN_CTX_SHARE="$2"
            shift
            shift
            ;;
        --mask_min_trgt_share)
            MASK_MIN_TRGT_SHARE="$2"
            shift
            shift
            ;;
        --mask_num_preds)
            MASK_NUM_PREDS="$2"
            shift
            shift
            ;;
        --model_dim_feedforward)
            MODEL_DIM_FEEDFORWARD="$2"
            shift
            shift
            ;;
        --model_dim_hidden)
            MODEL_DIM_HIDDEN="$2"
            shift
            shift
            ;;
        --model_dropout_prob)
            MODEL_DROPOUT_PROB="$2"
            shift
            shift
            ;;
        --model_ema_end)
            MODEL_EMA_END="$2"
            shift
            shift
            ;;
        --model_ema_start)
            MODEL_EMA_START="$2"
            shift
            shift
            ;;
        --model_feature_index_embedding)
            MODEL_FEATURE_INDEX_EMBEDDING="$2"
            shift
            shift
            ;;
        --model_feature_type_embedding)
            MODEL_FEATURE_TYPE_EMBEDDING="$2"
            shift
            shift
            ;;
        --model_num_heads)
            MODEL_NUM_HEADS="$2"
            shift
            shift
            ;;
        --model_num_layers)
            MODEL_NUM_LAYERS="$2"
            shift
            shift
            ;;
        --pin_memory)
            PIN_MEMORY="$2"
            shift
            shift
            ;;
        --pred_embed_dim)
            PRED_EMBED_DIM="$2"
            shift
            shift
            ;;
        --pred_num_heads)
            PRED_NUM_HEADS="$2"
            shift
            shift
            ;;
        --pred_num_layers)
            PRED_NUM_LAYERS="$2"
            shift
            shift
            ;;
        --pred_p_dropout)
            PRED_P_DROPOUT="$2"
            shift
            shift
            ;;
        --pred_type)
            PRED_TYPE="$2"
            shift
            shift
            ;;
        --probe_cadence)
            PROBE_CADENCE="$2"
            shift
            shift
            ;;
        --probe_model)
            PROBE_MODEL="$2"
            shift
            shift
            ;;
        --n_cls_tokens)
            N_CLS_TOKENS="$2"
            shift
            shift
            ;;
        --random)
            TJEPA_RANDOM=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

COMMAND="python run.py \
    --batch_size=$BATCH_SIZE \
    --data_path=$DATA_PATH \
    --data_set=$DATA_SET \
    --exp_cache_cadence=$EXP_CACHE_CADENCE \
    --exp_final_lr=$EXP_FINAL_LR \
    --exp_final_weight_decay=$EXP_FINAL_WEIGHT_DECAY \
    --exp_lr=$EXP_LR \
    --exp_patience=$EXP_PATIENCE \
    --exp_start_lr=$EXP_START_LR \
    --exp_train_total_epochs=$EXP_TRAIN_TOTAL_EPOCHS \
    --exp_warmup=$EXP_WARMUP \
    --exp_weight_decay=$EXP_WEIGHT_DECAY \
    --init_type=$INIT_TYPE \
    --log_tensorboard=$LOG_TENSORBOARD \
    --mask_max_ctx_share=$MASK_MAX_CTX_SHARE \
    --mask_max_trgt_share=$MASK_MAX_TRGT_SHARE \
    --mask_min_ctx_share=$MASK_MIN_CTX_SHARE \
    --mask_min_trgt_share=$MASK_MIN_TRGT_SHARE \
    --mask_num_preds=$MASK_NUM_PREDS \
    --model_dim_feedforward=$MODEL_DIM_FEEDFORWARD \
    --model_dim_hidden=$MODEL_DIM_HIDDEN \
    --model_dropout_prob=$MODEL_DROPOUT_PROB \
    --model_ema_end=$MODEL_EMA_END \
    --model_ema_start=$MODEL_EMA_START \
    --model_feature_index_embedding=$MODEL_FEATURE_INDEX_EMBEDDING \
    --model_feature_type_embedding=$MODEL_FEATURE_TYPE_EMBEDDING \
    --model_num_heads=$MODEL_NUM_HEADS \
    --model_num_layers=$MODEL_NUM_LAYERS \
    --pin_memory=$PIN_MEMORY \
    --pred_embed_dim=$PRED_EMBED_DIM \
    --pred_num_heads=$PRED_NUM_HEADS \
    --pred_num_layers=$PRED_NUM_LAYERS \
    --pred_p_dropout=$PRED_P_DROPOUT \
    --pred_type=$PRED_TYPE \
    --probe_cadence=$PROBE_CADENCE \
    --probe_model=$PROBE_MODEL \
    --n_cls_tokens=$N_CLS_TOKENS"

if [ "$TJEPA_RANDOM" = true ]; then
    COMMAND="$COMMAND --random=True"
fi

echo "Running command: $COMMAND"

$COMMAND