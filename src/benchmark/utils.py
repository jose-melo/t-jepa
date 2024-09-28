import dataclasses
from torch import nn

from src.models.autoint import AutoInt
from src.models.dcnv2 import DCNv2
from src.models.fttransformer import FTTransformer
from run_gbd import XGBoost
from src.models.linear_probing import LinearProbe
from src.models.mlp import MLP
from src.models.resnet import ResNet
from src.utils.models_utils import BaseModel

MODEL_CONFIG_BASE_PATH = "src/benchmark/model_configs/{dataset_name}/{model_name}.json"

MODEL_NAME_TO_MODEL_MAP: dict[str, BaseModel] = {
    "mlp": MLP,
    "resnet": ResNet,
    "fttransformer": FTTransformer,
    "dcnv2": DCNv2,
    "autoint": AutoInt,
    "linear_probe": LinearProbe,
    "xgboost": XGBoost,
}


def get_loss_from_task(task: str) -> nn.Module:
    if task == "regression":
        return nn.MSELoss(reduction="mean")
    elif task == "multi_class":
        return nn.CrossEntropyLoss(reduction="mean")
    elif task == "binary_class":
        return nn.BCEWithLogitsLoss(reduction="mean")
    else:
        raise ValueError(f"Task {task} not supported.")
