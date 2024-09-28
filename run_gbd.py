from argparse import Namespace
import json
import os
from typing import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tabulate import tabulate
from xgboost import XGBClassifier, XGBRegressor

from benchmark import load_config
from src.datasets.base import BaseDataset
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from src.utils.models_utils import TASK_TYPE


class XGBoost:
    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        task_type: str = "classification",
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        if task_type in [
            TASK_TYPE.MULTI_CLASS,
            TASK_TYPE.BINARY_CLASS,
        ]:
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif task_type == TASK_TYPE.REGRESSION:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=n_jobs,
                random_state=random_state,
            )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, verbose=True)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_model_args(
        datamodule,
        args: OrderedDict,
        model_args: OrderedDict,
    ) -> OrderedDict:
        return model_args


def preprocessing(
    data: np.array,
    dataset: BaseDataset,
    args: OrderedDict,
    **kwargs,
):
    train_encoded_dataset = data
    if not args.using_embedding:
        train_encoded_dataset = []
        categorical_idx = [card[0] for card in dataset.cardinalities]
        for col_index in range(dataset.D):
            train_col = data[:, col_index].reshape(-1, 1)
            if col_index in categorical_idx:
                fitted_encoder = OneHotEncoder(sparse_output=False).fit(train_col)
            else:
                fitted_encoder = MinMaxScaler().fit(train_col)
            encoded_train_col = fitted_encoder.transform(train_col).astype(np.float32)
            train_encoded_dataset.append(np.array(encoded_train_col).astype(np.float32))

        train_encoded_dataset = np.concatenate(train_encoded_dataset, axis=1)

    return train_encoded_dataset


def main(args: Namespace) -> dict:

    print("General args: ")
    print(
        tabulate(
            sorted(list(vars(args).items()), key=lambda x: x[0]),
            tablefmt="fancy_grid",
        )
    )

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
    dataset.load()
    args.task_type = dataset.task_type

    if args.model_name == "xgboost":
        model_class = XGBoost

    model = model_class(**vars(args))

    X, y = dataset.X, dataset.y
    X = preprocessing(X, dataset, args)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size_ratio,
        random_state=args.random_state,
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Predicting...")
    y_pred = model.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    return {"accuracy": accuracy}


def run_experiment(config_file, num_runs):

    output_dir = config_file.split("/")[-1].replace(".json", "")
    os.makedirs(output_dir, exist_ok=True)

    args = Namespace(**load_config(config_file))

    for i in range(num_runs):

        output_file = f"{output_dir}/output_{i}.json"
        results = main(args)

        results.update(vars(args))

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ML experiment multiple times and calculate mean and std."
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the JSON configuration file"
    )
    parser.add_argument(
        "--num_runs", type=int, default=10, help="Number of times to run the experiment"
    )

    args = parser.parse_args()

    run_experiment(args.config_file, args.num_runs)
