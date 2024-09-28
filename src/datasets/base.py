import numpy as np

from typing import TypedDict


class ArgsDataset(TypedDict):
    data_path: str
    D: int
    N: int
    cat_features: list
    num_features: list
    cardinalities: list
    is_loaded: bool
    name: str


class BaseDataset:

    def __init__(self, args: ArgsDataset):
        self.data_path = args.data_path
        self.D = None
        self.N = None
        self.X = None
        self.y = None
        self.is_data_loaded = False
        self.num_or_cat = {}
        self.cat_features = []
        self.num_features = []

        self.name = None

        ## list composed of the cardinality of each cat feature
        ## probably in the form of a tuple: (idx, cardinality)
        self.cardinalities = []
        self.is_loaded = False
        self.task_type = None

    def load(self):
        return None

    def __repr__(self):
        repr = (
            f"{self.name} dataset: {self.N} samples, {self.D} features\n"
            f"{len(self.cat_features)} categorical features\n"
            f"{len(self.num_features)} numerical features"
        )
        return repr
