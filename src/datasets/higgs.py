from pathlib import Path

import numpy as np
import pandas as pd
import os

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class Higgs(BaseDataset):

    def __init__(self, args):
        super(Higgs, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.name = "higgs"
        self.tmp_file_names = ["higgs.csv"]
        self.task_type = TASK_TYPE.BINARY_CLASS

    def load(self):

        file = os.path.join(self.args.data_path, self.tmp_file_names[0])

        data_table = pd.read_csv(file, header=None).to_numpy()
        self.y = data_table[1:, 0]
        self.X = data_table[1:, 1:]

        self.N, self.D = self.X.shape

        self.cardinalities = [(8, 4), (12, 4), (16, 4), (20, 4)]
        self.cat_features = [8, 12, 16, 20]

        self.num_features = [i for i in range(self.D) if i not in self.cat_features]

        self.is_data_loaded = True

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        for i in self.cat_features:
            self.X[:, i] = pd.factorize(self.X[:, i])[0]
