from pathlib import Path

import numpy as np
import pandas as pd
import os

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class HiggsEmbedded(BaseDataset):

    def __init__(self, args):
        super(HiggsEmbedded, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.name = "higgs_embedded"
        self.tmp_file_names = [
            "higgs_embedded.npy",
            "higgs.csv",
        ]
        self.task_type = TASK_TYPE.BINARY_CLASS

    def load(self):

        embedded_data = np.load(os.path.join(self.data_path, self.tmp_file_names[0]))
        file = os.path.join(self.args.data_path, self.tmp_file_names[1])

        data_table = pd.read_csv(file, header=None).to_numpy()
        self.y = data_table[:, 0]
        self.X = embedded_data

        self.N, self.D, self.H = self.X.shape

        # One dimension is the [CLS] target
        self.D -= 1

        del data_table
        del embedded_data

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.is_data_loaded = True

        self.cardinalities = []
        self.num_or_cat = {}
