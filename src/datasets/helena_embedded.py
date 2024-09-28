from scipy.io import arff
import os
import numpy as np
import pandas as pd

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class HelenaEmbedded(BaseDataset):
    """
    https://www.openml.org/search?type=data&sort=runs&id=41169&status=active

    Helena dataset
    number of instances	65196
    number of features	28
    number of classes	100
    number of missing values	0
    number of instances with missing values	0
    number of numeric features	27
    number of symbolic features	1

    """

    def __init__(self, args):
        super(HelenaEmbedded, self).__init__(args)

        self.is_data_loaded = False
        self.tmp_file_names = [
            "helena_embedded.npy",
            "helena.arff",
        ]
        self.name = "helena_embedded"
        self.args = args
        self.task_type = TASK_TYPE.MULTI_CLASS

    def load(self):
        embedded_data = np.load(os.path.join(self.data_path, self.tmp_file_names[0]))
        orig_data, _ = arff.loadarff(
            os.path.join(self.data_path, self.tmp_file_names[1])
        )
        orig_data = pd.DataFrame(orig_data)

        ####
        self.X = embedded_data
        self.y = orig_data["class"].apply(int).to_numpy()

        self.D, self.N, self.H = 27, 65196, 16

        del orig_data
        del embedded_data

        self.cardinalities = []
        self.num_or_cat = {}

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loaded = True
