import os
import numpy as np

from scipy.io import arff
import pandas as pd

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class JannisEmbedded(BaseDataset):
    """
    https://www.openml.org/search?type=data&sort=runs&id=41168&status=active

    number of instances	83733
    number of features	55
    number of classes	4
    number of missing values	0
    number of instances with missing values	0
    number of numeric features	54
    number of symbolic features	1
    """

    def __init__(self, args):
        super(JannisEmbedded, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.name = "jannis_embedded"
        self.tmp_file_names = [
            "jannis_embedded.npy",
            "jannis.arff",
        ]
        self.task_type = TASK_TYPE.MULTI_CLASS

    def load(self):

        embedded_data = np.load(os.path.join(self.data_path, self.tmp_file_names[0]))

        orig_data, _ = arff.loadarff(
            os.path.join(self.data_path, self.tmp_file_names[1])
        )
        orig_data = pd.DataFrame(orig_data)

        self.X = embedded_data
        self.y = orig_data["class"].apply(int).to_numpy()

        self.N = self.X.shape[0]
        self.D = 54
        self.H = self.X.shape[-1]

        self.cardinalities = []
        self.num_or_cat = {}

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loaded = True
