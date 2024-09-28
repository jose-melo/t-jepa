from scipy.io import arff
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class Aloi(BaseDataset):
    """
    https://www.openml.org/search?type=data&sort=runs&id=40906&status=active

    number of instances	49999
    number of features	28
    number of classes	2
    number of missing values	0
    number of instances with missing values	0
    number of numeric features	27
    number of symbolic features	1

    """

    def __init__(self, args):
        super(Aloi, self).__init__(args)

        self.is_data_loaded = False
        self.tmp_file_names = ["aloi.csv"]
        self.name = "aloi"
        self.args = args
        self.task_type = TASK_TYPE.MULTI_CLASS

    def load(self):

        data = pd.read_csv(os.path.join(self.args.data_path, "aloi.csv"))

        self.X = data.iloc[:, :-1].to_numpy()

        self.y = data.iloc[:, -1].to_numpy()

        self.N, self.D = self.X.shape

        del data

        self.cardinality = []
        self.num_or_cat = {}

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.is_data_loaded = True
