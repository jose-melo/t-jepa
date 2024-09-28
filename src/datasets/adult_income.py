import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE

CAT_NAMES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]


class Adult(BaseDataset):
    """
    https://www.kaggle.com/datasets/wenruliu/adult-income-dataset?resource=download

    Adult Income Dataset
    Number of instances 48842
    Number of features 14

    Feature description

    age: continuous.
    workclass: Private, Self-emp-not-inc,
               Self-emp-inc, Federal-gov, Local-gov,
               State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad,
               Prof-school, Assoc-acdm, Assoc-voc, 9th,
               7th-8th, 12th, Masters, 1st-4th, 10th,
               Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married,
                    Separated, Widowed, Married-spouse-absent,
                    Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales,
                Exec-managerial, Prof-specialty, Handlers-cleaners,
                Machine-op-inspct, Adm-clerical, Farming-fishing,
                Transport-moving, Priv-house-serv, Protective-serv,
                Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative,
                  Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico,
                    Canada, Germany, Outlying-US(Guam-USVI-etc), India,
                    Japan, Greece, South, China, Cuba, Iran, Honduras,
                    Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
                    Portugal, Ireland, France, Dominican-Republic, Laos,
                    Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
                    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador,
                    Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    """

    def __init__(self, args):
        super(Adult, self).__init__(args)

        self.is_data_loaded = False
        self.tmp_file_names = ["adult.csv"]
        self.name = "adult"
        self.args = args
        self.task_type = TASK_TYPE.BINARY_CLASS

    def load(self):

        path = os.path.join(self.args.data_path, self.tmp_file_names[0])
        data = pd.read_csv(path)

        le = LabelEncoder()
        for col in CAT_NAMES:
            data[col] = le.fit_transform(data[col])

        data["income"] = le.fit_transform(data["income"])
        self.y = data["income"].to_numpy()

        data = data.drop("income", axis=1)
        self.X = data.to_numpy()

        self.N, self.D = self.X.shape

        self.cardinalities = [
            (1, 9),
            (3, 16),
            (5, 7),
            (6, 15),
            (7, 6),
            (8, 5),
            (9, 2),
            (13, 42),
        ]
        self.num_features = [
            0,
            2,
            4,
            10,
            11,
            12,
        ]
        self.cat_features = [1, 3, 5, 6, 7, 8, 9, 13]
        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loader = True
