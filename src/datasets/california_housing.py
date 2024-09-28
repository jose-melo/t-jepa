import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class California(BaseDataset):
    """
    https://www.kaggle.com/datasets/camnugent/california-housing-prices

    Feature description

    1. longitude: A measure of how far west a house is;
                  a higher value is farther west
    2. latitude: A measure of how far north a house is;
                 a higher value is farther north
    3. housingMedianAge: Median age of a house within a block;
                         a lower number is a newer building
    4. totalRooms: Total number of rooms within a block
    5. totalBedrooms: Total number of bedrooms within a block
    6. population: Total number of people residing within a block
    7. households: Total number of households, a group of people
                   residing within a home unit, for a block
    8. medianIncome: Median income for households within a
                     block of houses (measured in tens of thousands
                     of US Dollars)
    9. medianHouseValue: Median house value for households
                         within a block (measured in US Dollars) -> target
    10. oceanProximity: Location of the house w.r.t ocean/sea
    """

    def __init__(self, args):
        super(California, self).__init__(args)

        self.is_data_loaded = False
        self.tmp_file_names = ["housing.csv"]
        self.name = "california"
        self.args = args
        self.task_type = TASK_TYPE.REGRESSION

    def load(self):

        path = os.path.join(self.args.data_path, self.tmp_file_names[0])
        data = pd.read_csv(path)

        data.drop(columns=["ocean_proximity"], inplace=True)

        self.X = data.drop("median_house_value", axis=1).to_numpy()
        self.y = data["median_house_value"].to_numpy()

        self.N, self.D = self.X.shape

        self.cardinalities = []
        self.num_features = list(range(self.D))
        self.cat_features = []

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}
