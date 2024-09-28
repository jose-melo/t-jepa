from src.datasets.adult_income_embedded import AdultEmbedded
from src.datasets.aloi_embedded import AloiEmbedded
from src.datasets.base import BaseDataset
from src.datasets.adult_income import Adult
from src.datasets.aloi import Aloi
from src.datasets.california_housing import California
from src.datasets.california_housing_embedded import CaliforniaEmbedded
from src.datasets.helena import Helena
from src.datasets.helena_embedded import HelenaEmbedded
from src.datasets.higgs import Higgs
from src.datasets.higgs_embedded import HiggsEmbedded
from src.datasets.jannis import Jannis
from src.datasets.jannis_embedded import JannisEmbedded

DATASET_NAME_TO_DATASET_MAP = {
    "adult": Adult,
    "aloi": Aloi,
    "california": California,
    "helena": Helena,
    "higgs": Higgs,
    "jannis": Jannis,
    "helena_embedded": HelenaEmbedded,
    "jannis_embedded": JannisEmbedded,
    "aloi_embedded": AloiEmbedded,
    "higgs_embedded": HiggsEmbedded,
    "california_embedded": CaliforniaEmbedded,
    "adult_embedded": AdultEmbedded,
}
