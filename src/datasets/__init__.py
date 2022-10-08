from .img_dataset import ImgData
from .seg_dataset import SegData
from .naso_dataset import NasoData

datasets = {_.__name__: _ for _ in [
    ImgData,
    SegData,
    NasoData
]}
