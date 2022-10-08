from .img_dataset import ImgData
from .seg_dataset import SegData

datasets = {_.__name__: _ for _ in [
    ImgData,
    SegData,
]}
