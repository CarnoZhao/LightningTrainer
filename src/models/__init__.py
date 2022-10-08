from .smp_models import SMPModel
from .segformer_models import Segformer
from .mmseg_models import MMSegModel
from .monai_models import MonaiModel
from .timm_models import TIMMModel

models = {_.__name__: _ for _ in [
    SMPModel,
    Segformer,
    MMSegModel,
    MonaiModel,
    TIMMModel
]}