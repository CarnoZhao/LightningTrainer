from .dice_metric import DiceMetric
from .classification_metric import ClassificationMetric
from .MIL_metric import MILMetric

metrics = {_.__name__: _ for _ in [
    DiceMetric, 
    ClassificationMetric,
    MILMetric
]}