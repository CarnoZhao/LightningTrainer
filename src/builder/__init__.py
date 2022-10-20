from .dataset import get_data, get_trans
from .loss import get_loss
from .metric import get_metric
from .model import get_model
from .optimizer import get_optimizer
from .registry import register, REGISTRY

def get(name, *args, **kwargs):
    return eval("get_" + name)(*args, **kwargs)
    
from .trainer import get_trainer
from .lightning import get_lightning