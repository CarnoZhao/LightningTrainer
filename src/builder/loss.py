import torch.nn as nn
from omegaconf import OmegaConf

from ..builder.registry import REGISTRY
LOSS = REGISTRY["LOSS"]

def _get_loss(cfg):
    cfg = cfg.copy()
    loss_type = cfg.pop("type")
    if loss_type.startswith("nn."):
        return getattr(nn, loss_type[3:])(**cfg)
    else:
        return LOSS[loss_type](**cfg)

def get_loss(cfg):
    cfg = cfg.copy()
    if OmegaConf.is_dict(cfg) and "losses" in cfg:
        multi_inputs = cfg.pop("multi_inputs", False)
        input_weights = cfg.pop("input_weights", None)
        cfg = cfg.losses
    elif OmegaConf.is_list(cfg):
        multi_inputs = False
    elif OmegaConf.is_dict(cfg) and "losses" not in cfg:
        multi_inputs = cfg.pop("multi_inputs", False)
        input_weights = cfg.pop("input_weights", None)
        cfg = [cfg]

    func = LOSS["MultiLoss"](cfg)
    if multi_inputs:
        func = LOSS["MultiInputLoss"](func, input_weights)
    return func
