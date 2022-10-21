import os
from omegaconf import OmegaConf

def parse_configs(args):
    cfg = args.config
    if os.path.exists(os.path.join(os.path.dirname(cfg), "_base_.yaml")):
        base_cfg = os.path.join(os.path.dirname(cfg), "_base_.yaml")
        cfg = OmegaConf.merge(OmegaConf.load(base_cfg), OmegaConf.load(cfg))
    else:
        cfg = OmegaConf.load(cfg)
    cfg.gpus = args.gpus
    if args.override is not None:
        cfg = override_config(args.override, cfg)
    return cfg

def override_config(override, cfg):
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(override))



        
        