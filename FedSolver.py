import os
import glob
import argparse
from omegaconf import OmegaConf
parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", dest = "config", default = "configs/fed", type = str)
args = parser.parse_args()


configs = glob.glob(os.path.join(args.config, "*"))
base_config = [_ for _ in configs if "_base_.yaml" in _][0]
fed_type = OmegaConf.load(base_config).train.federated.type

from src.federated import Server
Server(len(configs) - 1, fed_type).run()

