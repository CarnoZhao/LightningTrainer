import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", dest = "config", default = "config.yaml", type = str)
parser.add_argument("--gpus", dest = "gpus", default = "0", type = str)
parser.add_argument("--override", dest = "override", type = str, nargs = "+")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

from src.builder import get
from src.config import parse_configs

if __name__ == "__main__":
    cfg = parse_configs(args)
    trainer = get("trainer", cfg)
    model = get("lightning", cfg)
    trainer.fit(model)
    