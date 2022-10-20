import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", dest = "config", default = "config.yaml", type = str)
parser.add_argument("--gpus", dest = "gpus", default = "0", type = str)
parser.add_argument("--override", dest = "override", type = str, nargs = "+")

def parse_args():
    args = parser.parse_args()
    return args