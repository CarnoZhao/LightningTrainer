from copy import deepcopy
from io import BytesIO
from base64 import b64encode, b64decode

import torch
from pytorch_lightning.callbacks import Callback

from ..optimizer import get_optimizer


from .client import Client


def encode(stt):
    io = BytesIO()
    torch.save(stt, io)
    io.seek(0)
    content = b64encode(io.read()).decode()
    return content

def decode(content):
    io = BytesIO(b64decode(content.encode()))
    stt = torch.load(io)
    return stt

class FedAvg(Callback):
    def __init__(self, agg_interval = 1, finetune_epochs = 0, skipped_layer = [], **client_args):
        self.client = Client(**client_args)
        self.agg_interval = agg_interval
        self.finetune_epochs = finetune_epochs
        self.skipped_layer = skipped_layer

    @staticmethod
    def fed_func(contents):
        stt = [decode(_) for _ in contents]
        for k in stt[0]:
            for i in range(1, len(stt)):
                stt[0][k] += stt[i][k]
            stt[0][k] = stt[0][k] / len(stt)
        content = encode(stt[0])
        return content

    def make_content(self, trainer, pl_module):
        stt = pl_module.state_dict()
        stt = {k: v for k, v in stt.items() if all([not k.startswith(_) for _ in self.skipped_layer])}
        content = encode(stt)
        return content

    def parse_content(self, trainer, pl_module, content):
        stt = decode(content)
        pl_module.load_state_dict(stt, strict = False)

    def on_fit_start(self, trainer, pl_module):
        self.client.join()

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.agg_interval == 0 and pl_module.current_epoch < trainer.max_epochs - self.finetune_epochs:
            content = self.make_content(trainer, pl_module)
            content = self.client.communicate_weight(content)
            self.parse_content(trainer, pl_module, content)

    def on_fit_end(self, trainer, pl_module):
        self.client.end()

class FedAvgWeighted(FedAvg):
    def __init__(self, client_weight, **client_args):
        self.client = Client(**client_args)
        self.client_weight = client_weight

    @staticmethod
    def fed_func(contents):
        stt = [decode(_[0]) for _ in contents]
        weights = [_[1] for _ in contents]
        for k in stt[0]:
            stt[0][k] = stt[0][k] * weights[0]
            for i in range(1, len(stt)):
                stt[0][k] += stt[i][k] * weights[i]
            stt[0][k] = stt[0][k] / sum(weights)
        content = encode(stt[0])
        return content

    def make_content(self, trainer, pl_module):
        content = [encode(pl_module.state_dict()), self.client_weight]
        return content

    def parse_content(self, trainer, pl_module, content):
        pl_module.load_state_dict(decode(content))
        return

class FedAvgOpt(FedAvg):
    @staticmethod
    def fed_func(contents):
        stt = [decode(_[0]) for _ in contents]
        opt_stt = [decode(_[1]) for _ in contents]
        for k in stt[0]:
            for i in range(1, len(stt)):
                stt[0][k] += stt[i][k]
            stt[0][k] = stt[0][k] / len(stt)
        for param_id in opt_stt[0]["state"]:
            for k in opt_stt[0]["state"][param_id]:
                for i in range(1, len(opt_stt)):
                    opt_stt[0]["state"][param_id][k] += opt_stt[i]["state"][param_id][k]
                opt_stt[0]["state"][param_id][k] = opt_stt[0]["state"][param_id][k] / len(opt_stt)
        content = [encode(stt[0]), encode(opt_stt[0])]
        return content

    def make_content(self, trainer, pl_module):
        content = [encode(pl_module.state_dict()), encode(trainer.optimizers[0].state_dict())]
        return content

    def parse_content(self, trainer, pl_module, content):
        pl_module.load_state_dict(decode(content[0]))
        trainer.optimizers[0].load_state_dict(decode(content[1]))
        return
  

callbacks = {_.__name__: _ for _ in [
    FedAvg,
    FedAvgWeighted,
    FedAvgOpt,
]}