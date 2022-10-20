import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, StochasticWeightAveraging, LearningRateMonitor

from .registry import REGISTRY
CALLBACK = REGISTRY["CALLBACK"]


def get_trainer(cfg):
    cfg = cfg.copy()
    if "seed" in cfg: pl.seed_everything(cfg.seed)

    # logger
    logger = [
        CSVLogger("./logs", 
            name = cfg.name, 
            version = cfg.version, 
            flush_logs_every_n_steps = cfg.train.log_step),
    ]

    # callbacks
    monitor = cfg.train.get("monitor", "valid_metric")
    callbacks = [
        ModelCheckpoint(
            dirpath = os.path.join("./logs", cfg.name, cfg.version),
            filename = '{epoch}_{' + monitor + ':.3f}',
            save_last = True,
            save_top_k = cfg.train.get("save_topk", 3),
            save_weights_only = True,
            mode = "max",
            monitor = monitor),
        RichProgressBar(leave = True),
        LearningRateMonitor('step')
    ]
    if cfg.train.get("swa", False):
        callbacks.append(StochasticWeightAveraging())

    if cfg.train.get("federated", False):
        callbacks.append(CALLBACK[cfg.train.federated.pop("type")](**cfg.train.federated))
        
    # trainer
    trainer = pl.Trainer(
        accelerator = "gpu",
        gpus = list(range(len(cfg.gpus.split(",")))), 
        precision = 16, 
        strategy = cfg.train.get("strategy", "dp"),
        sync_batchnorm = cfg.train.get("strategy", "dp") == "ddp",
        gradient_clip_val = cfg.train.get("grad_clip", 0),
        accumulate_grad_batches = cfg.train.get("grad_acc", 1),
        max_epochs = cfg.train.num_epochs,
        logger = logger,
        callbacks = callbacks,
        log_every_n_steps = cfg.train.log_step,
        val_check_interval = 1.0 if isinstance(cfg.train.val_interval, int) else cfg.train.val_interval,
        check_val_every_n_epoch = 1 if isinstance(cfg.train.val_interval, float) else cfg.train.val_interval,
    )

    return trainer