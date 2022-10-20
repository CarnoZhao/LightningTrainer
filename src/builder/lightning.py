import pytorch_lightning as pl

from . import get

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = get("model", self.cfg.model)
        self.criterion = get("loss", self.cfg.loss)

        self.prepare_data()
        self.save_hyperparameters(self.cfg)

        self.metric = get("metric", self.cfg.metric, self.ds_valid)

    def prepare_data(self):
        self.data = get("data", self.cfg.data)
        (self.ds_train, self.ds_valid), (self.dl_train, self.dl_valid) = self.data

    def train_dataloader(self):
        return self.dl_train()

    def val_dataloader(self):
        return self.dl_valid()

    def configure_optimizers(self):
        optimizer, scheduler = get("optimizer", self, self.cfg.train)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        for k in loss:
            self.log("train_" + k, loss[k])
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        for k in loss:
            self.log("val_" + k, loss[k], prog_bar = True)
        return y, yhat

    def validation_step_end(self, output):
        outputs = self.metric.preprocess(output)
        return outputs

    def validation_epoch_end(self, outputs):
        for k, v in self.metric(outputs).items():
            self.log(k, v, prog_bar = True)

def get_lightning(*args, **kwargs):
    return Model(*args, **kwargs)