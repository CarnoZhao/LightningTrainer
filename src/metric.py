from .metrics import metrics as registry

def get_metric(cfg, dataset):
    """
    def metric_preprocess(val_step_output):
        return output

    def metric(outputs):
        return {"val_metic": ...}
    """
    cfg = cfg.copy()
    if isinstance(cfg, str):
        return registry[cfg](dataset = dataset)
    return registry[cfg.pop("type")](dataset = dataset, **cfg)
