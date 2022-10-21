from ..builder.registry import REGISTRY
METRIC = REGISTRY["METRIC"]

def get_metric(cfg, dataset):
    """
    def metric_preprocess(val_step_output):
        return output

    def metric(outputs):
        return {"val_metic": ...}
    """
    cfg = cfg.copy()
    if isinstance(cfg, str):
        return METRIC[cfg](dataset = dataset)
    return METRIC[cfg.pop("type")](dataset = dataset, **cfg)
