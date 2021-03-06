import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold, KFold

from torch.utils.data import DataLoader

# from .dataset import YourDataset

def get_data(cfg):
    data_type = cfg.type 
    fold = cfg.get("fold", 0)
    num_folds = cfg.get("num_folds", 5)
    batch_size = cfg.get("batch_size", 32) 
    stratified_by = cfg.get("stratified_by", None) 
    group_by = cfg.get("group_by", None)
    dataset_cfg = cfg.get("dataset", {})
    
    if data_type is not None:
        DataClass = eval(data_type)
    else:
        raise NotImplementedError()

    df = DataClass.prepare(**dataset_cfg)

    if stratified_by is not None or group_by is not None:
        split = StratifiedGroupKFold(num_folds, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, y = df[stratified_by], groups = df[group_by]))[fold]
    elif stratified_by is not None:
        split = StratifiedKFold(num_folds, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, y = df[stratified_by]))[fold]
    elif stratified_by is not None or group_by is not None:
        split = GroupKFold(num_folds)
        train_idx, valid_idx = list(split.split(df, groups = df[group_by]))[fold]
    elif "fold" in df.columns:
        train_idx = np.where(df.fold != fold)[0]
        valid_idx = np.where(df.fold == fold)[0]
    else:
        split = KFold(num_folds, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df))[fold]

    if fold == -1:
        train_idx = np.concatenate([train_idx, valid_idx])
    
    train_cfg = {
        "df": df.loc[train_idx].reset_index(drop = True),
        "phase": "train",
        **dataset_cfg         
    }
    valid_cfg = {
        "df": df.loc[valid_idx].reset_index(drop = True),
        "phase": "val",
        **dataset_cfg         
    }

    ds_train = DataClass(**train_cfg)
    ds_valid = DataClass(**valid_cfg)

    def dl_train(shuffle = True, drop_last = True, num_workers = 8):
        return DataLoader(ds_train, batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)

    def dl_valid(shuffle = False, num_workers = 8):
        return DataLoader(ds_valid, batch_size, shuffle = shuffle, num_workers = num_workers)

    return (ds_train, ds_valid), (dl_train, dl_valid)
