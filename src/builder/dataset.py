from ..builder.registry import REGISTRY
AUG = REGISTRY["AUG"]
DATASET = REGISTRY["DATASET"]
SAMPLER = REGISTRY["SAMPLER"]

import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold, KFold

import torch
from torch.utils.data import DataLoader
import torchsampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_trans(trans, T = A, trans_args = {}):
    if isinstance(T, str):
        T = eval(T)
    if trans is None: return None
    trans = deepcopy(trans)
    if OmegaConf.is_list(trans) or isinstance(trans, list):
        return T.Compose([get_trans(_, T = T) for _ in trans], **trans_args)
    elif trans["type"] in ("Compose", "OneOf", "SomeOf"):
        return getattr(T, trans.pop("type"))([get_trans(_, T = T) for _ in trans.pop("transforms")], **trans)
    elif trans["type"] == "ToTensorV2":
        trans.pop("type")
        return ToTensorV2(**trans)
    elif trans["type"] in AUG:
        return AUG[trans.pop("type")](**trans)
    else:
        return getattr(T, trans.pop("type"))(**trans)

def get_data(cfg):
    cfg = cfg.copy()
    data_type = cfg.type 
    fold = cfg.get("fold", 0)
    num_workers = cfg.get("num_workers", 16)
    num_folds = cfg.get("num_folds", 5)
    batch_size = cfg.get("batch_size", 32) 
    stratified_by = cfg.get("stratified_by", None) 
    group_by = cfg.get("group_by", None)
    fold_by = cfg.get("fold_by", "fold")
    dataset_cfg = cfg.get("dataset", {})
    sampler_cfg = cfg.get("sampler", None)

    df = DATASET[data_type].prepare(**dataset_cfg)

    if stratified_by is not None and group_by is not None:
        split = StratifiedGroupKFold(num_folds, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, y = df[stratified_by], groups = df[group_by]))[fold]
    elif stratified_by is not None:
        split = StratifiedKFold(num_folds, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, y = df[stratified_by]))[fold]
    elif group_by is not None:
        split = GroupKFold(num_folds)
        train_idx, valid_idx = list(split.split(df, groups = df[group_by]))[fold]
    elif fold_by in df.columns:
        train_idx = np.where(df.fold != fold)[0]
        valid_idx = np.where(df.fold == (fold if fold != -1 else 0))[0]
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

    ds_train = DATASET[data_type](**train_cfg)
    ds_valid = DATASET[data_type](**valid_cfg)

    if sampler_cfg is not None:
        sampler = {"sampler": SAMPLER[sampler_cfg.pop("type")](data_source = ds_train, **sampler_cfg)}
    elif ds_train.balance_key:
        sampler = {"sampler": torchsampler.ImbalancedDatasetSampler(ds_train)}
    else:
        sampler = {"shuffle": True}

    def dl_train(shuffle = True, drop_last = True, num_workers = num_workers):
        return DataLoader(ds_train, 
                        batch_size, 
                        drop_last = drop_last, 
                        num_workers = num_workers,
                        pin_memory = True,
                        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
                        **sampler)

    def dl_valid(shuffle = False, num_workers = num_workers):
        return DataLoader(ds_valid, 
                        batch_size, 
                        shuffle = shuffle, 
                        num_workers = num_workers,
                        pin_memory = True,)

    return (ds_train, ds_valid), (dl_train, dl_valid)
