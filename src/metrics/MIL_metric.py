import torch
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..builder.registry import register

def roc_auc_score_uni(y_true, y_score):
    if len(np.unique(y_true)) != 2:
        return 0
    else:
        return roc_auc_score(y_true, y_score)

@register(name = "METRIC")
class MILMetric(object):
    def __init__(self, 
                metrics = [],
                softmax = True,
                force_binary = True,
                dataset = None,
                bag_by = None,
                group_by = None,
                func = "max",
                **kwargs):
            self.metrics_dict = {
                "acc": accuracy_score,
                "pre": precision_score,
                "rec": recall_score,
                "auc": roc_auc_score_uni,
                "f1": f1_score
            }
            self.metrics = metrics if metrics else list(self.metrics_dict.keys())
            self.keep_logits = {"auc"}
            assert all([_ in self.metrics_dict for _ in self.metrics])
            self.softmax = softmax
            self.force_binary = force_binary
            self.bag = np.array(dataset.df[bag_by])
            self.group = np.array(dataset.df[group_by]) if group_by is not None else None
            self.func = func

    def preprocess(self, output):
        return output

    def __call__(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).detach()
        if self.softmax:
            yhat = yhat.softmax(1).cpu().numpy()
        else:
            yhat = yhat.sigmoid().cpu().numpy()
        
        ret = {}
        for metric in self.metrics:
            if metric not in self.keep_logits:
                yh = yhat.argmax(1) if self.softmax else yhat.round()
            else:
                yh = yhat[:,1] if self.softmax and self.force_binary else yhat
            d = pd.DataFrame({"y": y, "yh": yh, "b": self.bag[:len(y)]}).groupby("b").agg(self.func)
            val = self.metrics_dict[metric](np.array(d.y), np.array(d.yh))
            ret["val_" + metric] = val

        if self.group is not None:
            for metric in self.metrics:
                if metric not in self.keep_logits:
                    yh = yhat.argmax(1) if self.softmax else yhat.round()
                else:
                    yh = yhat[:,1] if self.softmax and self.force_binary else yhat
                d = pd.DataFrame({"y": y, "yh": yh, "b": self.bag[:len(y)], "g": self.group[:len(y)]}).groupby("b").agg(self.func)
                for g, dg in d.groupby("g"):
                    val = self.metrics_dict[metric](np.array(dg.y), np.array(dg.yh))
                    ret[f"val_{g}_" + metric] = val

        return ret

