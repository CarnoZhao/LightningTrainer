import os
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--method", dest = "method", type = str, default = "last")
args = parser.parse_args()

tags = {
    "whole": [
        # "",
        "sz224"
        # "30e"
    ],
    "hos": [
        # "",
        "sz224",
        # "30e"
    ],
    "fed": [
        # "",
        # "blk6",
        # "blk6_ft10",
        "sz224_blk6_ft10"
    ]
}
method = args.method

def get_whole(tag):
    aucs = []
    for fold in range(4):
        if os.path.exists(f"./logs/{tag}/f{fold}/metrics.csv"):
            auc = pd.read_csv(f"./logs/{tag}/f{fold}/metrics.csv")
            auc = np.array(auc[~auc.val_auc.isna()][[f"val_{hos}_auc" for hos in range(4)]])
            auc = auc.max(0) if method == "max" else auc[-1]
            aucs.append(auc)
        else:
            aucs.append(np.array([np.nan] * 4))
    return np.stack(aucs).mean(0) * 100

def get_inhos(tag):
    aucs = np.zeros((4, 4))
    aucs[:,:] = np.nan
    for fold in range(4):
        for hos in range(4):
            f = f"./logs/{tag}/hos{hos}/f{fold}/metrics.csv"
            if os.path.exists(f):
                auc = pd.read_csv(f)
                auc = np.array(auc[~auc.val_auc.isna()][f"val_auc"])
                auc = auc.max() if method == "max" else auc[-1]
                aucs[fold][hos] = auc
    return np.stack(aucs).mean(0) * 100

def get(k, tag):
    tag = k if not tag else f"{k}_{tag}"
    if k == "whole":
        ret = get_whole(tag)
    else:
        ret = get_inhos(tag)
    return np.concatenate([ret, ret.mean(keepdims = True)])

df = []
for k in tags:
    for tag in tags[k]:
        for i, value in enumerate(get(k, tag)):
            df.append([k, tag, str(i) if i != 4 else "mean", value])

df = pd.DataFrame(df, columns = ["version", "tag", "hos_id", "auc_ao4"]).set_index(["tag", "hos_id"])
df.auc_ao4.iloc[5:] -= np.tile(np.array(df.auc_ao4.iloc[:5]), len(df) // 5 - 1)

tab = []
for g, d in df.groupby("version", as_index = False, sort = False):
    d = d.drop(columns = ["version"])
    d.columns = [g]
    tab.append(d)
tab = pd.concat(tab, axis = 1).round(4).reset_index().set_index("tag")
# print(tab)

from rich.console import Console
from rich.table import Table

table = Table(row_styles = ["", "dim", "", "dim", "bold"])

table.add_column("tag", justify = "middle", style = "cyan")
table.add_column("hos_id", justify = "right", style = "magenta")
table.add_column("whole", justify = "right", style = "red")
table.add_column("hos", justify = "right", style = "green")
table.add_column("fed", justify = "right", style = "blue")

pre_idx = None
for row_idx, row in tab.iterrows():
    if row_idx != pre_idx:
        table.add_section()
        pre_idx = row_idx
    else:
        row_idx = ""
    table.add_row(row_idx, row.hos_id, *["-" if np.isnan(x) else f"{x:.2f}" for k, x in row.items() if k != "hos_id"])


console = Console()
console.print(table)


