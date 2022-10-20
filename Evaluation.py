import os
import argparse
import warnings
import numpy as np
import pandas as pd
from rich.table import Table
from rich.console import Console
from omegaconf import OmegaConf
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--config", dest = "config", type = str)
args = parser.parse_args()

def get_whole(tag, method):
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

def get_inhos(tag, method):
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

def get(k, tag, method):
    tag = k if not tag else f"{k}_{tag}"
    if k.startswith("whole"):
        ret = get_whole(tag, method)
    else:
        ret = get_inhos(tag, method)
    return np.concatenate([ret, ret.mean(keepdims = True)])

if __name__ == "__main__":
    config = OmegaConf.load(args.config)
    for exp_name, tags in config.items():
        method = tags.pop("method", "last")
        show = tags.pop("show", "rel")
        df = []
        for k in tags:
            for tag in tags[k]:
                if OmegaConf.is_list(tag):
                    full_tag = "_".join(tag)
                    main_tag = tag[0]
                else:
                    full_tag = tag
                    main_tag = tag
                for i, value in enumerate(get(k, full_tag, method)):
                    df.append([k, main_tag, str(i) if i != 4 else "mean", value])

        df = pd.DataFrame(df, columns = ["version", "tag", "hos_id", "auc_ao4"]).set_index(["tag", "hos_id"])
        if show == "rel":
            df.auc_ao4.iloc[5:] -= np.tile(np.array(df.auc_ao4.iloc[:5]), len(df) // 5 - 1)

        tab = []
        for g, d in df.groupby("version", as_index = False, sort = False):
            d = d.drop(columns = ["version"])
            d.columns = [g]
            tab.append(d)
        tab = pd.concat(tab, axis = 1).round(4).reset_index().set_index("tag")

        table = Table(title = exp_name, row_styles = ["", "dim", "", "dim", "bold"])

        table.add_column("tag", justify = "middle", style = "cyan")
        table.add_column("hos_id", justify = "right", style = "magenta")
        for i, col in enumerate(tab.columns[1:]):
            table.add_column(col.replace("_sz224", ""), justify = "right", style = ["red", "green", "blue"][i % 3])

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


