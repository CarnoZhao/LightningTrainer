import os
import cv2
import glob
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .builder import build_trans
from .img_dataset import ImgData

class NasoData(ImgData):
    @staticmethod
    def prepare(data_dir = "images", use_hos_id = None, **dataset_cfg):
        df = pd.read_csv("./data/train.csv")
        df.image_file = df.image_file.apply(lambda x: os.path.join("data", data_dir, x))
        if use_hos_id is not None:
            df = df[df.hos_id == use_hos_id].reset_index(drop = True)
        return df

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        if row["redirect"] != -1:
            idx = np.random.randint(len(self.value_df[row["redirect"]]))
            row = self.value_df[row["redirect"]].loc[idx]

        image_file = row["image_file"]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = int(row["label"])

        if self.trans is not None:
            aug = self.trans(image = img)
            img = aug['image']

        return img, label