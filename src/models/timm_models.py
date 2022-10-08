import torch
import torch.nn as nn

import timm

class TIMMModel(nn.Module):
    def __init__(self, 
                model_name = "resnet18",
                pretrained = True,
                in_chans = 3,
                num_classes = 1,
                **kwargs
                ):
        super().__init__()
        self.model = timm.create_model(model_name, 
            pretrained = pretrained,
            in_chans = in_chans,
            num_classes = num_classes)
        
    def forward(self, x):
        return self.model(x)