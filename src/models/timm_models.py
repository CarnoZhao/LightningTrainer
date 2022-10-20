import torch
import torch.nn as nn

import timm

from ..builder.registry import register

@register(name = "MODEL")
class TIMMModel(nn.Module):
    def __init__(self, 
                model_name = "resnet18",
                pretrained = True,
                in_chans = 3,
                num_classes = 1,
                drop_rate = 0.0,
                freeze_layers = 0,
                **kwargs
                ):
        super().__init__()
        self.model = timm.create_model(model_name, 
            pretrained = pretrained,
            in_chans = in_chans,
            num_classes = num_classes,
            drop_rate = drop_rate)

        self.freeze(model_name, freeze_layers)

    def freeze(self, model_name, freeze_layers):
        if freeze_layers == 0:
            pass
        if "tf_efficientnet" in model_name:
            self.model.conv_stem.requires_grad_(False)
            self.model.bn1.requires_grad_(False)
            self.model.act1.requires_grad_(False)
            for i in range(freeze_layers):
                self.model.blocks[i].requires_grad_(False)
        else:
            raise NotImplementedError()
        
    def forward(self, x):
        return self.model(x)