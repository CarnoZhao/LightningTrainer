from .dice_loss import DiceLoss
from .bce_loss import BCEWithIgnoreLoss

losses = {
    "DiceLoss": DiceLoss,
    "BCEWithIgnoreLoss": BCEWithIgnoreLoss
}