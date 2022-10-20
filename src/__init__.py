import os

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import cv2
cv2.setNumThreads(4)

from .dataset import *
from .losses import *
from .metrics import *
from .models import *
from .builder import *
from .federated import *