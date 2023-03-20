from __future__ import print_function, division

import numpy as np
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ConvertImageDtype


class ArrayToTensor(object):
    def __call__(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).type(torch.float32)


class RGBTensorToArray(object):
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.permute(1,2,0).detach().cpu().numpy()


class TensorToArray(object):
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().numpy()


IMG_TRANSFORMS = Compose([
        ToTensor(),
        ConvertImageDtype(torch.float32),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])


IMG_INV_TRANSFORMS = Compose([
        Normalize(torch.Tensor([0.0, 0.0, 0.0]), torch.Tensor([2.0, 2.0, 2.0])),
        Normalize(torch.Tensor([-0.5, -0.5, -0.5]), torch.Tensor([1.0, 1.0, 1.0])),
        RGBTensorToArray()
    ])


DEPTH_TRANSFORMS = Compose([
        ArrayToTensor()
    ])


DEPTH_INV_TRANSFORMS = Compose([
        TensorToArray()
    ])