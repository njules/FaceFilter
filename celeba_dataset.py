import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from .base_dataset import BaseDataset

from torchvision.datasets import CelebA


class CelebaDataset(CelebA, BaseDataset):
    def __init__(self):
        super(CelebaDataset, self).__init__()

    def name(self):
        return 'CelebaDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __getattribute__(self, name: str):
        img, labels = super().__getattribute__(name)
        return img, labels[-1]

    def initialize(self, opt):
        pass
