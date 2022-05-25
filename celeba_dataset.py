import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from .base_dataset import BaseDataset

from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor


class CelebaDataset(BaseDataset):
    def __init__(self):
        super(CelebaDataset, self).__init__()
        self._dataset = None

    def name(self):
        return 'CelebaDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __getitem__(self, index: int):
        img, label = self._dataset[index]
        img = to_tensor(img)
        return {'A': img, 'A_paths': label}

    def initialize(self, opt):
        self._dataset = CelebA(root=opt.dataroot)

    def __len__(self):
        return len(self._dataset)
