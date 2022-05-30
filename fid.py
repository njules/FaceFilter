import os
import argparse

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
from torchvision.models import inception_v3
import torchvision.transforms as T

from scipy import linalg

from time import perf_counter


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, *, transforms) -> None:
        super(ImageDataset, self).__init__()
        self.path = path
        self.images = os.listdir(self.path)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.images[index]))
        if self.transforms:
            image = self.transforms(image)
        return image


def create_dataloader(path: str, batch_size=64):
    dataset = ImageDataset(path, transforms=T.Compose([T.Resize((299, 299)), T.ToTensor()]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
        # pin_memory=True
    )


def compute_fid(generated_images: str, target_images: str, batch_size: int, device: str):
    model = inception_v3(pretrained=True)
    # drop dropout and fc layers
    model.dropout = nn.Identity()
    model.fc = nn.Identity()

    model = model.to(device)
    model.train(False)

    # create the dataloader
    gen_dataloader = create_dataloader(generated_images, batch_size)
    tgt_dataloader = create_dataloader(target_images, batch_size)

    total_fid, n = 0, 0

    for i, (gen_data, tgt_data) in enumerate(zip(gen_dataloader, tgt_dataloader)):
        batch_start = perf_counter()
        print(f"Starting batch {i}")

        gen_data = gen_data.to(device)
        tgt_data = tgt_data.to(device)

        with torch.inference_mode():
            gen_act = model(gen_data)
            tgt_act = model(tgt_data)

        # mean and covariance of the activation maps
        mu1, sigma1 = gen_act.mean(dim=0), gen_act.cov()
        mu2, sigma2 = tgt_act.mean(dim=0), tgt_act.cov()

        ssdiff = torch.sum((mu1 - mu2) ** 2.0)
        # covmean = torch.linalg.matrix_power(sigma1 @ sigma2, 0.5)
        covmean = linalg.sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy())
        covmean = torch.from_numpy(covmean).to(ssdiff.device)

        if torch.is_complex(covmean):
            covmean = covmean.real

        fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
        print(f"[Epoch {i:3d}]: FID={fid:.5f} ({perf_counter() - batch_start:.3f} s)")

        n += gen_data.shape[0]
        total_fid += float(fid) * gen_data.shape[0]

    avg_fid = total_fid / n
    print(f"Average fid is {avg_fid:.3f}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--generated-images", type=str, required=True)
    args_parser.add_argument("--target-images", type=str, required=True)
    args_parser.add_argument("--batch-size", type=int, default=64)
    args_parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")

    args = args_parser.parse_args()
    compute_fid(args.generated_images, args.target_images, args.batch_size, args.device)
