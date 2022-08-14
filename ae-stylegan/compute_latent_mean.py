# TODO: make script agnostic to provided split

import argparse
from dataset import get_image_dataset
from model import Encoder
from train import data_sampler
from typing import Tuple
import torch
from torch.utils import data


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(
        model_path=args.model_path,
        image_size=args.size,
        latent_size=args.latent_size,
        channel_multiplier=args.channel_multiplier,
        which_phi=args.which_phi,
        stddev_group=args.stddev_group,
        device=device
    )
    dataloader_young, dataloader_old = load_data(
        args=args,
        path=args.data_path,
        batch_size=args.batch_size
    )
    with torch.no_grad():
        print("computing young mean")
        young_mean = compute_latent_mean(
            encoder=encoder,
            dataloader=dataloader_young,
            latent_size=args.latent_size,
            device=device
        )
        print("computing old mean")
        old_mean = compute_latent_mean(
            encoder=encoder,
            dataloader=dataloader_old,
            latent_size=args.latent_size,
            device=device
        )
    torch.save((young_mean, old_mean), args.out)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", type=str,
        help="Path to the celeba dataset with young/old splits"
    )
    # TODO: default
    parser.add_argument(
        '--out', type=str,
        help="File the computed mean should be stored in"
    )
    parser.add_argument(
        '--model-path', type=str,
        help="Path to the model weights"
    )
    parser.add_argument(
        "--size", type=int, default=128,
        help="Height and Width of the input images for the model"
    )
    parser.add_argument(
        "--latent-size", type=int, default=512,
        help="Size of the latent vector"
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--channel_multiplier", type=int, default=2,
        help="Channel multiplier factor for the model. config-f = 2, else = 1"
    )
    parser.add_argument(
        "--which_phi", type=str, default='lin2',
        help="Specify the final layer of the encoder"
    )
    parser.add_argument("--stddev_group", type=int, default=1)

    args = parser.parse_args()
    return args


def load_encoder(
    model_path: str,
    image_size: int,
    latent_size: int,
    channel_multiplier: int,
    which_phi: str,
    stddev_group: int,
    device: torch.device
) -> Tuple[Encoder]:
    encoder = Encoder(
        size=image_size,
        style_dim=latent_size,
        channel_multiplier=channel_multiplier,
        which_latent='w_plus',
        which_phi=which_phi,
        stddev_group=stddev_group,
        return_tuple=False
    ).to(device)
    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage
    )
    encoder.load_state_dict(checkpoint['e_ema'])

    return encoder


def load_data(
    args: argparse.Namespace,
    path: str,
    batch_size: int,
    dataset: str = 'imagefolder',
) -> Tuple[data.DataLoader, data.DataLoader]:
    # Create the training datasets
    dataset = get_image_dataset(args, dataset, path, train=True)

    young_indices = [
        idx for idx, _cls in enumerate(dataset.targets) if _cls == 1
    ]
    old_indices = [
        idx for idx, _cls in enumerate(dataset.targets) if _cls == 0
    ]

    dataset_young = data.Subset(dataset, young_indices)
    dataset_old = data.Subset(dataset, old_indices)

    loader_young = data.DataLoader(
        dataset_young,
        batch_size=batch_size,
        sampler=data_sampler(dataset_young, shuffle=True, distributed=False),
        drop_last=True
    )
    loader_old = data.DataLoader(
        dataset_old,
        batch_size=batch_size,
        sampler=data_sampler(dataset_old, shuffle=True, distributed=False),
        drop_last=True
    )

    return loader_young, loader_old


def compute_latent_mean(
    encoder: Encoder,
    dataloader: data.DataLoader,
    latent_size: int,
    device: torch.device
) -> torch.Tensor:
    num_samples = len(dataloader)
    mean = 0
    for batch, _ in dataloader:
        batch = batch.to(device)
        styles = encoder(batch)
        styles = styles.view(styles.shape[0], -1, latent_size)
        mean += styles.sum(dim=0) / num_samples
    return mean


if __name__ == '__main__':
    main()
