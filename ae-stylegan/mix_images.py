# TODO: implement batch processing

import argparse
from dataset import CenterCropLongEdge
from model import Encoder, Generator
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Tuple


NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, generator = load_model(
        model_path=args.model_path,
        image_size=args.image_size,
        latent_size=args.latent_size,
        channel_multiplier=args.channel_multiplier,
        which_phi=args.which_phi,
        stddev_group=args.stddev_group,
        n_mlp_g=args.n_mlp_g,
        device=device
    )
    images = read_images(
        args.images,
        image_size=args.image_size,
        device=device
    )
    with torch.no_grad():
        styles = encode_images(
            encoder=encoder,
            images=images,
            latent_size=args.latent_size
        )
        mixed_styles = mix_styles(
            styles=styles,
            mixing_levels=args.mixing_levels,
            n_latent=generator.n_latent
        )
        mixed_image = generate_image(
            generator=generator, style=mixed_styles
        )

    mixed_image = mixed_image[0, :, :, :]
    store_image(mixed_image, path=args.out)


def parse_args():
    parser = argparse.ArgumentParser()

    # example:
    #  python mix_images
    #   --image imageA imageB imageC \
    #   --mixing-levels 2 4 \
    #   --mixing-levels 6 \
    #
    # Take style of imageB at level 2 and 4, style of imageC at level 6 and
    # higher and imageA for every other level.

    # /home/mla_group_13/experiments_output/ae-stylegan_266738/ffhq_aegan_wplus_joint/weight/075000.pt

    parser.add_argument(
        '--images', nargs='+', type=str,
        help=
            "Paths to the images you want to mix. By default the latent"
            " representation of the first image will be taken at every style"
            " level. For every other image a '--mixing-levels' argument must"
            " be provided separately to specify at which style level the image"
            " should be mixed."
    )
    parser.add_argument(
        '--mixing-levels', nargs='+', type=int, action='append', default=[],
        help=
            "Indices to specify at which level styles should be mixed."
            " Provide a separate '--mixing-levels' argument for each image you"
            " want to mix. Argument number i corresponds to image number"
            " i+1. If the argument is a list of indices, take the style of"
            " that image at exactly those indices. If only a single index is"
            " provided, mix the image at that level and higher levels."
            " Priority is given to last provided argument. To mix the style of"
            " an image at exactly one style level provide that index twice."
            " There must be exactly n-1 '--mixing-levels' arguments for n"
            " provided images."
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help="File the mixed image should be stored in"
    )
    parser.add_argument(
        '--model-path', type=str,
        help="Path to the model weights"
    )
    parser.add_argument(
        "--image-size", type=int, default=128,
        help="Height and Width of the input images for the model"
    )
    parser.add_argument(
        "--latent-size", type=int, default=512,
        help="Size of the latent vector"
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
    parser.add_argument("--n_mlp_g", type=int, default=8)

    args = parser.parse_args()
    return args


def load_model(
    model_path: str,
    image_size: int,
    latent_size: int,
    channel_multiplier: int,
    which_phi: str,
    stddev_group: int,
    n_mlp_g: int,
    device: torch.device
) -> Tuple[Encoder, Generator]:

    encoder = Encoder(
        size=image_size,
        style_dim=latent_size,
        channel_multiplier=channel_multiplier,
        which_latent='w_plus',
        which_phi=which_phi,
        stddev_group=stddev_group,
        return_tuple=False
    ).to(device)
    generator = Generator(
        size=image_size,
        style_dim=latent_size,
        n_mlp=n_mlp_g,
        channel_multiplier=channel_multiplier
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage
    )
    encoder.load_state_dict(checkpoint['e_ema'])
    generator.load_state_dict(checkpoint['g_ema'])

    return encoder, generator


def read_images(
    paths: List[str],
    image_size: int,
    device: torch.device
) -> List[torch.Tensor]:
    """Read the images specified by paths and return them as tensors

    Parameters
    ----------
    paths: List[str]
        Paths to the image files.

    image_size: int
        Height and Width of the resulting tensor.

    device: torch.device
        Device on which the tensor should be stored.

    Returns
    -------
    List[torch.Tensor]
        List of images as tensors. Each tensor has shape
        [batch_size, 3, image_size, image_size].
    """
    transform = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(image_size, Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    images = []
    for path in paths:
        image = Image.open(path)
        image_tensor = transform(image).unsqueeze(0).to(device)
        images.append(image_tensor)
    return images


def encode_images(
    encoder: Encoder,
    images: List[torch.Tensor],
    latent_size: int
) -> List[torch.Tensor]:
    """Encode the images into latent space

    Parameters
    ----------
    encoder: Encoder
        The encoder model.

    images: List[torch.Tensor]
        The list of images to be encoded. Each image has shape
        [batch_size, 3, --image-size, --image-size].

    latent_size: int
        Number of dimensions of the latent space.

    Returns
    -------
    List[torch.Tensor]
        List of encoded images in latent space. Each latent vector has shape
        [batch_size, n_latent, --latent-size].
    """
    styles = []
    for image in images:
        style = encoder(image)
        style = style.view(style.shape[0], -1, latent_size)
        styles.append(style)
    return styles


def mix_styles(
    styles: List[torch.Tensor],
    mixing_levels: List[List[int]],
    n_latent: int
) -> torch.Tensor:
    """Mix the styles at the specified levels

    Parameters
    ----------
    styles : List[torch.Tensor]
        A list of latent representations of images. Each latent vector has
        shape [batch_size, n_latent, --latent-size].

    mixing levels : List[List[int]]
        Levels at which styles should be mixed. By default the first style is
        taken at every level. For every other style a List[int] of level
        indices must be provided. If only a single index is provided, mix at
        every level from the specified index and higher. Indices must be in the
        range of [0, n_latent).

    n_latent: int
        Number of levels of the styleGAN.

    Returns
    -------
    torch.Tensor
        Tensor describing the style that should be mixed in the generator at
        each level. Has shape [1, n_latent, --latent-size].
    """

    if len(styles) != len(mixing_levels) + 1:
        raise ValueError(
            "You need to specify the mixing levels for each provided image"
            " except the first one."
        )

    mixed_style = styles.pop(0).clone()
    for style, levels in zip(styles, mixing_levels):
        if len(levels) == 1:
            levels = range(levels[0], n_latent)
        for level in levels:
            mixed_style[:, level, :] = style[:, level, :]

    return mixed_style


def generate_image(
    generator: Generator,
    style: torch.Tensor
) -> torch.Tensor:
    """Generates an image from the latent style vector

    Parameters
    ----------
    generator: Generator
        The style-GAN model.

    style: torch.Tensor
        Latent style vector. Has shape [1, n_latent, --latent-size].

    Returns
    -------
    torch.Tensor
        Resulting image as tensor. Has shape
        [1, 3, --image-size, --image-size].
    """
    image, _ = generator([style], input_is_latent=True)
    return image


def store_image(image: torch.Tensor, path: str):
    """Stores an images that was loaded as a tensor with the read_images function

    Parameters
    ----------
    image: torch.Tensor
        The image tensor.

    path: str
        The file the image should be saved in.
    """
    inverse_std = (1/NORMALIZE_STD[0], 1/NORMALIZE_STD[1], 1/NORMALIZE_STD[2])
    inverse_mean = (-NORMALIZE_MEAN[0], -NORMALIZE_MEAN[1], -NORMALIZE_MEAN[2])
    transform = transforms.Compose([
        transforms.Normalize((0, 0, 0), inverse_std),
        transforms.Normalize(inverse_mean, (1, 1, 1)),
        transforms.ToPILImage()
    ])
    image = transform(image)
    if path is not None:
        image.save(path)


if __name__ == '__main__':
    main()
