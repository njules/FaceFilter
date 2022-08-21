# TODO: implement batch processing

import argparse
from functools import reduce
import os
from dataset import CenterCropLongEdge
import matplotlib.pyplot as plt
from model import Encoder, Generator
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Tuple


NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def main(args):
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
    images = [Image.open(path) for path in args.images]
    image_tensors = [
        image_to_tensor(image, args.image_size).to(device) for image in images
    ]
    with torch.no_grad():
        styles = [
            encoder(image).view(image.shape[0], -1, args.latent_size)
            for image in image_tensors
        ]
        mixed_style = mix_styles(
            styles=styles,
            mixing_levels=args.mixing_levels,
            n_latent=generator.n_latent
        )
        mixed_image, _ = generator([mixed_style], input_is_latent=True)

    mixed_image = mixed_image[0, :, :, :]
    mixed_image = tensor_to_image(mixed_image)
    if args.out is not None:
        mixed_image.save(args.out)


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


def image_to_tensor(
    image: Image,
    image_size: int
) -> torch.Tensor:
    """
    Take a PIL image and return it as a torch.Tensor. Tensor has shape
    [1, 3, image_size, image_size]
    """
    transform = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(image_size, Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def tensor_to_image(image: torch.Tensor) -> Image:
    """Unnormalizes tensor and retrieves image

    Parameters
    ----------
    image: torch.Tensor
        The image tensor.

    Returns
    -------
    Image
        The PIL Image.
    """
    inverse_std = (1/NORMALIZE_STD[0], 1/NORMALIZE_STD[1], 1/NORMALIZE_STD[2])
    inverse_mean = (-NORMALIZE_MEAN[0], -NORMALIZE_MEAN[1], -NORMALIZE_MEAN[2])
    transform = transforms.Compose([
        transforms.Normalize((0, 0, 0), inverse_std),
        transforms.Normalize(inverse_mean, (1, 1, 1)),
        transforms.ToPILImage()
    ])
    image = transform(image[0, :, :, :])
    return image


def encode_image(
    encoder: Encoder,
    image: torch.Tensor,
    latent_size: int
) -> torch.Tensor:
    """Encode an image in latent space

    Parameters
    ----------
    encoder: Encoder
        The encoder model.

    image: torch.Tensor
        The image to be encoded. Has shape
        [batch_size, 3, image_size, image_size].

    latent_size: int
        Number of dimensions of the latent space.

    Returns
    -------
    torch.Tensor
        The image encoded in latent space. Has shape
        [batch_size, n_latent, latent_size].
    """
    return encoder(image).view(image.shape[0], -1, latent_size)


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


def plots_report():
    """Alternative main function to generate plots for project report
    """

    young_image_paths = [
        'data/small/young/000006.jpg',
        'data/small/young/000007.jpg',
        'data/small/young/000010.jpg',
        'data/small/young/000014.jpg'
    ]
    old_image_paths = [
        'data/small/old/000018.jpg',
        'data/small/old/000021.jpg',
        'data/small/old/000051.jpg',
        'data/small/old/000053.jpg',
        'data/small/old/000084.jpg'
    ]
    mixing_levels = {
        'top levels': [[8]],
        'high levels': [[6]],
        'mid levels': [[4]]
    }
    mixing_levels_alternating = {
        'alternating 1': [[5, 7, 9, 11]],
        'alternating 2': [[4, 6, 8, 10]],
        'alternating 3': [[8, 10]]
    }
    model = 'data/model075000.pt'
    mean = 'data/mean.pt'

    image_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_size = 512
    channel_multiplier = 2
    which_phi = 'lin2'
    stddev_group = 1
    n_mlp_g = 8
    encoder, generator = load_model(
        model_path=model,
        image_size=image_size,
        latent_size=latent_size,
        channel_multiplier=channel_multiplier,
        which_phi=which_phi,
        stddev_group=stddev_group,
        n_mlp_g=n_mlp_g,
        device=device
    )

    young_images = [Image.open(path) for path in young_image_paths]
    young_image_tensors = [
        image_to_tensor(image, image_size).to(device)
        for image in young_images
    ]
    young_images = [tensor_to_image(tensor) for tensor in young_image_tensors]
    young_image_latents = [
        encode_image(encoder=encoder, image=image, latent_size=latent_size)
        for image in young_image_tensors
    ]
    old_images = [Image.open(path) for path in old_image_paths]
    old_image_tensors = [
        image_to_tensor(image, image_size).to(device)
        for image in old_images
    ]
    old_images = [tensor_to_image(tensor) for tensor in old_image_tensors]
    old_image_latents = [
        encode_image(encoder=encoder, image=image, latent_size=latent_size)
        for image in old_image_tensors
    ]
    
    young_mean, old_mean = torch.load(mean)
    young_mean = young_mean.unsqueeze(0).to(device)
    old_mean = old_mean.unsqueeze(0).to(device)

    ######## plot ae reconstruction

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(young_images) + len(old_images)
    )
    axes[0][0].set_ylabel("original")
    axes[1][0].set_ylabel("reconstructed")
    
    images_and_styles = zip(
        young_images + old_images,
        young_image_latents + old_image_latents
    )
    for col_idx, (image, latent) in enumerate(images_and_styles):
        reconstructed_image = generate_image(
            generator=generator, style=latent
        )
        reconstructed_image = tensor_to_image(reconstructed_image)

        plot_original = axes[0][col_idx]
        plot_reconstructed = axes[1][col_idx]

        if col_idx < len(young_images):
            label = f"young {col_idx+1}"
        else:
            label = f"old {col_idx - len(young_images) + 1}"

        plot_original.set_xlabel(label)
        plot_original.xaxis.set_label_position('top')
        plot_original.set_xticks([])
        plot_original.set_yticks([])
        plot_original.imshow(image)
        plot_reconstructed.set_xticks([])
        plot_reconstructed.set_yticks([])
        plot_reconstructed.imshow(reconstructed_image)

    ######## plot stlye mixing

    # normal mixing

    fig, axes = plt.subplots(
        nrows=len(mixing_levels) + 1,
        ncols=len(old_images)
    )

    axes[0][0].set_ylabel("original")
    for col_idx, image in enumerate(old_images):
        axes[0][col_idx].set_xlabel(f"old {col_idx+1}")
        axes[0][col_idx].xaxis.set_label_position('top')
        axes[0][col_idx].set_xticks([])
        axes[0][col_idx].set_yticks([])
        axes[0][col_idx].imshow(image)

    for row_idx, levels in enumerate(mixing_levels, start=1):
        axes[row_idx][0].set_ylabel(levels)
        for col_idx, style in enumerate(old_image_latents):
            mixed_latent = mix_styles(
                styles=[young_image_latents[0], style],
                mixing_levels=mixing_levels[levels],
                n_latent=encoder.n_latent
            )
            image = generate_image(generator, mixed_latent)
            image = tensor_to_image(image)
            axes[row_idx][col_idx].set_xticks([])
            axes[row_idx][col_idx].set_yticks([])
            axes[row_idx][col_idx].imshow(image)

    # alternating mixing

    fig, axes = plt.subplots(
        nrows=len(mixing_levels_alternating) + 1,
        ncols=len(old_images)
    )

    axes[0][0].set_ylabel("original")
    for col_idx, image in enumerate(old_images):
        axes[0][col_idx].set_xlabel(f"old {col_idx+1}")
        axes[0][col_idx].xaxis.set_label_position('top')
        axes[0][col_idx].set_xticks([])
        axes[0][col_idx].set_yticks([])
        axes[0][col_idx].imshow(image)

    for row_idx, (names, levels) in enumerate(mixing_levels_alternating.items(), start=1):
        axes[row_idx][0].set_ylabel(names)
        for col_idx, style in enumerate(old_image_latents):
            mixed_latent = mix_styles(
                styles=[young_image_latents[0], style],
                mixing_levels=levels,
                n_latent=encoder.n_latent
            )
            image = generate_image(generator, mixed_latent)
            image = tensor_to_image(image)
            axes[row_idx][col_idx].set_xticks([])
            axes[row_idx][col_idx].set_yticks([])
            axes[row_idx][col_idx].imshow(image)

    ######## multi mixing TODO: this works quite well

    multi_mixing_latents = [
        [old_image_latents[0], old_image_latents[1]],
        [old_image_latents[0], old_image_latents[1], old_image_latents[2]],
        [old_image_latents[0], old_image_latents[1]]
    ]
    multi_mixing_levels = {
        'two styles': [[8, 10], [9, 11]],
        'three styles': [[9, 9], [10, 10], [8, 11]],
        'two alternating': [[8, 8], [10, 10]]
    }

    fig, axes = plt.subplots(
        nrows=len(young_images),
        ncols=len(multi_mixing_latents) + 1
    )
    axes[0][0].set_xlabel("original")
    axes[0][0].xaxis.set_label_position('top')
    for row_idx, (image, latent) in enumerate(zip(young_images, young_image_latents)):
        axes[row_idx][0].set_ylabel(f"young {row_idx+1}")
        axes[row_idx][0].set_xticks([])
        axes[row_idx][0].set_yticks([])
        axes[row_idx][0].imshow(image)
        for col_idx, (latents, levels) in enumerate(zip(multi_mixing_latents, multi_mixing_levels), start=1):
            if row_idx == 0:
                axes[0][col_idx].set_xlabel(levels)
                axes[0][col_idx].xaxis.set_label_position('top')
            mixed_latent = mix_styles(
                styles=[latent] + latents,
                mixing_levels=multi_mixing_levels[levels],
                n_latent=encoder.n_latent
            )
            image = generate_image(generator, mixed_latent)
            image = tensor_to_image(image)
            axes[row_idx][col_idx].set_xticks([])
            axes[row_idx][col_idx].set_yticks([])
            axes[row_idx][col_idx].imshow(image)

    ######## plot mean

    fig, axes = plt.subplots(nrows=1, ncols=2)
    young_mean = reduce(lambda mean, img: mean + img/len(young_image_latents), young_image_latents, 0)
    mean_young_image = generate_image(generator, young_mean)
    mean_young_image = tensor_to_image(mean_young_image)
    old_mean = reduce(lambda mean, img: mean + img/len(old_image_latents), old_image_latents, 0)
    mean_old_image = generate_image(generator, old_mean)
    mean_old_image = tensor_to_image(mean_old_image)
    axes[0].set_xlabel("mean young")
    axes[0].xaxis.set_label_position('top')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].imshow(mean_young_image)
    axes[1].set_xlabel("mean old")
    axes[1].xaxis.set_label_position('top')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].imshow(mean_old_image)

    ######## mean mixing

    fig, axes = plt.subplots(
        nrows=len(young_images),
        ncols=len(mixing_levels | mixing_levels_alternating) + 1
    )

    axes[0][0].set_xlabel("original")
    axes[0][0].xaxis.set_label_position('top')
    for row_idx, (image, style) in enumerate(zip(young_images, young_image_latents)):
        axes[row_idx][0].set_ylabel(f"young {row_idx+1}")
        axes[row_idx][0].set_xticks([])
        axes[row_idx][0].set_yticks([])
        axes[row_idx][0].imshow(image)
        for col_idx, (name, levels) in enumerate((mixing_levels | mixing_levels_alternating).items(), start=1):
            if row_idx == 0:
                axes[0][col_idx].set_xlabel(name)
                axes[0][col_idx].xaxis.set_label_position('top')
            mixed_latent = mix_styles(
                styles=[style, old_mean],
                mixing_levels=levels,
                n_latent=encoder.n_latent
            )
            image = generate_image(generator, mixed_latent)
            image = tensor_to_image(image)
            axes[row_idx][col_idx].set_xticks([])
            axes[row_idx][col_idx].set_yticks([])
            axes[row_idx][col_idx].imshow(image)

    plt.show()


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


if __name__ == '__main__':
    #args = parse_args()
    #main(args)
    with torch.no_grad():
        plots_report()
