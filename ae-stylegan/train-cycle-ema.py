import argparse
from json import encoder
import os
import itertools as it

import numpy as np
import torch
import torch.nn as nn
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, utils
from torch.utils import data
from torch.utils.checkpoint import checkpoint
import functools as ft
from tqdm import tqdm
import util
from calc_inception import load_patched_inception_v3
import pickle
import pdb
st = pdb.set_trace

from dataset import get_image_dataset
from distributed import get_rank, synchronize

from fid import extract_feature_from_samples, calc_fid, extract_feature_from_reconstruction
from train import load_real_samples, requires_grad, data_sampler, accumulate, d_r1_loss


#@torch.no_grad()
def train(args, loader_young, loader_old, 
        test_loader_young, test_loader_old,
        generator_young2old, generator_old2young, 
        generator_young2old_ema, generator_old2young_ema, 
        encoder_young, encoder_old,
        encoder_young_ema, encoder_old_ema,
        discriminator_young, discriminator_old,
        vggnet, g_young2old_optim, g_old2young_optim, 
        e_young_optim, e_old_optim, 
        d_young_optim, d_old_optim, 
        device):

    inception = mean_latent = None
    real_mean_young = real_cov_young = None
    real_mean_old = real_cov_old = None

    if args.eval_every > 0 and len(args.which_metric) > 0:
        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()

        with open(args.inception_young, "rb") as f:
            embeds = pickle.load(f)
            real_mean_young, real_cov_young = embeds["mean"], embeds["cov"]
        with open(args.inception_old, "rb") as f:
            embeds = pickle.load(f)
            real_mean_old, real_cov_old = embeds["mean"], embeds["cov"]

    if get_rank() == 0:
        if args.eval_every > 0:
            with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")
        if args.log_every > 0:
            with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")

    loader_young, loader_old = it.cycle(loader_young), it.cycle(loader_old)
    test_loader_young, test_loader_old = it.cycle(test_loader_young), it.cycle(test_loader_old)

    sample_young = load_real_samples(args, test_loader_young).to(device)
    sample_old = load_real_samples(args, test_loader_old).to(device)

    train_sample_young = load_real_samples(args, loader_young).to(device)
    train_sample_old = load_real_samples(args, loader_old).to(device)
    
    for idx in range(args.iter): #pbar:
        i = idx

        if i > args.iter:
            print("Done!")
            break
       
        if args.debug: util.seed_everything(i)

        real_young_imgs = next(loader_young)[0].to(device)
        real_old_imgs = next(loader_old)[0].to(device)

        e_young_optim.zero_grad()
        g_young2old_optim.zero_grad()
        d_old_optim.zero_grad()
        e_old_optim.zero_grad()
        g_old2young_optim.zero_grad()
        d_young_optim.zero_grad()

        loss = 0

        ##################################
        #  Step 0: train discriminators  #
        ##################################

        requires_grad(discriminator_young, True)
        requires_grad(discriminator_old, True)

        # train discriminator old
        real_pred = discriminator_old(real_old_imgs)
        fake_pred = discriminator_old(real_young_imgs)

        (F.softplus(fake_pred).mean() + F.softplus(-real_pred).mean()).backward()
        d_old_optim.step()

        # train discriminator young
        real_pred = discriminator_young(real_young_imgs)
        fake_pred = discriminator_young(real_old_imgs)

        d_young_loss_fake = F.softplus(fake_pred).mean()
        d_young_loss_real = F.softplus(-real_pred).mean()

        (F.softplus(fake_pred).mean() + F.softplus(-real_pred).mean()).backward()
        d_young_optim.step()


        d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
        if d_regularize:
            real_old_imgs.requires_grad = True
            real_young_imgs.requires_grad = True

            # regularize discriminator old
            real_pred = discriminator_old(real_old_imgs)
            r1_loss = d_r1_loss(real_pred, real_old_imgs)

            discriminator_old.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_old_optim.step()

            # regularize discriminator young
            real_pred = discriminator_young(real_young_imgs)
            r1_loss = d_r1_loss(real_pred, real_young_imgs)

            discriminator_young.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_young_optim.step()

            real_old_imgs.requires_grad = False
            real_young_imgs.requires_grad = False
            real_old_imgs.grad = None
            real_young_imgs.grad = None

        ###################################################
        #  Part 1: Real Young -> Rec. Old -> Young again  #
        ###################################################

        # train the first half of the network
        requires_grad(encoder_young, True)
        requires_grad(generator_young2old, True)
        requires_grad(discriminator_old, False)

        requires_grad(encoder_old, False)
        requires_grad(generator_old2young, False)
        requires_grad(discriminator_young, False)

        # encode the young images into the latent space
        latent_young_real, _ = encoder_young(real_young_imgs)
        # reconstruct the images in the second domain
        rec_young2old, _ = checkpoint(ft.partial(generator_young2old, input_is_latent=True), latent_young_real)
        # encode the rec. old images into the latent space 
        latent_old_rec, _ = checkpoint(encoder_old_ema, rec_young2old)
        # reconstruct the images in the original domain
        rec_young, _ = checkpoint(ft.partial(generator_old2young_ema, input_is_latent=True), latent_old_rec)
        
        # compute the prediction for the reconstructed images and the real ones
        # real_pred = discriminator_old(real_old_imgs)
        fake_pred = checkpoint(discriminator_old, rec_young2old)

        adv_loss = F.softplus(-fake_pred).mean()

        # Reconstruction loss
        pix_loss, vgg_loss = 0, 0

        if args.lambda_pix > 0:
            assert rec_young.requires_grad
            # pixel to pixel difference of reconstructed and real image
            if args.pix_loss == 'l2':
                pix_loss = torch.mean((rec_young - real_young_imgs) ** 2)
            elif args.pix_loss == 'l1':
                pix_loss = F.l1_loss(rec_young, real_young_imgs)
        
        if args.lambda_vgg > 0:
            # use a vgg network to compute the difference between the features of real
            # and reconstructed images
            vgg_loss = torch.mean((vggnet(real_young_imgs) - vggnet(rec_young)) ** 2)

        (args.lambda_adv * adv_loss + args.lambda_pix * pix_loss + args.lambda_vgg * vgg_loss).backward()
        e_young_optim.step()
        g_young2old_optim.step()
        d_old_optim.step()

        print(f"[{i:4d}/{args.iter:4d}] First step: adv_loss={adv_loss:.5f}, pix_loss={pix_loss:.5f}")
        
        #################################################
        #  Part 2: Real Old -> Rec. Young -> Old again  #
        #################################################

        # train the second half of the network
        requires_grad(encoder_young, False)
        requires_grad(generator_young2old, False)
        requires_grad(discriminator_old, False)

        requires_grad(encoder_old, True)
        requires_grad(generator_old2young, True)
        requires_grad(discriminator_young, False)

        # encode the old images into the latent space
        latent_old_real, _ = encoder_old(real_old_imgs)
        # reconstruct the images in the second domain
        rec_old2young, _ = checkpoint(ft.partial(generator_old2young, input_is_latent=True), latent_old_real)
        # encode the rec. young images into the latent space 
        latent_young_rec, _ = checkpoint(encoder_young_ema, rec_old2young)
        # reconstruct the images in the original domain
        rec_old, _ = checkpoint(ft.partial(generator_young2old_ema, input_is_latent=True), latent_young_rec)
        
        # compute the prediction for the reconstructed images and the real ones
        # real_pred = discriminator_young(real_young_imgs)
        fake_pred = checkpoint(discriminator_young, rec_old2young)

        adv_loss = F.softplus(-fake_pred).mean()

        # Reconstruction loss
        pix_loss, vgg_loss = 0, 0

        if args.lambda_pix > 0:
            # pixel to pixel difference of reconstructed and real image
            if args.pix_loss == 'l2':
                pix_loss = torch.mean((rec_old - real_old_imgs) ** 2)
            elif args.pix_loss == 'l1':
                pix_loss = F.l1_loss(rec_old, real_old_imgs)
        
        if args.lambda_vgg > 0:
            # use a vgg network to compute the difference between the features of real
            # and reconstructed images
            vgg_loss = torch.mean((vggnet(real_old_imgs) - vggnet(rec_old)) ** 2)

        (args.lambda_adv * adv_loss + args.lambda_pix * pix_loss + args.lambda_vgg * vgg_loss).backward()
        e_old_optim.step()
        g_old2young_optim.step()
        d_young_optim.step()

        print(f"[{i:4d}/{args.iter:4d}] Second step: adv_loss={adv_loss:.5f}, pix_loss={pix_loss:.5f}")

        # Update EMA
        ema_nimg = args.ema_kimg * 1000
        if args.ema_rampup is not None:
            ema_nimg = min(ema_nimg, i * args.batch * args.ema_rampup)
        accum = 0.5 ** (args.batch / max(ema_nimg, 1e-8))
        accumulate(generator_young2old_ema, generator_young2old, accum)
        accumulate(generator_old2young_ema, generator_old2young, accum)
        accumulate(encoder_old_ema, encoder_old, accum)
        accumulate(encoder_young_ema, encoder_young, accum)
     

        #####################
        #  Backpropagation  #
        #####################

        del real_young_imgs
        del real_old_imgs

        #nn.utils.clip_grad_value_(encoder_young.parameters(), clip_value=1.0)
        #nn.utils.clip_grad_value_(encoder_old.parameters(), clip_value=1.0)
        #nn.utils.clip_grad_value_(generator_old2young.parameters(), clip_value=1.0)
        #nn.utils.clip_grad_value_(generator_young2old.parameters(), clip_value=1.0)
        #nn.utils.clip_grad_value_(discriminator_old.parameters(), clip_value=1.0)
        #nn.utils.clip_grad_value_(discriminator_young.parameters(), clip_value=1.0)

        if get_rank() == 0:

            # Evaluation
            if args.eval_every > 0 and i % args.eval_every == 0:
                with torch.no_grad():
                    fid_sa = fid_re = fid_sr = 0

                    encoder_old_ema.eval()
                    encoder_young_ema.eval()
                    generator_old2young_ema.eval()
                    generator_young2old_ema.eval()

                    nrow = int(args.n_sample**0.5)
                    nchw = list(sample_young.shape)[1:]
                    
                    # Reconstruction of test young images
                    latent_x, _ = encoder_young_ema(sample_young)
                    rec_old, _ = generator_young2old_ema([latent_x], input_is_latent=True)
                    sample = torch.cat((sample_young.reshape(args.n_sample // nrow, nrow, *nchw), rec_old.reshape(args.n_sample // nrow, nrow, * nchw)), 1)
                    utils.save_image(
                        sample.reshape(2 * args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-young2old.png"),
                        nrow=nrow,
                        normalize=True,
                    )
                        
                    # Reconstruction of test old images
                    latent_x, _ = encoder_old_ema(sample_old)
                    rec_young, _ = generator_old2young_ema([latent_x], input_is_latent=True)
                    sample = torch.cat((sample_old.reshape(args.n_sample // nrow, nrow, *nchw), rec_young.reshape(args.n_sample // nrow, nrow, * nchw)), 1)
                    utils.save_image(
                        sample.reshape(2 * args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-old2young.png"),
                        nrow=nrow,
                        normalize=True,
                    )
                    
                    # Reconstruction of train young images
                    latent_x, _ = encoder_young_ema(train_sample_young)
                    rec_old, _ = generator_young2old_ema([latent_x], input_is_latent=True)
                    sample = torch.cat((train_sample_young.reshape(args.n_sample // nrow, nrow, *nchw), rec_old.reshape(args.n_sample // nrow, nrow, * nchw)), 1)
                    utils.save_image(
                        sample.reshape(2 * args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-train-young2old.png"),
                        nrow=nrow,
                        normalize=True,
                    )
                        
                    # Reconstruction of train old images
                    latent_x, _ = encoder_old_ema(train_sample_old)
                    rec_young, _ = generator_old2young_ema([latent_x], input_is_latent=True)
                    sample = torch.cat((train_sample_old.reshape(args.n_sample // nrow, nrow, *nchw), rec_young.reshape(args.n_sample // nrow, nrow, * nchw)), 1)
                    utils.save_image(
                        sample.reshape(2 * args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-train-old2young.png"),
                        nrow=nrow,
                        normalize=True,
                    )

                    fid_young, fid_old = 0, 0 
                    
                    if 'fid_old' in args.which_metric:
                        if args.truncation < 1:
                            mean_latent = generator_young2old_ema.mean_latent(4096)
                        features = extract_feature_from_reconstruction(
                            encoder_young_ema, generator_young2old_ema,
                            inception, args.truncation,
                            mean_latent, it.islice(test_loader_young, args.n_sample_fid // args.batch),
                            args.device, input_is_latent=True, mode='recon',
                        ).numpy()
                        sample_old_mean = np.mean(features, 0)
                        sample_old_cov = np.cov(features, rowvar=False)
                        fid_old = calc_fid(sample_old_mean, sample_old_cov, real_mean_old, real_cov_old)

                    if 'fid_young' in args.which_metric:
                        if args.truncation < 1:
                            mean_latent = generator_old2young_ema.mean_latent(4096)
                        features = extract_feature_from_reconstruction(
                            encoder_old_ema, generator_old2young_ema,
                            inception, args.truncation,
                            mean_latent, it.islice(test_loader_old, args.n_sample_fid // args.batch),
                            args.device, input_is_latent=True, mode='recon',
                        ).numpy()
                        sample_young_mean = np.mean(features, 0)
                        sample_young_cov = np.cov(features, rowvar=False)
                        fid_young = calc_fid(sample_young_mean, sample_young_cov, real_mean_young, real_cov_young)

                    with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                        f.write(f"{i:07d}; fid_young: {float(fid_young):.4f}; fid_old: {float(fid_old):.4f};\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--path", type=str, help="path to the celeba dataset")
    parser.add_argument("--path-test", type=str, help="path to the test part of the celeba dataset")
    parser.add_argument("--arch", type=str, default='stylegan2', help="model architectures (stylegan2 | swagan)")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--sample_cache", type=str, default=None)
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=100, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=200, help="save latest checkpoints every # iters")
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_every", type=int, default=8, help="probability update interval of the adaptive augmentation")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_plus')
    parser.add_argument("--stddev_group", type=int, default=1)
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--vgg_ckpt", type=str, default="vgg16.pth")
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--lambda_vgg", type=float, default=5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_fake_d", type=float, default=1.0)
    parser.add_argument("--lambda_rec_d", type=float, default=1.0, help="d1, recon of real image")
    parser.add_argument("--lambda_fake_g", type=float, default=1.0)
    parser.add_argument("--lambda_rec_w", type=float, default=0, help="recon sampled w")
    parser.add_argument("--pix_loss", type=str, default='l2')
    parser.add_argument("--joint", action='store_true', help="update generator with encoder")
    parser.add_argument("--inception_young", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--inception_old", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--eval_every", type=int, default=1000, help="interval of metric evaluation")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--n_sample_fid", type=int, default=50000, help="number of the samples for calculating FID")
    parser.add_argument("--nframe_num", type=int, default=5)
    parser.add_argument("--decouple_d", action='store_true')
    parser.add_argument("--n_step_d", type=int, default=1)
    parser.add_argument("--n_step_e", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--e_ckpt", type=str, default=None, help="path to the checkpoint of encoder")
    parser.add_argument("--g_ckpt", type=str, default=None, help="path to the checkpoint of generator")
    parser.add_argument("--d_ckpt", type=str, default=None, help="path to the checkpoint of discriminator")
    parser.add_argument("--d2_ckpt", type=str, default=None, help="path to the checkpoint of discriminator2")
    parser.add_argument("--train_from_scratch", action='store_true')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument("--g_decay", type=float, default=None, help="g decay factor")
    parser.add_argument("--n_mlp_g", type=int, default=8)
    parser.add_argument("--ema_kimg", type=int, default=10, help="Half-life of the exponential moving average (EMA) of generator weights.")
    parser.add_argument("--ema_rampup", type=float, default=None, help="EMA ramp-up coefficient.")
    parser.add_argument("--no_ema_e", action='store_true')
    parser.add_argument("--no_ema_g", action='store_true')
    parser.add_argument("--which_metric", type=str, nargs='*', choices=['fid_young', 'fid_old'], default=[])
    parser.add_argument("--use_adaptive_weight", action='store_true', help="adaptive weight borrowed from VQGAN")
    parser.add_argument("--disc_iter_start", type=int, default=30000)
    parser.add_argument("--which_phi_e", type=str, default='lin2')
    parser.add_argument("--which_phi_d", type=str, default='lin2')
    parser.add_argument("--latent_space", type=str, default='w', help="latent space (w | p | pn | z)")
    parser.add_argument("--lambda_rec_w_extra", type=float, default=0, help="recon sampled w")

    args = parser.parse_args()
    util.seed_everything()
    device = args.device

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # args.n_mlp = 8
    args.n_latent = int(np.log2(args.size)) * 2 - 2
    args.latent = 512
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError

    assert((not args.use_adaptive_weight) or args.joint)
    assert(args.latent_space in ['w', 'z'])

    args.start_iter = 0
    args.iter += 1
    util.set_log_dir(args)
    util.print_args(parser, args)

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    # Setup generators
    generator_young2old = Generator(args.size, args.latent, args.n_mlp_g, channel_multiplier=args.channel_multiplier).to(device)
    generator_old2young = Generator(args.size, args.latent, args.n_mlp_g, channel_multiplier=args.channel_multiplier).to(device)
    generator_young2old_ema = Generator(args.size, args.latent, args.n_mlp_g, channel_multiplier=args.channel_multiplier).to(device)
    generator_old2young_ema = Generator(args.size, args.latent, args.n_mlp_g, channel_multiplier=args.channel_multiplier).to(device)
    #g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.g_reg_every > 0 else 1.
    g_reg_ratio = 1 # 0.01

    g_young2old_optim = optim.Adam(
        generator_young2old.parameters(),
        lr=args.lr *  g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    g_old2young_optim = optim.Adam(
        generator_old2young.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # Setup discriminators
    discriminator_old = Discriminator(args.size, channel_multiplier=args.channel_multiplier, which_phi=args.which_phi_d).to(device)
    discriminator_young = Discriminator(args.size, channel_multiplier=args.channel_multiplier, which_phi=args.which_phi_d).to(device)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.d_reg_every > 0 else 1.

    d_old_optim = optim.Adam(
        discriminator_old.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    d_young_optim = optim.Adam(
        discriminator_young.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # Setup encoder
    from model import Encoder
    encoder_young = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
        which_latent=args.which_latent, which_phi=args.which_phi_e,
        stddev_group=args.stddev_group).to(device)
    encoder_old = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
        which_latent=args.which_latent, which_phi=args.which_phi_e,
        stddev_group=args.stddev_group).to(device)
    
    encoder_young_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
        which_latent=args.which_latent, which_phi=args.which_phi_e,
        stddev_group=args.stddev_group).to(device)
    encoder_old_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
        which_latent=args.which_latent, which_phi=args.which_phi_e,
        stddev_group=args.stddev_group).to(device)

    e_reg_ratio = 1 #0.01
    e_young_optim = optim.Adam(
        encoder_young.parameters(),
        lr=args.lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )
    e_old_optim = optim.Adam(
        encoder_old.parameters(),
        lr=args.lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )

    # from idinvert_pytorch.models.perceptual_model import VGG16
    from torchvision.models import vgg16
    # vggnet = vgg16(output_layer_idx=args.output_layer_idx).to(device) #????
    vggnet = vgg16(pretrained=True).to(device)
    # vgg_ckpt = torch.load(args.vgg_ckpt, map_location=lambda storage, loc: storage)
    # vggnet.load_state_dict(vgg_ckpt)

    if args.resume:
        if args.ckpt is None:
            args.ckpt = os.path.join(args.log_dir, 'weight', f"latest.pt")
        print("Resuming models from:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            if 'iter' in ckpt:
                args.start_iter = ckpt["iter"]
            else:
                args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator_young2old.load_state_dict(ckpt["g"])
        generator_old2young.load_state_dict(ckpt["g"])
        generator_young2old_ema.load_state_dict(ckpt["g_ema"])
        generator_old2young_ema.load_state_dict(ckpt["g_ema"])
        g_young2old_optim.load_state_dict(ckpt["g_optim"])
        g_old2young_optim.load_state_dict(ckpt["g_optim"])

        discriminator_old.load_state_dict(ckpt["d"])
        discriminator_young.load_state_dict(ckpt["d"])
        d_old_optim.load_state_dict(ckpt["d_optim"])
        d_young_optim.load_state_dict(ckpt["d_optim"])

        encoder_old.load_state_dict(ckpt["e"])
        encoder_young.load_state_dict(ckpt["e"])
        encoder_old_ema.load_state_dict(ckpt["e_ema"])
        encoder_young_ema.load_state_dict(ckpt["e_ema"])
        e_young_optim.load_state_dict(ckpt["e_optim"])
        e_old_optim.load_state_dict(ckpt["e_optim"])

    if args.distributed:
        generator_young2old = nn.parallel.DistributedDataParallel(
            generator_young2old,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        generator_old2young = nn.parallel.DistributedDataParallel(
            generator_old2young,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        generator_young2old_ema = nn.parallel.DistributedDataParallel(
            generator_young2old_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        generator_old2young_ema = nn.parallel.DistributedDataParallel(
            generator_old2young_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        encoder_young = nn.parallel.DistributedDataParallel(
            encoder_young,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        encoder_old = nn.parallel.DistributedDataParallel(
            encoder_old,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        encoder_young_ema = nn.parallel.DistributedDataParallel(
            encoder_young_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        encoder_old_ema = nn.parallel.DistributedDataParallel(
            encoder_old_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator_young = nn.parallel.DistributedDataParallel(
            discriminator_young,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator_old = nn.parallel.DistributedDataParallel(
            discriminator_old,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # Create the training datasets
    dataset = get_image_dataset(args, args.dataset, args.path, train=True)

    dataset_young = data.Subset(dataset, [idx for idx, _cls in enumerate(dataset.targets) if _cls == 1])
    dataset_old = data.Subset(dataset, [idx for idx, _cls in enumerate(dataset.targets) if _cls == 0])

    if args.limit_train_batches < 1:
        indices = torch.randperm(len(dataset_young))[:int(args.limit_train_batches * len(dataset_young))]
        dataset_young = data.Subset(dataset_young, indices)

        indices = torch.randperm(len(dataset_old))[:int(args.limit_train_batches * len(dataset_old))]
        dataset_old = data.Subset(dataset_old, indices)

    loader_young = data.DataLoader(
        dataset_young,
        batch_size=args.batch,
        sampler=data_sampler(dataset_young, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )
    loader_old = data.DataLoader(
        dataset_old,
        batch_size=args.batch,
        sampler=data_sampler(dataset_old, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )

    print(f"Length of loader_young: {len(loader_young)}")
    print(f"Length of loader_old: {len(loader_old)}")


    # Create the test datasets
    test_dataset = get_image_dataset(args, args.dataset, args.path_test, train=False)

    test_dataset_young = data.Subset(test_dataset, [idx for idx, _cls in enumerate(test_dataset.targets) if _cls == 1])
    test_dataset_old = data.Subset(test_dataset, [idx for idx, _cls in enumerate(test_dataset.targets) if _cls == 0])

    test_loader_young = data.DataLoader(
        test_dataset_young,
        batch_size=args.batch,
        sampler=data_sampler(test_dataset_young, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader_old = data.DataLoader(
        test_dataset_old,
        batch_size=args.batch,
        sampler=data_sampler(test_dataset_old, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )

    train(
        args, loader_young, loader_old, 
        test_loader_young, test_loader_old,
        generator_young2old, generator_old2young, 
        generator_young2old_ema, generator_old2young_ema, 
        encoder_young, encoder_old,
        encoder_young_ema, encoder_old_ema,
        discriminator_young, discriminator_old,
        vggnet, g_young2old_optim, g_old2young_optim, 
        e_young_optim, e_old_optim, 
        d_young_optim, d_old_optim, 
        device
    )

    if torch.cuda.is_available():
        print(f"Max cuda allocated memory: {torch.cuda.max_memory_allocated()}")


