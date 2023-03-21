from __future__ import print_function, division

import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data import dataset_fitting_dict
from modules import blur_model_dict
from modules.blur import blur_formation_eval
from configs.utils import get_dataset_info, read_config
from modules.utils import imsave, save_error_image, float_or_bool
from modules.deblur import deblurring

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def parser_fn(argv):
    """Parse command line arguments."""

    parser = argparse.ArgumentParser('Parallax Blur interface')
    parser.add_argument("--config", type=str, default='./configs/config.yaml', help='Path to config file')
    parser.add_argument("--device", type=str, default=None, help='Device to run experiment')
    parser.add_argument("--results_dir", type=str, default=None, help='Path to results directory')
    subparsers = parser.add_subparsers(dest='command', metavar={'blur', 'deblur'}, help='sub-command help')
    blur_parser = subparsers.add_parser('blur', help='Run blur formation')
    deblur_parser = subparsers.add_parser('deblur', help='Run deblurring')
    for subparser in [blur_parser, deblur_parser]:
        subparser.add_argument("--dataset", type=str, choices=['VirtualCMB', 'RealCMB'], required=True, help='Dataset to run experiment')
        subparser.add_argument("--model", type=str, choices=['ICB', 'PWB'], required=True, help='Blur formation model to be used. "ICB": Proposed Image Compositing Blur model. "PWB": Baseline Pixel-Wise convolution Blur model')
        subparser.add_argument("--id", type=int, required=True, help='Id number to run experiment')
    # Blur formation options
    blur_parser.add_argument("--blur_diff", type=float, default=None, help='Blur difference to be added to the blur model')
    blur_parser.add_argument("--sigma", type=float, default=None, help='Blur sigma to be added to the blur model')
    # Deblurring options
    deblur_parser.add_argument("--nn_model", type=str, choices=['SIREN', 'FOURIER_MAPPED_MLP'], default=None, help='Neural network model to be used')
    deblur_parser.add_argument("--hidden_features", type=int, default=None, help='Number of hidden features')
    deblur_parser.add_argument("--hidden_layers", type=int, default=None, help='Number of hidden layers')
    deblur_parser.add_argument("--fourier_scale", type=float_or_bool, default=0.0, help='Fourier scale')
    deblur_parser.add_argument("--lr", type=float, default=None, help='Learning rate')
    deblur_parser.add_argument("--num_epochs", type=int, default=None, help='Number of epochs')
    deblur_parser.add_argument("--scheduler_eta_min", type=float, default=None, help='Minimum learning rate')
    deblur_parser.add_argument("--clip_grad", type=float_or_bool, default=None, help='Gradient clipping value')
    deblur_parser.add_argument("--gradient_fn", type=str, choices=['net_grad', 'filter'], default=None, help='Gradient function')
    deblur_parser.add_argument("--gradient_weight", type=float, default=None, help='Gradient weight')
    deblur_parser.add_argument("--p_norm", type=float, default=None, help='p-norm value')
    deblur_parser.add_argument("--save_nn_model", action='store_true', help='Save neural network model')
    deblur_parser.add_argument("--load_nn_model", action='store_true', help='Load neural network model')
    deblur_parser.add_argument("--blur_diff", type=float, default=None, help='Blur difference to be added to the blur model')
    deblur_parser.add_argument("--sigma", type=float, default=None, help='Blur sigma to be added to the blur model')

    args = parser.parse_args(argv)
    return args


def save_images(est, gt, img_exp_data, CONFIG, args):
    """Save estimated and error images"""
    est_path = os.path.join(
        CONFIG['RESULTS_DIR'],
        args.command,
        args.dataset,
        args.model,
        'png',
        'output',
        '{}.png'.format(img_exp_data['img_exp'])
    )

    error_path = os.path.join(
        CONFIG['RESULTS_DIR'],
        args.command,
        args.dataset,
        args.model,
        'png',
        'error',
        '{}.png'.format(img_exp_data['img_exp'])
    )

    os.makedirs(os.path.dirname(est_path), exist_ok=True)
    os.makedirs(os.path.dirname(error_path), exist_ok=True)

    imsave(est, est_path)
    save_error_image(est, gt, error_path)


def model_path(img_exp_data, CONFIG, args):
    """Get model path"""
    if args.load_nn_model:
        model_path = os.path.join(
            CONFIG['RESULTS_DIR'],
            args.command,
            args.dataset,
            args.model,
            'ckpt',
            '{}.pth'.format(img_exp_data['img_exp'])
        )
    else:
        model_path = None
    return model_path


def save_model(model, img_exp_data, CONFIG, args):
    """Save model"""
    model_path = os.path.join(
        CONFIG['RESULTS_DIR'],
        args.command,
        args.dataset,
        args.model,
        'ckpt',
        '{}.pth'.format(img_exp_data['img_exp'])
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)


def save_log(df_log: pd.DataFrame, img_exp_data:dict, CONFIG:dict, args):
    """Save log"""
    log_path = os.path.join(
        CONFIG['RESULTS_DIR'],
        args.command,
        args.dataset,
        args.model,
        'logs',
        '{}.csv'.format(img_exp_data['img_exp'], index=True, index_label="index", header=True)
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_log.to_csv(log_path, index=True, index_label="index", header=True)


def blur(CONFIG, args):
    """Run blur formation experiment"""
    # Blur model parameters
    blur_model_params_dict ={
        'ICB': CONFIG['PARALLAX_ICB_MODEL'],
        'PWB': {}    
    }
    # Get dataset info
    dataset_name = args.dataset
    dataset_config_info = argparse.Namespace(**get_dataset_info(dataset_name, CONFIG))
    df = pd.read_csv(dataset_config_info.INFO_CSV, keep_default_na=False)
    # Get image experiment info
    img_exp_data = df.loc[args.id].to_dict()
    print('Run blur formation: {}'.format(img_exp_data['img_exp']))
    # Generate dataset loader
    dataset = dataset_fitting_dict[dataset_name](dataset_config_info.ROOT, img_exp_data)
    # Get blur model function, device and patch size
    blur_model = blur_model_dict[args.model]
    model_params = blur_model_params_dict[args.model]
    device = torch.device(CONFIG['DEVICE'])
    patch_size = dataset_config_info.PATCH_SIZE['blur']
    # Evaluate blur formation
    perf, est, gt = blur_formation_eval(
        dataset=dataset,
        blur_model=blur_model,
        model_params=model_params,
        device=device,
        patch_size=patch_size
    )

    return est, gt, perf, img_exp_data, df


def deblur(CONFIG, args):
    """Run deblurring experiment"""
    # Blur model parameters
    blur_model_params_dict ={
        'ICB': CONFIG['PARALLAX_ICB_MODEL'],
        'PWB': {}    
    }
    # Get dataset info
    dataset_name = args.dataset
    dataset_config_info = argparse.Namespace(**get_dataset_info(dataset_name, CONFIG))
    df = pd.read_csv(dataset_config_info.INFO_CSV, keep_default_na=False)
    # Get image experiment info
    img_exp_data = df.loc[args.id].to_dict()
    print('Run deblurring: {}'.format(img_exp_data['img_exp']))
    dataset = dataset_fitting_dict[dataset_name](dataset_config_info.ROOT, img_exp_data)
    # Get blur model function, device and patch size
    blur_model = blur_model_dict[args.model]
    model_params = blur_model_params_dict[args.model]
    device = torch.device(CONFIG['DEVICE'])
    patch_size = dataset_config_info.PATCH_SIZE['deblur']
    # Init blur model
    blur_nn = blur_model(**dataset.get_meta_data(), patch_size=patch_size, **model_params)
    dataset.slicing_data(patch_size=patch_size, padding=blur_nn.ks//2)
    # Get model path
    ckpt_path = model_path(img_exp_data, CONFIG, args)
    # Evaluate deblurring
    perf, est, gt, mlp_nn = deblurring(
        blur_dataset=dataset,
        blur_nn=blur_nn,
        deblur_params=CONFIG['DEBLUR'],
        device=device,
        load_ckpt=args.load_nn_model,
        ckpt_path=ckpt_path
    )
    # Save model if needed
    if args.save_nn_model:
        save_model(mlp_nn, img_exp_data, CONFIG, args)

    return est, gt, perf, img_exp_data, df


def main(argv=None):
    """Main function"""
    # Parse arguments
    args = parser_fn(argv)
    # Read config
    CONFIG = read_config(args.config, args)
    # Run experiment
    est, gt, perf, img_exp_data, df = globals()[args.command](CONFIG, args)
    # Save images
    save_images(est, gt, img_exp_data, CONFIG, args)
    # Save metrics
    df_log = df.loc[args.id]
    for metric in ['PSNR', 'SSIM', 'LPIPS', 'elapsedTime', 'modelSize']:
        df_log[metric] = perf[metric]
        print("{}: {:5f}".format(metric, perf[metric]))

    save_log(df_log, img_exp_data, CONFIG, args)


if __name__ == '__main__':
    main()