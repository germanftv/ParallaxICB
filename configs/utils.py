from __future__ import print_function, division
import yaml
import os

def assign(x, y):
    return x if x is not None else y

def read_config(config_file:str, args=None):
    """Read config file and assign command line arguments if provided.

    Args:
    -----
        config_file (str): 
            Path to config file.
        args (argparse.Namespace): 
            Command line arguments.

    Returns:
    --------
        dict: Config dictionary.
    """
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if args is None:
        return config
    config['DEVICE'] = assign(args.device, config['DEVICE'])
    config['RESULTS_DIR'] = assign(args.results_dir, config['RESULTS_DIR'])

    config['PARALLAX_ICB_MODEL']['device'] = config['DEVICE']
    config['PARALLAX_ICB_MODEL']['blur_diff'] = assign(args.blur_diff, config['PARALLAX_ICB_MODEL']['blur_diff'])
    config['PARALLAX_ICB_MODEL']['sigma'] = assign(args.sigma, config['PARALLAX_ICB_MODEL']['sigma'])
    if args.command == 'deblur':
        config['DEBLUR']['nn_model'] = assign(args.nn_model, config['DEBLUR']['nn_model'])
        config['DEBLUR'][config['DEBLUR']['nn_model']]['hidden_features'] = assign(args.hidden_features, config['DEBLUR'][config['DEBLUR']['nn_model']]['hidden_features'])
        config['DEBLUR'][config['DEBLUR']['nn_model']]['hidden_layers'] = assign(args.hidden_layers, config['DEBLUR'][config['DEBLUR']['nn_model']]['hidden_layers'])
        if args.fourier_scale != 0.0:
            config['DEBLUR'][config['DEBLUR']['nn_model']]['fourier_scale'] = args.fourier_scale
        elif args.fourier_scale == False:
            config['DEBLUR'][config['DEBLUR']['nn_model']]['fourier_scale'] = None
        config['DEBLUR'][config['DEBLUR']['nn_model']]['fourier_scale'] = args.fourier_scale if args.fourier_scale != 0.0 else config['DEBLUR'][config['DEBLUR']['nn_model']]['fourier_scale']
        config['DEBLUR']['num_epochs'] = assign(args.num_epochs, config['DEBLUR']['num_epochs'])
        config['DEBLUR']['lr'] = assign(args.lr, config['DEBLUR']['lr'])
        config['DEBLUR']['scheduler_eta_min'] = assign(args.scheduler_eta_min, config['DEBLUR']['scheduler_eta_min'])
        config['DEBLUR']['clip_grad'] = assign(args.clip_grad, config['DEBLUR']['clip_grad'])
        config['DEBLUR']['gradient_fn'] = assign(args.gradient_fn, config['DEBLUR']['gradient_fn'])
        config['DEBLUR']['gradient_weight'] = assign(args.gradient_weight, config['DEBLUR']['gradient_weight'])
        config['DEBLUR']['p_norm'] = assign(args.p_norm, config['DEBLUR']['p_norm'])
    return config


def get_dataset_info(dataset:str, CONFIG:dict):
    """Get dataset info from config file.

    Args:
    -----
        dataset (str): Dataset name.
        CONFIG (dict): Config dictionary.

    Returns:
    --------
        dict: Dataset info.
    """
    for dataset_info in CONFIG['DATASETS']:
        if dataset_info['NAME'] == dataset:
            return dataset_info