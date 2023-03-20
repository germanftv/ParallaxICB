from __future__ import print_function, division

import os
import argparse
import pandas as pd
import numpy as np

from configs.utils import read_config, get_dataset_info
from modules.utils import search_with_pattern

pd.set_option('display.float_format', lambda x: '%.3e' % x)

METRICS = ['PSNR', 'SSIM', 'LPIPS', 'elapsedTime', 'modelSize']


def parser_fn(argv):
    """Parse arguments"""
    parser = argparse.ArgumentParser('Summary interface')
    parser.add_argument("--config", type=str, default='./configs/config.yaml', help='Path to config file')
    parser.add_argument("--results_dir", type=str, default=None, help='Path to results directory')
    subparsers = parser.add_subparsers(dest='command', metavar={'blur', 'deblur'}, help='sub-command help')
    blur = subparsers.add_parser('blur', help='Run blur formation')
    deblur = subparsers.add_parser('deblur', help='Run deblurring')
    for subparser in [blur, deblur]:
        subparser.add_argument("--dataset", type=str, choices=['VirtualCMB', 'RealCMB'], required=True, help='Dataset to run experiment')
        # subparser.add_argument("--model", type=str, choices=['ICB', 'PWB'], required=True, help='Model to run summary results. "ICB": Proposed Image Compositing Blur model. "PWB": Baseline Pixel-Wise convolution Blur model')
        subparser.add_argument("--model", required=True, help='Model to run summary results')
    args = parser.parse_args(argv)
    return args


def read_logs(CONFIG:dict, args):
    """Read logs from results directory"""

    # Get logs
    logs_dir = os.path.join(
        CONFIG['RESULTS_DIR'],
        args.command,
        args.dataset,
        args.model,
        'logs'
    )
    logs = search_with_pattern(logs_dir, '*.csv')
    # Read dataset info
    dataset_name = args.dataset
    dataset_config_info = argparse.Namespace(**get_dataset_info(dataset_name, CONFIG))
    df_list = pd.read_csv(dataset_config_info.INFO_CSV, keep_default_na=False)
    log_list = []
    # loop over img exp
    for log in logs:
        df = pd.read_csv(log, index_col="index", keep_default_na=False).T
        if df['img_exp'].values in df_list['img_exp'].values:
            log_list.append(df)
    # Concatenate logs
    df_logs = pd.concat(log_list, axis=0)
    print('Number of logs: {}'.format(len(df_logs)))
    df_logs= df_logs.sort_index()
    return df_logs


def read_metrics_VirtualCMB(df: pd.DataFrame, metrics: list, translation=None, rotation=None, case=None):
    """Read VirtualCMB metrics"""

    mask_tm = np.ones(len(df)).astype(bool) if translation is None else (df['trans_mode'].values == translation)
    mask_rm = np.ones(len(df)).astype(bool) if rotation is None else (df['rot_mode'].values == rotation)
    mask_tags = np.ones(len(df)).astype(bool) if case is None else (df['tag'].values == case)
    df_filtered = df.iloc[mask_tm & mask_rm & mask_tags].copy()
    columns=[]
    if translation is None:
        columns.append('trans_mode')
    if rotation is None:
        columns.append('rot_mode')
    if case is None:
        columns.append('tag')
    columns += metrics 

    for column in columns:
        if column in metrics:
            if column in df_filtered.columns:
                df_filtered[column] = df_filtered[column].map(lambda x: np.NaN if x=='' else np.float64(x))
            else:
                df_filtered[column] = np.NaN

    return df_filtered[columns].copy()


def read_metrics_RealCMB(df: pd.DataFrame, metrics: list):
    """Read RealCMB metrics"""

    for metric in metrics:
        if metric in df.columns:
            df[metric] = df[metric].map(lambda x: np.NaN if x=='' else np.float64(x))
        else:
            df[metric] = np.NaN

    return df[metrics].copy()


def VirtualCMB_summary(df_logs):
    """Summary VirtualCMB results"""

    motion_types={
        'parallax':{
            'trans': 'xy',
            'rot': '',
            'prompt': 'Parallax motion only'
        },
        'parallax_plus_rot':{
            'trans': 'xy',
            'rot': 'xy',
            'prompt': 'Parallax motion with "xy" rotations'
        },
        '6D':{
            'trans': 'xyz',
            'rot': 'xyz',
            'prompt': '6D motion'
        },
        'overall':{
            'trans': None,
            'rot': None,
            'prompt': 'Overall'
        }
    }

    for case in motion_types.keys():
        translation = motion_types[case]['trans']
        rotation    = motion_types[case]['rot']
        df = read_metrics_VirtualCMB(
            df_logs, 
            METRICS, 
            translation=translation,
            rotation=rotation)
        if df.empty:
            continue
        columns = ['tag', *METRICS]
        if case != 'overall':
            df_results = df.loc[:, columns].groupby('tag').mean(numeric_only=True)
        else:
            df_results = df.loc[:, METRICS].mean(numeric_only=True)
    
        print('\nResults in VirtualCMB ({}):'.format(motion_types[case]['prompt']))
        print(df_results.T)


def RealCMB_summary(df_logs):
    """Summary RealCMB results"""

    df = read_metrics_RealCMB(df_logs, METRICS)
    df_results = df.loc[:, METRICS].mean(numeric_only=True)
    print('\nResults in RealCMB:')
    print(df_results.T)


summary_dict = {
    'VirtualCMB': VirtualCMB_summary,
    'RealCMB': RealCMB_summary
}


def main(argv=None):
    args = parser_fn(argv)
    CONFIG = read_config(args.config)
    df_logs = read_logs(CONFIG, args)
    summary = summary_dict[args.dataset]
    summary(df_logs)
    


if __name__ == '__main__':
    main()