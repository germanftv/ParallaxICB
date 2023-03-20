from __future__ import print_function, division, absolute_import

import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from modules.utils import read_log, read_RGBA_encoded_depth, imread
from modules.trajectories import VirtualCMB_CamLogs, RealCMBBundle
from data.utils import IMG_TRANSFORMS, DEPTH_TRANSFORMS


# Default transforms
TRANSFORMS = {
    'img': IMG_TRANSFORMS,
    'depth': DEPTH_TRANSFORMS
}


def get_mgrid(img_size):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    Args:
    ----------
        img_size (tuple): Image size.

    Returns:
    ----------
        torch.Tensor: Grid of coordinates.'''
    tensors = tuple([torch.linspace(-1, 1, steps=steps, dtype=torch.float32) for steps in reversed(img_size)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='xy'), dim=0)
    return mgrid


class BaseCMBFitting(Dataset):
    '''Base class for CMB fitting datasets.

    Args:
    ----------
        root (str): Path to dataset root.
        img_exp_data (dict): Experimental image data.
        transforms (dict): Transforms to apply to images and depth maps.
    '''
    def __init__(self, root:str, img_exp_data:dict, transforms=TRANSFORMS) -> None:
        super().__init__()
        self.img_exp = img_exp_data['img_exp']
        self.camera_intrinsics = None
        self.camera_motion = None
        self.blurry = None
        self.blurry_tns = None
        self.sharp = None
        self.sharp_tns = None
        self.depth = None
        self.img__size = None
        self.sliced = False
        self.transforms = transforms
    
    def get_meta_data(self):
        """Returns meta data of the dataset."""
        return {
            'depth': self.transforms['depth'](self.depth),
            'camera_motion': self.camera_motion,
            'camera_intrinsics': self.camera_intrinsics
        }

    def slicing_data(self, patch_size:tuple, padding:int):
        """Slices the data into patches of size patch_size.

        Args:
        ----------
            patch_size (tuple): Patch size.
            padding (int): Padding to apply to the patches.
        """
        if patch_size is None:
            patch_size = self.img_size
        elif all(list(map(lambda x, y: x>=y, patch_size, self.img_size))):
            patch_size = self.img_size
        sharp = self.transforms['img'](self.sharp)
        C = sharp.shape[0]
        sharp = F.pad(torch.unsqueeze(sharp, 0), (padding, padding, padding, padding), "replicate")
        patch_padded_size = list(map(lambda x: x + 2*padding, patch_size))
        sharp_unfolded = F.unfold(sharp, patch_padded_size, stride=patch_size)

        self.sharp_tns = sharp_unfolded.view(C,*patch_padded_size, -1).permute(3,0,1,2)

        blurry = self.transforms['img'](self.blurry)
        blurry_unfolded = F.unfold(torch.unsqueeze(blurry, 0), patch_size, stride=patch_size)
        self.blurry_tns = blurry_unfolded.view(C, *patch_size, -1).permute(3,0,1,2)

        img_padded_size = list(map(lambda x: x + 2*padding, self.img_size))
        coords = get_mgrid(img_padded_size)
        coords_unfolded = F.unfold(torch.unsqueeze(coords, 0), patch_padded_size, stride=patch_size)
        self.coords = coords_unfolded.view(2,*patch_padded_size, -1).permute(3,0,1,2)
        self.patch_size = patch_size
        self.sliced = True
    
    def __len__(self):
        return self.sharp_tns.shape[0] if self.sliced else 1

    def __getitem__(self, idx):   

        if not self.sliced:
            raise ValueError('Data need to be sliced')
        
        sample = {
            'sharp': self.sharp_tns[idx],
            'blurry': self.blurry_tns[idx],
            'coords' : self.coords[idx],
            'idx': idx
        }
        return sample


class VirtualCMBFitting(BaseCMBFitting):
    """Virtual CMB fitting dataset.

    Args:
    ----------
        root (str): Path to dataset root.
        img_exp_data (dict): Experimental image data.
        transforms (dict): Transforms to apply to images and depth maps.
    """
    def __init__(self, root:str, img_exp_data:dict, transforms=TRANSFORMS) -> None:
        super().__init__(root, img_exp_data, transforms)
        self.sharp_file = os.path.join(root, 'sharp_ref', '{}.png'.format(self.img_exp))
        self.blurry_file = os.path.join(root, 'blurry', '{}.png'.format(self.img_exp))
        self.depth_map_file = os.path.join(root, 'depth_ref', '{}.png'.format(self.img_exp))
        self.cam_logs_dir = os.path.join(root, 'camera_logs', self.img_exp)
        self.param_log_file = os.path.join(root, 'param_logs', '{}.txt'.format(self.img_exp))

        sampling_freq = 2000.0 if img_exp_data['tag'] == 'Trucking' else 500.0
        # Read param and camera logs
        param_log   = read_log(self.param_log_file)
        cam_logs    = VirtualCMB_CamLogs(self.cam_logs_dir, sampling_freq)
        # Get camera intrinsics and motion
        self.camera_intrinsics  = cam_logs.camera_intrinsics()
        self.camera_motion      = cam_logs.camera_motion()
        ref_camera_log          = cam_logs.ref_cam_log()
        # Get camera motion mode
        trans_mode, rot_mode = param_log['traslationMode'], param_log['rotationMode']

        # Read images and depth map
        self.sharp      = imread(self.sharp_file)
        self.blurry     = imread(self.blurry_file)
        self.depth      = read_RGBA_encoded_depth(
                            self.depth_map_file,
                            inf_value=1.5*ref_camera_log['depthRange']['max'],
                            clipping=(ref_camera_log['depthRange']['min'], ref_camera_log['depthRange']['max'])
                            )
        self.img_size = self.depth.shape

        
class RealCMBFitting(BaseCMBFitting):
    """Real CMB fitting dataset.

    Args:
    ----------
        root (str): Path to dataset root.
        img_exp_data (dict): Experimental image data.
        transforms (dict): Transforms to apply to images and depth maps.
    """
    def __init__(self, root:str, img_exp_data:dict, transforms=TRANSFORMS) -> None:
        super().__init__(root, img_exp_data, transforms)
        self.img_exp = img_exp_data['img_exp']
        self.sharp_file = os.path.join(root, 'sharp_ref', '{}.png'.format(self.img_exp))
        self.blurry_file = os.path.join(root, 'blurry', '{}.png'.format(self.img_exp))
        self.depth_map_file = os.path.join(root, 'depth_ref', '{}.npz'.format(self.img_exp))
        self.bundle_info_file = os.path.join(root, 'bundle_info', '{}.npz'.format(self.img_exp))
        self.transforms = transforms
        # Read images and depth map
        self.sharp       = imread(self.sharp_file)
        self.blurry      = imread(self.blurry_file)
        self.depth       = np.load(self.depth_map_file, allow_pickle=True)['depth_ref']
        # Read bundle info
        bundle_info = np.load(self.bundle_info_file, allow_pickle=True)['bundle_info']
        realCMB_bundle = RealCMBBundle(bundle_info)
        # Get camera intrinsics and motion
        self.camera_intrinsics   = realCMB_bundle.camera_intrinsics()
        self.camera_motion       = realCMB_bundle.camera_motion()

        self.img_size = self.depth.shape


    

