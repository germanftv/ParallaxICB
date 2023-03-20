import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
import lpips
import math
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from modules.matting import LayersGenerator, AlphaMattes
from modules.kernels import ParallaxICBKernels, PointwiseConvKernels
from modules.utils import get_model_size
from data.utils import IMG_INV_TRANSFORMS, DEPTH_INV_TRANSFORMS


class ParallaxICBlurModel(nn.Module):
    """ParallaxICBlurModel class.

    Args:
    ----------
        depth (torch.Tensor): 
            Depth map.
        camera_motion (modules.trajectories.Trajectory6D): 
            Camera motion.
        camera_intrinsics (modules.utils.CameraParameters): 
            Camera intrinsics.
        blur_diff (int, optional): 
            Number of pixels for blur variation. Defaults to 1.
        sigma (float, optional): 
            Sigma for the Gaussian smoothing. Defaults to 1.0.
        device (str, optional): 
            Device to use. Defaults to "cpu".
        patch_size (tuple, optional): 
            Patch size to use. Defaults to None.
    """
    def __init__(self, depth, camera_motion, camera_intrinsics, blur_diff=1, sigma=1.0, device="cpu", patch_size=None):
        super(ParallaxICBlurModel, self).__init__()
        min_depth = torch.min(depth)
        device = torch.device(device)
        # Define patch size
        if patch_size is None:
            self.patch_size = tuple(depth.shape[0:2])
        else:
            self.patch_size = patch_size
        # Define layers generator
        layers_generator = LayersGenerator(camera_motion, camera_intrinsics, blur_diff=blur_diff, min_depth=min_depth, show_plot=False)
        if device != torch.device("cpu"):
            depth = depth.to(device)
            layers_generator.depth_bins = layers_generator.depth_bins.to(device)
        # Get depth layers
        depth_layers = layers_generator(depth)
        # Get blur kernels
        kernels_generator = ParallaxICBKernels(camera_motion, depth, depth_layers, layers_generator.depth_bins, camera_intrinsics, show_plot=False)
        self.ks = kernels_generator.kernel_size()[0]
        self.kernels = kernels_generator()
        # Get alpha mattes
        alpha_mattes_generator = AlphaMattes(self.kernels, sigma=sigma, show_plot=False) 
        self.alpha_mattes = alpha_mattes_generator(depth_layers)

        # Define convolutional layers
        self.num_layers, self.ks = self.kernels.shape[0:2]
        self.conv  = nn.Conv2d(3, 3*self.num_layers, (self.ks,self.ks), bias=False, groups=3, padding=0)
        # Initialize weights
        self.weights_init()

    def weights_init(self):
        """Initialize convolutional weights."""
        # Initialize convolutional weights
        weights_conv = torch.cat([torch.flip(self.kernels, dims=(1,2)) for _ in range(3)])
        self.conv.weight = nn.Parameter(weights_conv.view(3*self.num_layers, 1, self.ks, self.ks), requires_grad=False)

        # Initialize alpha mattes
        alpha_mattes = F.unfold(self.alpha_mattes, self.patch_size, stride=self.patch_size)
        alpha_mattes = alpha_mattes.view(self.num_layers, *self.patch_size, -1).permute(3,0,1,2)
        self.alpha_mattes = nn.Parameter(alpha_mattes, requires_grad=False)

    def forward(self, img, idx):
        """Forward pass.
        
        Args:
        ----------
            img (torch.Tensor): Input image patch.
            idx (int): Index of the image patch.
        
        Returns:
        ----------
            torch.Tensor: Blurred image patch.
        """
        B, C = img.shape[0:2]
        blurs = self.conv(img)
        H, W = blurs.shape[2:]

        # Image compositing blur
        blurry = torch.sum(blurs.view(B, C, self.num_layers, H, W) * self.alpha_mattes[idx].view(B, 1, self.num_layers, H, W), dim=2, keepdim=False)

        return blurry


class PointwiseConvBlurModel(nn.Module):
    """PointwiseConvBlurModel class.
    
    Args:
    ----------
        depth (torch.Tensor):
            Depth map.
        camera_motion (modules.trajectories.Trajectory6D):
            Camera motion.
        camera_intrinsics (modules.utils.CameraParameters):
            Camera intrinsics.
        patch_size (tuple, optional):
            Patch size to use. Defaults to None.
        """
    def __init__(self, depth, camera_motion, camera_intrinsics, patch_size=None, **kwargs):
        super(PointwiseConvBlurModel, self).__init__()
        # Define patch size
        if patch_size is None:
            self.patch_size = tuple(depth.shape[0:2])
        else:
            self.patch_size = patch_size
        # Define kernels
        kernels_generator = PointwiseConvKernels(depth, camera_motion, camera_intrinsics)
        self.kernels = kernels_generator()
        self.ks = kernels_generator.kernel_size()[0]
        # Define unfold layer
        self.unfold = nn.Unfold(kernel_size=(self.ks, self.ks), dilation=1, padding=0, stride=1)
        # Initialize weights
        self.weights_init()

    def weights_init(self):
        """Initialize weights."""
        kernels = F.unfold(self.kernels, self.patch_size, stride=self.patch_size)   #(1,ks**2*Hp*Wp,L)
        kernels = kernels.view(1, self.ks**2, math.prod(self.patch_size), -1).permute(3,0,1,2)  #(L,1,ks**2,Hp*Wp) 
        self.kernels = nn.Parameter(kernels, requires_grad=False)

    def forward(self, img, idx):
        """Forward pass.
        
        Args:
        ----------
            img (torch.Tensor): Input image patch.
            idx (int): Index of the image patch.
            
        Returns:
        ----------
            torch.Tensor: Blurred image patch.
        """
        B, C, Hp, Wp = img.shape
        Hp -= 2 * (self.ks//2)
        Wp -= 2 * (self.ks//2) 
        # Pointwise convolution
        unfold_img = self.unfold(img)   #(B,C*ks**2,Hp*Wp)
        blurry = (self.kernels[idx] * unfold_img.view(B,C,self.ks**2,math.prod(self.patch_size))).sum(dim=2, keepdims=True)

        return blurry.view(B,C,Hp,Wp)


def blur_formation_eval(dataset, blur_model, model_params, device, patch_size):
    """Blur formation evaluation.
    
    Args:
    ----------
        dataset (data.datasets.BaseCMBFitting):
            Dataset.
        blur_model:
            Blur model function.
        model_params (dict):
            Blur model parameters.
        device (torch.device):
            Device to use.
        patch_size (tuple):
            Patch size to use.
    
    Returns:
    ----------
        dict: Blur formation evaluation metrics.
        numpy.ndarray: Estimated blurred image.
        numpy.ndarray: GT blurred image.
        """
    # lpips network
    lpips_fn = lpips.LPIPS(net='vgg')

    start_time = timer()
    # Blur formation model
    blur_nn = blur_model(**dataset.get_meta_data(), **model_params, patch_size=patch_size)
    # Slicing dataset
    dataset.slicing_data(patch_size=patch_size, padding=blur_nn.ks//2)
    # Move to device
    blur_nn.to(device)

    dataloader = DataLoader(dataset, batch_size=1)
    blur = []
    # Loop over patches
    with torch.no_grad():
        for sample in dataloader:
            sharp = sample['sharp'].to(device)
            idx = sample['idx']
            blur.append(blur_nn(sharp, idx))

    # Reconstruct image
    blur = torch.cat(blur).permute(1,2,3,0).view(1,-1, len(dataset))
    blur = F.fold(blur, output_size=dataset.blurry.shape[0:2], kernel_size=dataset.patch_size, stride=dataset.patch_size)
    end_time = timer()

    # Compute metrics
    blurry_gt = dataset.blurry
    blurry_est = IMG_INV_TRANSFORMS(blur[0]).astype(np.float64)
    psnr = peak_signal_noise_ratio(blurry_gt, blurry_est)
    ssim = structural_similarity(blurry_gt, blurry_est,  channel_axis=2)
    lpips_value = lpips_fn.forward(lpips.im2tensor(blurry_gt), lpips.im2tensor(blurry_est)).detach().numpy().squeeze()
    elapsed_time = end_time - start_time
    model_size = get_model_size(blur_nn)
    # Concatenate metrics
    perf = {
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips_value,
        'elapsedTime': elapsed_time,
        'modelSize': model_size
    }

    return perf, blurry_est, blurry_gt