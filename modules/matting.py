import numpy as np
import torch
import os
from kornia.morphology import dilation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal.windows import gaussian
import torch.nn.functional as F

from modules.trajectories import Trajectory2D, Trajectory6D
from modules.utils import CameraParameters, savefig, tensor_to_numpy


class LayersGenerator(object):
    """Class to compute the depth layers for which blur behaves equally.
    
    Args:
    -----
        camera_motion (modules.trajectories.Trajectory6D): 
            Camera motion.
        camera_intrinsics (modules.utils.CameraParameters):
            Camera intrinsics.
        blur_diff (int):
            Number of pixels for blur variation.
        min_depth (float, optional):
            Minimum depth to consider. Defaults to None.
        max_layers (int, optional):
            Maximum number of layers to consider. Defaults to 100.
        show_plot (bool, optional):
            Show plot. Defaults to False.
        save_fig_dir (str, optional):
            Save figure directory. Defaults to ''.
    """
    def __init__(self,
            camera_motion: Trajectory6D,
            camera_intrinsics: CameraParameters,
            blur_diff: int,
            min_depth: float = None,
            max_layers: int = 100,
            show_plot: bool = False,
            save_fig_dir = '') -> None:
        self.depth_bins = self.compute_depth_bins(
                camera_motion,
                camera_intrinsics,
                blur_diff,
                min_depth,
                max_layers)
        self.show_plot = show_plot
        self.save_fig_dir = save_fig_dir

    def __call__(self, depth:torch.Tensor) -> torch.Tensor:
        """Returns the binarized depth layers.
        
        Args:
        -----
            depth (torch.Tensor):
                Depth map.
                
        Returns:
        --------
            layers (torch.Tensor):
                Binarized depth layers.
        """
        layers = self.depth_segmentation(depth)
        if self.show_plot:
            (fig, ax) = self.plot_depth_bins()
            if self.save_fig_dir != '':
                savefig(os.path.join(self.save_fig_dir, 'bins.pdf'), fig)
            (fig, ax) = self.show_layers(tensor_to_numpy(layers[0]))
            if self.save_fig_dir != '':
                savefig(os.path.join(self.save_fig_dir, 'layers.pdf'), fig)

        return layers

    def compute_depth_bins(self,
                        camera_motion: Trajectory6D,
                        camera_intrinsics: CameraParameters,
                        blur_diff: int,
                        min_depth: float = None,
                        max_layers: int = 100,
                        show_plot: bool = False,
                        save_fig: str = '') -> np.ndarray:
        """Compute the depth bins for which blur behaves equally
        
        Args:
        -----
            camera_motion (modules.trajectories.Trajectory6D):
                Camera motion.
            camera_intrinsics (modules.utils.CameraParameters):
                Camera intrinsics.
            blur_diff (int):
                Number of pixels for blur variation.
            min_depth (float, optional):
                Minimum depth to consider. Defaults to None.
            max_layers (int, optional):
                Maximum number of layers to consider. Defaults to 100.
            show_plot (bool, optional):
                Show plot. Defaults to False.
            save_fig (str, optional):
                Save figure directory. Defaults to ''.
        
        Returns:
        --------
            depth_bins (torch.Tensor):
                Depth bins.
        """
        # Extract the in-plane translation from the camera motion
        tx = torch.from_numpy(camera_motion.trans[:,0])
        ty = torch.from_numpy(camera_motion.trans[:,1])
        trajectory_2D = torch.stack([tx, ty], dim=1)
        # focal length
        f = camera_intrinsics.f
        # pixel size
        delta = torch.tensor(camera_intrinsics.px_size)
        # number of pixels for blur variation
        n = blur_diff
        # sequence constant
        C = torch.max(torch.abs(trajectory_2D / delta * f), dim=0).values
        # Compute the depth bins
        if min_depth is not None:
            db = torch.stack([2.0*C / (2.0*n*l + 1.0) for l in range(max_layers) if torch.any(2*C / (2*n*(l-1) + 1) > min_depth) or l == 0], dim=0)
            dbx, dby = db[:, 0], db[:, 1]
            limit = torch.max(db[db < min_depth])
            dbx = torch.cat([dbx[0, None], dbx[1:][dbx[1:] >= limit]])
            dby = torch.cat([dby[0, None], dby[1:][dby[1:] >= limit]])
            db = db[db >= limit]
            depth_bins = torch.sort(db.ravel(), descending=True).values
        else:
            db = torch.tensor([2*C / (2*n*l + 1) for l in range(max_layers)])
            dbx, dby = db[:, 0], db[:, 1]
            depth_bins = torch.sort(db.ravel(), descending=True).values

        self.dbx, self.dby = dbx, dby

        return depth_bins


    def plot_depth_bins(self, ax: plt.Subplot=None):
        """Plot the blur extension in pixels for given depth sequences

        Parameters
        ----------
        dx: ndarray
            depth sequence in x.
        dy:
            depth sequence in y.
        """

        Tx = np.arange(len(self.dbx))
        Ty = np.arange(len(self.dby))
        if ax is None:
            fig, ax = plt.subplots()
        ax.semilogx(tensor_to_numpy(self.dbx), Tx, 'or', label=r'$x$-axis')
        ax.semilogx(tensor_to_numpy(self.dby), Ty, 'ob', label=r'$y$-axis')
        plt.xlabel(r'$D_l$ [m]', fontsize=26)
        plt.ylabel(r'$T$', fontsize=26)
        ax.legend(loc='upper right', fontsize=24)
        plt.show()
        return (fig, ax)


    def depth_segmentation(self, depth: torch.Tensor) -> torch.Tensor:
        """Depth segmentation into layers"""
        h, w = depth.shape
        cum_layers = depth.view((1, 1, h, w)) < self.depth_bins.view(1, -1, 1, 1)
        depth_layers = torch.diff(torch.cat([torch.ones((1, 1, h, w), device=depth.device, dtype=torch.bool), cum_layers], dim=1), dim=1)
        return depth_layers.type_as(depth)

    @staticmethod
    def show_layers(layers: np.ndarray, ax: plt.Subplot=None):
        """Show the computed layers"""

        seg = np.sum(np.arange(layers.shape[0])[:, None, None] * layers.astype(float), axis=0)
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(seg, vmin=0.0, vmax=float(layers.shape[0]))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.axis('off')
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(r'$l$', rotation=270, fontsize=18, labelpad=18)
        plt.show()
        return (fig, ax)


class AlphaMattes(object):
    """Class for alpha mattes computation
    
    Args:
    -----
        kernels (torch.Tensor):
            Blur kernels.
        sigma (float):
            Smoothing gaussian sigma.
        show_plot (bool, optional):
            Show plot. Defaults to False.
    """
    def __init__(self, kernels:torch.Tensor, sigma:float, show_plot) -> None:
        self.kernels = kernels
        self.sigma = sigma
        self.ks = kernels.shape[1:]
        self.num_layers = kernels.shape[0]
        self.show_plot = show_plot

    def get_smoothing_filters(self, sigma, device=torch.device("cpu")):
        """Get the smoothing filters"""
        ks = int(self.ks[0])
        filters = torch.zeros((self.num_layers, self.num_layers, ks, ks), device=device)
        for l in range(self.num_layers):
            # sigma_k = sigma/float(ks) * float(torch.maximum(torch.sum(self.dilation_kernels[l].max(dim=1).values), torch.sum(self.dilation_kernels[l].max(dim=0).values)))
            supp = int(torch.maximum(torch.sum(self.dilation_kernels[l].max(dim=1).values), torch.sum(self.dilation_kernels[l].max(dim=0).values)))
            supp = 2 * (supp//2) + 1
            filters[l, l] = torch.from_numpy(self.smooth_func(supp, sigma, ks)).type(torch.float32)
        return filters
        
    def dilation(self, layers):
        """Dilation of the layers"""
        dilated_layers = []
        dilation_kernels = []
        for l in range(self.num_layers):
            kernel =  self.kernels[l] > 1.0/self.kernels[l].numel() #torch.ceil(self.kernels[l])
            dilated_layers.append(dilation(layers[:,l:l+1,:,:], kernel.type(torch.float32), engine='convolution'))
            dilation_kernels.append(kernel)
        self.dilation_kernels = torch.stack(dilation_kernels, dim=0)
        return torch.cat(dilated_layers, dim=1)


    @staticmethod
    def smooth_func(supp, sigma, ks):
        """Smooth function"""
        w = gaussian(supp, sigma)
        w2 = w[:,None] @ w[None,:]
        return np.pad(w2,(ks-supp)//2) / np.sum(w2)

    def __call__(self, layers:torch.Tensor) -> torch.Tensor:
        """Compute the alpha mattes
        
        Args:
        -----
            layers (torch.Tensor):
                Depth layers for which blur behaves equally.
        
        Returns:
        --------
            alpha_mattes (torch.Tensor):
                Alpha mattes.
        """
        # Dilate the layers
        dilated_layers = self.dilation(layers)
        b, _, h, w = layers.shape
        # Smooth the dilated layers
        filters = self.get_smoothing_filters(self.sigma, device=layers.device)
        dilated_layers_padded = F.pad(dilated_layers, (self.ks[1]//2, self.ks[1]//2, self.ks[0]//2, self.ks[0]//2), "replicate")
        smoothed_mattes = F.conv2d(dilated_layers_padded, filters)

        # Compute the occlusion mattes
        occlusion_mattes = []
        foo = torch.ones_like(layers, device=layers.device) - smoothed_mattes
        for j in range(0,self.num_layers-1):
            occlusion_mattes += [torch.prod(foo[:,j+1:,:,:], dim=1)] 
        occlusion_mattes += [torch.ones((b, h, w) , device=layers.device)]
        occlusion_mattes = torch.stack(occlusion_mattes, dim=1)
        # Compute the alpha mattes
        alpha_mattes = smoothed_mattes * occlusion_mattes
        alpha_mattes = alpha_mattes / torch.sum(alpha_mattes, dim=1, keepdim=True)

        if self.show_plot:
            (fig, ax) = LayersGenerator.show_layers(tensor_to_numpy(alpha_mattes[0]))

        return alpha_mattes

