from cmath import pi
from math import degrees
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from numba import njit, jit, prange

from modules.utils import CameraParameters
from modules.trajectories import Trajectory2D, Trajectory6D
from modules.utils import round, tensor_to_numpy


class ParallaxICBKernels(object):
    """Class for the approximation of the ICB depth-dependent blur kernels

    The generated kernels approximate 6DoF camera motion with xy-translation + xy-rotation.
    Large focal length is assumed in order to approximate xy-rotation as uniform planar
    motion. Kernels are discretized using the mean depth of the values for each range.

    Args:
    ----------
        camera_motion (modules.trajectories.Trajectory6D):
            6DoF camera trajectory.
        depth (torch.Tensor):
            depth map.
        depth_layers (torch.Tensor):
            binary depth layers discretized using depth_bins.
        depth_bins (torch.Tensor):
            D-length array with ordered depth bins for which blur behaves equally.
        camera_intrinsics (modules.utils.CameraParameters):
            Camera intrinsic parameters.
        show_plot (bool, optional):
            Show plot of the generated kernels. Defaults to False.
    """
    def __init__(self, 
                 camera_motion: Trajectory6D, 
                 depth: torch.Tensor,
                 depth_layers: torch.Tensor,
                 depth_bins: torch.Tensor,
                 camera_intrinsics: CameraParameters,
                 show_plot:bool=False):

        # Translation 
        ts_x = torch.from_numpy(camera_motion.trans[:, 0])
        ts_y = torch.from_numpy(camera_motion.trans[:, 1])
        blur_t_xy = torch.stack([ts_x, ts_y], dim=1).to(depth.device)
        # Rotation 
        tr_x = torch.from_numpy(camera_motion.rot[:, 1])
        tr_y = torch.from_numpy(-camera_motion.rot[:, 0])
        blur_r_xy = torch.stack([tr_x, tr_y], dim=1).to(depth.device) * torch.pi/180.0
        # Focal length and pixel size
        f = camera_intrinsics.f
        delta = torch.tensor(camera_intrinsics.px_size).to(depth.device)
        # Rotation discretization
        rd = (blur_r_xy * f) / (delta)

        # Trajectory discretization
        discretized_depths = torch.stack([torch.mean(torch.masked_select(depth[None, :, :], depth_layers[0, i].type(torch.bool))) if i>0 else depth_bins[i] for i in range(len(depth_bins))])
        indices = torch.argwhere(torch.isnan(discretized_depths))
        discretized_depths[indices] = depth_bins[indices]
        sd = (blur_t_xy * f) / (delta*discretized_depths[:, None, None])

        # Kernel generation
        srd = round(sd + rd[None, :, :]).type(torch.LongTensor)
        self.supp = int(2*torch.abs(srd).max() + 1)
        kernels = torch.zeros((srd.shape[0], self.supp, self.supp), dtype=depth.dtype, device=depth.device)
        kernels = self.get_kernels(srd, kernels, self.supp)
        kernels = kernels / torch.sum(kernels,dim=[1,2], keepdim=True)

        # Save kernels
        self.kernels = kernels
        # Plot kernels
        if show_plot:
            self.plot_kernels()

    def __len__(self):
        return self.kernels.shape[0]

    def __call__(self) -> torch.Tensor:
        return self.kernels

    def kernel_size(self):
        return (self.supp, self.supp)

    @torch.jit.script
    def get_kernels(srd:torch.Tensor, kernels:torch.Tensor, supp:int):
        """Generate kernels from discretized trajectory"""
        for l in range(srd.shape[0]):
            for i, j in zip(-srd[l, :, 1] + supp // 2, srd[l, :, 0] + supp // 2):
                kernels[l, i, j] += 1
        return kernels

    def plot_kernels(self):
        """Plot kernels"""
        num_kernels = self.__len__()
        num_axes = int(np.ceil(np.sqrt(num_kernels)))
        fig, axes = plt.subplots(num_axes, num_axes)
        for n, ax in enumerate(axes.ravel()):
            if n < num_kernels:
                ax.imshow(tensor_to_numpy(self.kernels[n]))
            ax.axis('off')
        plt.show()



class PointwiseConvKernels(object):
    """Class for the computation of the pointwise convolution blur kernels

    Args:
    ----------
    depth (torch.Tensor):
        depth map.
    camera_motion (modules.trajectories.Trajectory6D):
        6DoF camera trajectory.
    camera_intrinsics (modules.utils.CameraParameters):
        Camera intrinsic parameters.
    """
    def __init__(self, 
                 depth: torch.Tensor, 
                 camera_motion: Trajectory6D, 
                 camera_intrinsics: CameraParameters
                ):
        
        @njit(parallel=True)
        def compute_histogram(S, N, T, kernels, nbins):
            """Compute histogram of the trajectory discretization"""
            for n in prange(N):
                for t in prange(T):
                    kernels[n, -S[n,t, 1] + nbins // 2, S[n,t, 0] + nbins // 2] +=1    
            return kernels
    

        blur_trajectory = camera_motion
        self.img_size = depth.shape

        # Compute rotation matrices
        Rs = torch.from_numpy(Rotation.from_euler('xyz', blur_trajectory.rot, degrees=True).as_matrix()).type(torch.float32)
        # Compute translation vectors
        Ts = torch.from_numpy(blur_trajectory.trans).type(torch.float32)
        # Compute pixel grid
        x, y = torch.meshgrid(torch.arange(self.img_size[1], dtype=torch.float32), torch.arange(self.img_size[0], dtype=torch.float32), indexing='xy')
        z = torch.ones_like(x, dtype=torch.float32)
        # Pixel coordinates in homogeneous coordinates
        X = torch.stack([x.ravel(), y.ravel(), z.ravel()])
        # Camera intrinsics
        self.C = torch.from_numpy(camera_intrinsics.matrix()).type(torch.float32)
        self.C_inv = torch.linalg.inv(self.C)
        # reshape depth
        d = depth.ravel()
        # Initialize pointwise motion
        self.S = torch.zeros((len(blur_trajectory), 2, X.shape[1]), dtype=torch.long)
        # Compute pointwise motion
        self.point_wise_motion(Ts, Rs, X, d)
        
        # Compute pointwise convolution kernels
        self.S  = torch.moveaxis(self.S , -1, 0)
        self.nbins = int(2 * torch.abs(self.S ).max() + 1)
        kernels = torch.zeros((X.shape[1], self.nbins, self.nbins), dtype=torch.float32)
        self.kernels = torch.from_numpy(compute_histogram(self.S.numpy(), X.shape[1], self.S.shape[1], kernels.numpy(), self.nbins))
        # fix kernels with zero sum
        impulse = torch.zeros((self.nbins, self.nbins), dtype=torch.float32, device=self.kernels.device)
        impulse[self.nbins // 2 , self.nbins // 2 ] = 1
        tag = torch.argwhere(torch.sum(self.kernels, dim=(1,2)) == 0)
        self.kernels[tag] = impulse
        # Normalize kernels
        self.kernels = self.kernels/ torch.sum(self.kernels, dim=(1,2), keepdim=True)

    def __len__(self):
        return len(self.kernels)

    def kernel_size(self):
        return (self.nbins, self.nbins)

    def __call__(self) -> torch.Tensor:
        """Return the kernels"""
        kernels = self.kernels.view(1,*self.img_size,-1).permute(0,3,1,2)   #(1,bins**2,H,W)
        return torch.flip(kernels, dims=(1,))

    def point_wise_motion(self, trans_vecs, rot_mats, X, d):
        """Compute the pointwise motion of given translation vectors, rotation matrices, and depth map
        
        Args:
        ----------
            trans_vecs (torch.Tensor):
                translation vectors.
            rot_mats (torch.Tensor):
                rotation matrices.
            X (torch.Tensor):
                Pixels in homogenous coordinates.
            d (torch.Tensor):
                Depth map.
        """
        for i , (T, R) in enumerate(zip(trans_vecs, rot_mats)):
            Td = torch.matmul(torch.outer(torch.reciprocal(d), T)[:,:,None], torch.tensor([[0.0, 0.0, 1.0]]))
            P = torch.swapaxes(torch.tensordot(self.C, torch.matmul((R + Td), self.C_inv), dims=([1],[1])), 0, 1)
            Xp = torch.einsum('hij,jh->ih', P, X)
            Xp_hat = Xp/Xp[2,:]
            self.S[i,:,:] = (round(Xp_hat - X))[0:2,:]
    
