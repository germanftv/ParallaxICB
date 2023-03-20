import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lpips
from collections import OrderedDict
from timeit import default_timer as timer
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from data.utils import IMG_INV_TRANSFORMS, DEPTH_INV_TRANSFORMS
from modules.utils import get_model_size


def fourier_mapping(x, B):
    """Fourier mapping.
    
    Args:
    ----------
        x (torch.Tensor): Input tensor.
        B (torch.Tensor): Fourier basis.
        
    Returns:
    ----------
        torch.Tensor: Fourier mapped tensor.
    
    Notes:
    ----------
        Adapted from Tancik et al. (2020) `<https://github.com/tancik/fourier-feature-networks>`_.
"""
    
    if B is None:
        return x
    else:
        x_proj = (2.*torch.pi*x) @ B.T
        return torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SineLayer(nn.Module):
    """Sine layer.
    
    Args:
    ----------
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to use bias. Defaults to True.
        is_first (bool, optional): Whether it is the first layer. Defaults to False.
        omega_0 (int, optional): Omega_0. Defaults to 30.
        
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_."""
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    """Siren model.
    
    Args:
    ----------
        in_features (int): 
            Number of input features.
        hidden_features (int): 
            Number of hidden features.
        hidden_layers (int): 
            Number of hidden layers.
        out_features (int): 
            Number of output features.
        outermost_linear (bool, optional): 
            Whether to use linear activation for the outermost layer. Defaults to False.
        first_omega_0 (int, optional): 
            Omega_0 for the first layer. Defaults to 30.
        hidden_omega_0 (int, optional): 
            Omega_0 for the hidden layers. Defaults to 30.
        fourier_scale (float, optional): 
            Fourier scale. Defaults to None.
        
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., fourier_scale=None):
        super().__init__()
        
        self.net = []
        if fourier_scale is None:
            self.B = None
            self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        elif fourier_scale == 'eye':
            self.B = torch.eye(2)
            self.net.append(SineLayer(in_features*2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        else:
            self.B = torch.normal(0, fourier_scale, (32, in_features))
            self.net.append(SineLayer(32*2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        if fourier_scale is not None:
            self.weights_init()
    
    def weights_init(self):
        self.B = nn.Parameter(self.B, requires_grad=False)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(fourier_mapping(coords, self.B))
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def net_laplace(y, x):
    """Compute the Laplacian.
    
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_.
    """
    grad = net_gradient(y, x)
    return net_divergence(grad, x)


def net_divergence(y, x):
    """"Compute the divergence
    
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_.
    """
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def net_gradient(y, x, grad_outputs=None):
    """Compute the gradient.
    
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad




class CoordinateBasedMLP(nn.Module):
    """Coordinate-based MLP.
    
    Args:
    ----------
        in_features (int):
            Number of input features.
        hidden_features (int):
            Number of hidden features.
        hidden_layers (int):
            Number of hidden layers.
        out_features (int):
            Number of output features.
        outermost_linear (bool, optional):
            Whether to use linear activation for the outermost layer. Defaults to False.
        fourier_scale (float, optional):
            Fourier scale. Defaults to None.
        
    Notes:
    ----------
        Adapted from Tancik et al. (2020) `<https://github.com/tancik/fourier-feature-networks>`_. 
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, fourier_scale=None):
        super().__init__()
        
        self.net = []
        if fourier_scale is None:
            self.B = None
            self.net.append(nn.Linear(in_features, hidden_features))
        else:
            self.B = torch.normal(0, fourier_scale, (hidden_features//2, in_features))
            # self.B = torch.randn(hidden_features//2, in_features) * fourier_scale
            self.net.append(nn.Linear(hidden_features, hidden_features))


        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            self.net.append(final_linear)
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
        if fourier_scale is not None:
            self.weights_init()
    
    def weights_init(self):
        self.B = nn.Parameter(self.B, requires_grad=False)
    
    def forward(self, x):
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(fourier_mapping(x, self.B))
        return output, x    


class SobelLayer(nn.Module):
    """Sobel filter layer.
    
    Args:
    ----------
        p_norm (int): p-norm to use.
    """
    def __init__(self, p_norm) -> None:
        super().__init__()
        self.gx_filter = nn.Conv2d(3, 1, (3, 3), bias=False, padding=(3//2, 3//2), padding_mode='replicate')
        self.gy_filter = nn.Conv2d(3, 1, (3, 3), bias=False, padding=(3//2, 3//2), padding_mode='replicate')
        self.p_norm = p_norm
        self.weights_init()
    
    def weights_init(self):
        """Initialize weights of the Sobel filter."""
        weights_gy = torch.stack([1/3 * torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) for _ in range(3)])
        weights_gx = torch.stack([1/3 * torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) for _ in range(3)])
        self.gx_filter.weight = nn.Parameter(weights_gx.view(1,3,3,3).type(torch.FloatTensor), requires_grad=False)
        self.gy_filter.weight = nn.Parameter(weights_gy.view(1,3,3,3).type(torch.FloatTensor), requires_grad=False)

    def forward(self, x):
        """Forward pass.
        
        Args:
        ----------
            x (torch.Tensor): Input tensor.

        Returns:
        ----------
            torch.Tensor: Gradient magnitude."""
        gx = self.gx_filter(x)
        gy = self.gy_filter(x)

        return torch.norm(torch.cat([gx,gy], dim=1), dim=1, p=self.p_norm, keepdim=True)


class GradientFilter(nn.Module):
    """Gradient filter layer.
    
    Args:
    ----------
        p_norm (int): p-norm to use.
    """
    def __init__(self, p_norm) -> None:
        super().__init__()
        self.gx_filter = nn.Conv2d(3, 1, (1, 3), bias=False, padding=(1,0), padding_mode='replicate')
        self.gy_filter = nn.Conv2d(3, 1, (3, 1), bias=False, padding=(0,1), padding_mode='replicate')
        self.p_norm = p_norm
        self.weights_init()
    
    def weights_init(self):
        """Initialize weights of the Sobel filter."""
        weights_gy = torch.stack([torch.tensor([[-1/3, 0, 1/3]]) for _ in range(3)])
        weights_gx = torch.stack([torch.tensor([[-1/3], [0], [1/3]]) for _ in range(3)])
        self.gx_filter.weight = nn.Parameter(weights_gx.view(1,3,3,1).type(torch.FloatTensor), requires_grad=False)
        self.gy_filter.weight = nn.Parameter(weights_gy.view(1,3,1,3).type(torch.FloatTensor), requires_grad=False)

    def forward(self, x):
        """Forward pass.
        
        Args:
        ----------
            x (torch.Tensor): Input tensor.

        Returns:
        ----------
            """
        gx = self.gx_filter(x)
        gy = self.gy_filter(x)

        return torch.norm(torch.cat([gx,gy], dim=1), dim=1, p=self.p_norm, keepdim=True)

nn_dict = {
    'SIREN': Siren,
    'FOURIER_MAPPED_MLP': CoordinateBasedMLP,
}


def deblurring(blur_dataset, blur_nn: nn.Module, deblur_params:dict, device:torch.DeviceObjType, load_ckpt:bool=False, ckpt_path:str=None):
    """Sharp neural representations from blur function.
    
    Args:
    ----------
        blur_dataset (data.datasets.BaseCMBFitting): Blur dataset.
        blur_nn (nn.Module): Blurring neural network.
        deblur_params (dict): Deblurring parameters.
        device (torch.DeviceObjType): Device to use.
        load_ckpt (bool): Whether to load checkpoint.
        ckpt_path (str): Checkpoint path.

    Returns:
    ----------
        dict: Deblurring evaluation metrics.
        numpy.ndarray: Deblurred image.
        numpy.ndarray: Ground truth image.
        nn.Module: Implicit neural representation.
    """

    # Set up dataloader
    batch_size = 1
    train_dataloader = DataLoader(blur_dataset, batch_size=batch_size, pin_memory=True, num_workers=0)
    # lpips network
    lpips_fn = lpips.LPIPS(net='vgg')

    start_time = timer()
    # Set up implicit neural network
    nn_model = nn_dict[deblur_params['nn_model']]
    mlp_nn = nn_model(in_features=2, out_features=3, **deblur_params[deblur_params['nn_model']], outermost_linear=True)
    mlp_nn.to(device)

    # Set up optimization parameters
    num_epochs = deblur_params['num_epochs']
    optimizer = torch.optim.Adam(lr=deblur_params['lr'], params=mlp_nn.parameters())
    clip_grad = deblur_params['clip_grad']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=deblur_params['scheduler_eta_min'])
    
    criterion = nn.MSELoss()
    if deblur_params['gradient_fn'] == 'filter':
        grad_nn = GradientFilter(p_norm=deblur_params['p_norm'])
        grad_nn.to(device)
    elif deblur_params['gradient_fn'] == 'net_grad':
        grad_nn = net_gradient
    else:
        raise ValueError("Not implemented 'gradient_fn': {}".format(deblur_params['gradient_fn']))
    beta = deblur_params['gradient_weight']
    blur_nn.to(device)
    
    padding = (blur_nn.ks//2)
    img_padded_size = list(map(lambda x: x + 2*padding, blur_dataset.img_size))
    patch_padded_size = list(map(lambda x: x + 2*padding, blur_dataset.patch_size))

    if not load_ckpt:
        # Training loop
        for _ in tqdm(range(num_epochs)):
            # Iterate over batches
            for sample in train_dataloader:
                input_coords, gt_blur = sample['coords'].to(device), sample['blurry'].to(device)
                idx = sample['idx']
                optimizer.zero_grad()
                # Evaluate implicit MLP
                mlp_sharp_ravel, mlp_coords = mlp_nn(input_coords.permute(0,2,3,1).view(-1,2))
                mlp_sharp = mlp_sharp_ravel.view(1,*patch_padded_size,3).permute(0,3,1,2)
                # Evaluate blur function
                pred_blur = blur_nn(mlp_sharp, idx)
                # Compute loss
                loss = criterion(pred_blur, gt_blur)
                if deblur_params['gradient_fn'] == 'filter':
                    mlp_grad = grad_nn(mlp_sharp)
                    mlp_grad = torch.squeeze(mlp_grad)
                    mlp_grad = mlp_grad.permute(1, 0).reshape((-1, 1))
                    loss += beta * mlp_grad.mean()
                elif deblur_params['gradient_fn'] == "net_grad":
                    mlp_grad = grad_nn(mlp_sharp_ravel, mlp_coords)
                    loss += beta * torch.norm(mlp_grad, dim=1, p=deblur_params['p_norm']).mean()

                loss.backward()
                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(mlp_nn.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(mlp_nn.parameters(), max_norm=clip_grad)
                optimizer.step()
            scheduler.step()
    else:
        mlp_nn.load_state_dict(torch.load(ckpt_path))
        mlp_nn.eval()

    pred_sharp = []
    # Evaluate sharp image from implicit MLP
    with torch.no_grad():
        # Iterate over batches
        for sample in train_dataloader:
            input_coords = sample['coords'].to(device)
            mlp_sharp, _ = mlp_nn(input_coords.permute(0,2,3,1).view(-1,2))
            mlp_sharp = mlp_sharp.view(1,*patch_padded_size,3).permute(0,3,1,2)
            pred_sharp.append(mlp_sharp[0,:,padding:-padding,padding:-padding])
    
    # Reconstruct sharp image
    sharp_est = torch.stack(pred_sharp, dim=-1).view(1,-1,len(blur_dataset))
    sharp_est = F.fold(sharp_est, output_size=blur_dataset.img_size, kernel_size=blur_dataset.patch_size, stride=blur_dataset.patch_size)
    end_time = timer()

    # Compute metrics
    sharp_gt = blur_dataset.sharp
    sharp_est = IMG_INV_TRANSFORMS(sharp_est[0]).astype(np.float64)
    psnr = peak_signal_noise_ratio(sharp_gt, sharp_est)
    ssim = structural_similarity(sharp_gt, sharp_est,  channel_axis=2)
    lpips_value = lpips_fn.forward(lpips.im2tensor(sharp_gt), lpips.im2tensor(sharp_est)).detach().numpy().squeeze()
    elapsed_time = end_time - start_time
    model_size = get_model_size(mlp_nn)
    # Concatenate metrics
    perf = {
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips_value,
        'elapsedTime': elapsed_time,
        'modelSize': model_size
    }

    return perf, sharp_est, sharp_gt, mlp_nn


    