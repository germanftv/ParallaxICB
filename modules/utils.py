from __future__ import print_function, division
import cv2
import os
import numpy as np
import warnings
import pathlib
import json
import torch
import matplotlib
matplotlib.use('Agg' ,force=True)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from distutils.spawn import find_executable

if find_executable('latex'):
    matplotlib.rcParams.update({'font.size': 36, 'text.usetex': True})
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


# Valid image extensions
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

# Valid figure extensions
FIG_EXTENSIONS = ['.pdf', '.ps', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
                  '.tif']


def is_image_file(path: str) -> bool:
    """Check file for valid image formats."""
    return any(path.endswith(extension) for extension in IMG_EXTENSIONS)


def is_fig_file(path: str) -> bool:
    """Check file for valid image formats."""
    return any(path.endswith(extension) for extension in FIG_EXTENSIONS)


def get_image_path_list(path: str) -> list:
    """Get image path list from image folder.

    Args:
    ----------
        path (str): Path to image folder.
    
    Returns
    -------
        list: List of image paths.
    """

    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def search_with_pattern(root: str, pattern: str) -> list:
    """Search in directory by pattern.

    Args:
    ----------
    root(str): Root directory.
    pattern(str): Pattern of search.

    Returns
    -------
    list: List of paths.
    """
    output = [str(path) for path in pathlib.Path(root).rglob(pattern)]
    output.sort()
    return output


def get_subfolders(root: str) -> list:
    """Get immediate subfolders in directory

    Args:
    ----------
    root(str): Root directory.
    
    Returns
    -------
    list: List with subfolder paths.
    """
    output = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    output.sort()
    return output


def imread(path: str, image_space: str = 'RGB', image_range=(0.0, 1.0)) -> np.ndarray:
    """Read image file.

    Args:
    ----------
    path(str): Path to image file.
    image_space(str): Color space of image. Default: 'RGB'
    image_range(tuple): Range of image values. Default: (0.0, 1.0)
    
    Returns
    -------
    img(np.ndarray): numpy array with image.
    """

    assert is_image_file(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Transform color space as retrieve in opts
    img = img if image_space == 'GRB' else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize input range
    max_value = np.iinfo(img.dtype).max
    img = img.astype(np.float64) / max_value
    a, b = image_range
    img = (b-a)*img + a
    return img


def imshow(img: np.ndarray, ax: plt.Subplot = None):
    """Show image.

    Args:
    ----------
    img(np.ndarray): Image to show.
    ax(plt.Subplot): Subplot object. Default: None
    """

    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(img)
    ax.axis('off')


def depthshow(depth: np.ndarray, inf_value: float = 1000, ax: plt.Subplot = None):
    """Show depth map

    Args:
    ----------
        depth(np.ndarray): Depth map to show.
        inf_value(float): Depth value at infinity. Default: 1000
        ax(plt.Subplot): Subplot object. Default: None
    """
    depth[depth == inf_value] = np.nan
    if ax is None:
        plt.figure()
        ax = plt.gca()
    im = ax.imshow(depth, norm=LogNorm())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis('off')
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=28) 
    cbar.set_label('$D(\mathbf{p})$ [m]', rotation=270, fontsize=32, labelpad=18)


def read_RGBA_encoded_depth(path: str, inf_value: float = 1000, clipping: tuple = (0.01, 4)) -> np.ndarray:
    """Read depth map file using a RGBA encoding method.

    Args:
    ----------
        path(str): Path to depth map file.
        inf_value(float): Depth value at infinity. Default: 1000
        clipping(tuple): Clipping depth range. Default: (0.01, 4)

    Returns
    -------
        depth(np.ndarray): Depth map.
    """

    depth_BGR = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth_RGB = cv2.cvtColor(depth_BGR, cv2.COLOR_BGR2RGB)
    encoded_depth = cv2.cvtColor(depth_RGB, cv2.COLOR_RGB2RGBA)
    dec_fun = np.array([1.0, 1 / 255.0, 1 / 65025.0, 1 / 16581375.0])
    depth = np.dot(encoded_depth.astype(np.float), dec_fun) / 255.0 * clipping[1]
    depth[depth < clipping[0]] = inf_value
    return depth


def imsave(img: np.ndarray, path: str, image_space='RGB'):
    """Save image to file.

    Args:
    ----------
        img(np.ndarray): Image to save.
        path(str): Path to output image file.
        image_space(str): Color space of image. Default: 'RGB'
    """

    assert is_image_file(path)
    if (img.min(), img.max()) != (0.0, 1.0):
        img = np.clip(img, 0, 1)
        warnings.warn("Image is clipped in the range [0,1] before storage")
    if image_space == 'RGB':
        cv2.imwrite(path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, (img * 255).astype(np.uint8))


def savefig(path: str, fig: plt.Figure = None):
    """Save matplotlib figure

    Args:
    ----------
        path(str): Path to output figure file.
        fig(plt.Figure): Figure object. Default: None
    """

    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, bbox_inches='tight')


def read_log(filename: str) -> dict:
    """Read log file.

    Args:
    ----------
        filename(str): Path to log file with json extension.

    Returns
    -------
        log(dict): Dictionary with log data.
    """
    file = open(filename, "r")
    contents = file.read()
    log = json.loads(contents)
    file.close()
    return log


class CameraLog:
    """Camera log class.
    
    Attributes:
    ----------
    pos(np.ndarray): Camera position.
    eulerAngles(np.ndarray): Camera euler angles.
    focalLength(float): Camera focal length.
    sensorSize(np.ndarray): Camera sensor size.
    imageSize(np.ndarray): Camera image size.
    projMat(np.ndarray): Projection matrix.
    world2CamMat(np.ndarray): World to camera matrix.
    cam2WorldMat(np.ndarray): Camera to world matrix.
    depthRange(np.ndarray): Depth range.
    world2CamSetMat(np.ndarray): World to camera set matrix.
    camSet2WorldMat(np.ndarray): Camera set to world matrix.
    """
    def __init__(self) -> None:
        self.pos = np.zeros(3)
        self.eulerAngles = np.zeros(3)
        self.focalLength = 0.0
        self.sensorSize = np.zeros(2)
        self.imageSize = np.zeros(2)
        self.projMat = np.zeros((4, 4))
        self.world2CamMat = np.zeros((4, 4))
        self.cam2WorldMat = np.zeros((4, 4))
        self.depthRange = np.zeros(2)
        self.world2CamSetMat = np.zeros((4, 4))
        self.camSet2WorldMat = np.zeros((4, 4))
    
    @staticmethod
    def _read_tuple(tuple):
        """Read tuple from log file."""
        return np.array([tuple['x'], tuple['y']])
    
    @staticmethod
    def _read_vec(vec):
        """Read vector from log file."""
        return np.array([vec['x'], vec['y'], vec['z']])
    
    @staticmethod
    def _read_4x4_mat(mat):
        """Read 4x4 matrix from log file."""
        return np.array([[mat['e00'], mat['e01'], mat['e02'], mat['e03']],
                         [mat['e10'], mat['e11'], mat['e12'], mat['e13']],
                         [mat['e20'], mat['e21'], mat['e22'], mat['e23']],
                         [mat['e30'], mat['e31'], mat['e32'], mat['e33']]
        ])
    
    @staticmethod
    def _read_range(rng):
        """Read range from log file."""
        return np.array([rng['min'], rng['max']])

    def from_log(self, camera_log):
        """Read camera log from log file.
        
        Args:
        ----------
            camera_log(dict): Dictionary with camera log data.
        """
        self.pos = self._read_vec(camera_log['cameraPos']['pos'])
        self.eulerAngles = self._read_vec(camera_log['cameraPos']['eulerAngles'])
        self.focalLength = camera_log['cameraIntrinsics']['focalLength']
        self.sensorSize = self._read_tuple(camera_log['cameraIntrinsics']['sensorSize'])
        self.imageSize = self._read_tuple(camera_log['cameraIntrinsics']['imageSize'])
        self.projMat = self._read_4x4_mat(camera_log['cameraIntrinsics']['projectionMatrix'])
        self.world2CamMat = self._read_4x4_mat(camera_log['cameraIntrinsics']['worldToCameraMatrix'])
        self.cam2WorldMat = self._read_4x4_mat(camera_log['cameraIntrinsics']['cameraToWorldMatrix'])
        self.depthRange = self._read_range(camera_log['depthRange'])
        self.camSet2WorldMat = self._read_4x4_mat(camera_log['camSetToWorldMatrix'])
        self.world2CamSetMat = self._read_4x4_mat(camera_log['worldToCamSetMatrix'])

        return self


_DEFAULT_CAMERA_PARAMETERS = {
        'focal_length': 2.8E-3,
        'principal_point': [0.5 * 5.6E-3, 0.5 * 3.2332E-3],
        'aspect_ratio': 1.0,
        'skew_coefficient': 0.0,
        'pixel_size': [5.6E-3 / 640, 3.2332E-3 / 360]
    }


class CameraParameters:
    """Class for camera intrinsic parameters

    Attributes
    ----------
    f(float): Focal length in meters.
    u0(float): Principal point in meters for the x axis.
    v0(float): Principal point in meters for the y axis.
    a(float): Aspect ratio.
    gamma(float): Skew coefficient.
    px_size(float): Pixel size in meters.

    Args:
    ----------
        params(dict): Dictionary including camera intrinsic parameters. Default: `_DEFAULT_CAMERA_PARAMETERS`
 
    """
    def __init__(self, params: dict = None):
        if params is None:
            params = _DEFAULT_CAMERA_PARAMETERS
        valid_params = ['focal_length', 'principal_point', 'aspect_ratio', 'skew_coefficient', 'pixel_size']
        assert all(param in params.keys() for param in valid_params)
        self.f = params['focal_length']
        self.u0 = params['principal_point'][0]
        self.v0 = params['principal_point'][1]
        self.a = params['aspect_ratio']
        self.gamma = params['skew_coefficient']
        self.px_size = params['pixel_size']

    def from_log(self, camera_intrinsics):
        """Read camera intrinsics from log file.
        
        Args:
        ----------
            camera_intrinsics(dict): Dictionary with camera intrinsics data.
        """
        self.f = camera_intrinsics['focalLength'] * 1E-3
        principal_point = (0.5E-3*camera_intrinsics['sensorSize']['x'],
                           0.5E-3*camera_intrinsics['sensorSize']['y'])
        pixel_size = (camera_intrinsics['sensorSize']['x']*1E-3 / camera_intrinsics['imageSize']['x'],
                      camera_intrinsics['sensorSize']['y']*1E-3 / camera_intrinsics['imageSize']['y'])
        self.u0 = principal_point[0] 
        self.v0 = principal_point[1]
        self.a = 1.0
        self.gamma = 0.0
        self.px_size = pixel_size

        return self

    def matrix(self):
        """Get matrix of camera intrinsics"""
        fx = self.f / self.px_size[0]
        fy = self.f * self.a / self.px_size[1]
        u0 = self.u0 / self.px_size[0]
        v0 = self.v0 / self.px_size[1]
        return np.array([[fx, self.gamma, u0], [0, fy, v0], [0, 0, 1]])


def round(n, decimals=0):
    """Round a tensor to a given number of decimals."""
    multiplier = 10 ** decimals
    return torch.floor(n * multiplier + 0.5) / multiplier


def tensor_to_numpy(tensor:torch.Tensor):
    """Convert tensor to numpy array."""
    tensor = tensor.detach()
    if tensor.device != "cpu":
        tensor = tensor.cpu()
    return tensor.numpy()


def save_error_image(estimate:np.ndarray, gt:np.ndarray, path:str, vmax=0.2, dpi:int=10):
    """Save error image between ground truth and estimate.
    
    Args:
    ----------
        estimate(np.ndarray): Estimate image.
        gt(np.ndarray): Ground truth image.
        path(str): Path to save error image.
        vmax(float,optional): Maximum value for error image. Default: 0.2
        dpi(int,optional): Dots per inch. Default: 10
    """

    def rgb2gray(rgb):
        """Convert RGB image to grayscale image."""
        return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

    error = np.abs(rgb2gray(gt) - rgb2gray(estimate))
    h,w = estimate.shape[:2]

    fig = plt.figure(figsize=(w/dpi,h/dpi))
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.axis('off')
    plt.margins(0,0)
    plt.imshow(error, cmap='hot', vmin=0.0, vmax=vmax)
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


def get_model_size(model:torch.nn.Module):
    """Get model size in MB.
    
    Args:
    ----------
        model(torch.nn.Module): Model to get size.
        
    Returns:
    ----------
        size_all_mb(float): Model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb

def float_or_bool(x):
    """Convert string to float or bool."""
    if x.lower() == 'true':
        return True
    elif x.lower() == 'false':
        return False
    else:
        return float(x)