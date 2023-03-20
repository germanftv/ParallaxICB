import numpy as np
import os
from distutils.spawn import find_executable
from modules.utils import read_log, savefig
from modules.utils import CameraParameters, CameraLog, search_with_pattern
from transforms3d import affines, euler

if find_executable('latex'):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 24, 'text.usetex': True})

import matplotlib.pyplot as plt


class Trajectory2D:
    """ Class for 2D camera motion trajectory.

    Args:
    ----------
        x (np.ndarray):
            [N] array containing X camera translation component.
        y (np.ndarray):
            [N] array containing Y camera rotational component.
    """
    def __init__(self, x, y):
        assert type(x) is np.ndarray
        assert type(y) is np.ndarray
        assert x.shape == y.shape
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        return Trajectory2D(x, y)

    def __mul__(self, a):
        return a * np.stack([self.x, self.y], axis=-1)

    def __add__(self, a):
        return a + np.stack([self.x, self.y], axis=-1)

    def __truediv__(self, a):
        if a is list or np.ndarray:
            return np.stack([self.x / a[0], self.y / a[1]], axis=-1)
        else:
            return np.stack([self.x, self.y], axis=-1) / a

    def __neg__(self):
        return np.stack([-self.x, -self.y], axis=-1)

    def plot(self, ax=None):
        """ Plot trajectory in time-agnostic way"""
        if ax is None:
            fig, ax = plt.subplots()
        plt.plot(self.x * 1E3, self.y * 1E3, '--k')
        plt.xlabel(r'$x$ [mm]', fontsize=26)
        plt.ylabel(r'$y$ [mm]', fontsize=26)
        plt.axis('equal')


class Trajectory6D:
    """ Class for 6D camera motion trajectory.

    Args:
    ----------
        trans (np.ndarray):
            [Nx3] array containing XYZ camera translation components.
        rot (np.ndarray):
            [Nx3] array containing XYZ camera rotational components.
    """
    def __init__(self, trans, rot):
        assert type(trans) is np.ndarray
        assert type(rot) is np.ndarray
        assert trans.shape == rot.shape
        self.trans = trans
        self.rot = rot

    def __neg__(self):
        return Trajectory6D(-self.trans, -self.rot)

    def __len__(self):
        return self.trans.shape[0]

    def __getitem__(self, item):
        trans = self.trans[item]
        rot = self.rot[item]
        return Trajectory6D(trans, rot)


class Kohler6DCameraMotion(object):
    """Class for reading camera trajectories in Kohler dataset

    Args:
    ----------
        filename (str):
            Path to file in Kohler dataset.
        translations (str, optional):
            Translational components to be read. Default: 'xyz'
        rotations (str, optional):
            Rotational components to be read. Default: 'xyz'
        exposure (float, optional):
            Exposure limit. Default: Whole trajectory
        center (bool, optional):
            Center trajectory. Default: True
        path_to_save (str, optional):
            Path to save plot with frame-wise camera motion. Default: None
    """
    def __init__(self,
                 filename: str,
                 translations: str = 'xyz',
                 rotations: str = 'xyz',
                 exposure=None,
                 center=True,
                 path_to_save=None):

        assert filename.endswith('txt')
        fs = float(os.path.basename(os.path.dirname(filename)).replace('Frames', ''))
        t_list = list(translations)
        r_list = list(rotations)
        walk = np.loadtxt(filename, delimiter=',')
        walk = np.concatenate([np.zeros([1, walk.shape[1]]), walk], axis=0)
        tx, ty, tz = walk[:, 3], walk[:, 5], walk[:, 4]
        for t_ax in {'x', 'y', 'z'}.difference(t_list):
            if t_ax == 'x':
                tx = np.zeros_like(walk[:, 3])
            if t_ax == 'y':
                ty = np.zeros_like(walk[:, 3])
            if t_ax == 'z':
                tz = np.zeros_like(walk[:, 3])
        trans = np.stack([tx, ty, tz], axis=1) * 1E-3
        rx, ry, rz = walk[:, 0], walk[:, 2], walk[:, 1]
        for r_ax in {'x', 'y', 'z'}.difference(r_list):
            if r_ax == 'x':
                rx = np.zeros_like(walk[:, 3])
            if r_ax == 'y':
                ry = np.zeros_like(walk[:, 3])
            if r_ax == 'z':
                rz = np.zeros_like(walk[:, 3])
        rot = np.stack([rx, ry, rz], axis=1)

        if exposure is not None:
            idx = np.arange(0,len(tx))
            valid_idx = idx[idx<round(exposure*fs)]
            trans = trans[valid_idx]
            rot = rot[valid_idx]

        if center:
            total_samples = trans.shape[0]
            trans = trans - trans[total_samples//2 + 1]
            rot = rot - rot[total_samples//2 + 1]
        
        self.trajectory = Trajectory6D(-trans, -rot)
        self.readout = {'trans': translations, 'rot': rotations}
        self.iters = len(self.trajectory)
        self.path_to_save = path_to_save
        self.exposure = exposure

    def plot_trajectory(self, figsize=(8,6), save: bool = False):
        """Plot read component-wise camera trajectory"""
        t = np.linspace(0, self.exposure, self.iters)
        plt.figure(figsize=figsize)
        if self.readout['trans'] != '' and self.readout['rot'] != '':
            plt.subplot(121)
            plt.plot(t, self.trajectory.trans * 1E3)
            plt.xlabel(r'Time [s]')
            plt.ylabel(r'Translation [mm]')
            plt.xlim([0, self.exposure])
            plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
            plt.subplot(122)
            plt.plot(t, self.trajectory.rot)
            plt.xlabel(r'Time [s]')
            plt.ylabel(r'Rotation [$^\circ$]')
            plt.xlim([0, self.exposure])
            plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
            plt.tight_layout()
        elif self.readout['trans'] != '':
            plt.plot(t, self.trajectory.trans*1E3)
            plt.xlabel(r'Time [s]')
            plt.ylabel(r'Translation [mm]')
            plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
            plt.xlim([0, self.exposure])
            plt.tight_layout()
        elif self.readout['rot'] != '':
            plt.plot(t, self.trajectory.rot)
            plt.xlabel(r'Time [s]')
            plt.ylabel(r'Rotation [$^\circ$]')
            plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
            plt.xlim([0, self.exposure])
            plt.tight_layout()
        else:
            raise Exception('empty trajectory')
        if save:
            if self.path_to_save is None:
                raise Exception('Please create Trajectory instance with path_to_save')
            savefig(self.path_to_save)


class RealCMBBundle(object):
    """Class for reading bundle info in RealCMB dataset

    Args:
        bundle_info (list):
            List of bundle info
    """
    def __init__(self,
                 bundle_info: list):
        self.bundle_info = bundle_info
        # Define timestamps
        self.timestamps = np.array([info['timestamp'] for info in bundle_info])

    def camera_motion(self, ref_idx=None, show=False, **kargs):
        """Compute camera motion between frames
        
        Args:
        -----
            ref_idx (int, optional):
                Index of reference frame. Default is the middle frame
            show (bool, optional):
                Show camera motion. Default is False
            kargs (dict):
                Keyword arguments for plot_camera_motion
        
        Returns:
        --------
            traj (modules.trajectory.Trajectory6D):
                Camera motion trajectory
        
        Notes:
        ------
            Adapted from Chugonov et al. (2022) official code `<https://github.com/princeton-computational-imaging/hndr>`_.
        """

        def eye_like(mat):
            """Return identity matrix with same shape as mat"""
            return np.eye(mat.shape[0])
        
        # Default reference frame is the middle frame
        ref_idx = len(self.bundle_info) //2 +1 if ref_idx == None else ref_idx
        assert ref_idx in range(len(self.bundle_info))
        # Read reference frame info
        ref_info = self.bundle_info[ref_idx]
        ref_world_to_camera = ref_info['world_to_camera']
        ref_camera_to_world = np.linalg.inv(ref_world_to_camera)

        # Compute camera motion
        forward_transforms = []
        trans_list = []
        rot_list = []
        for idx in range(len(self.bundle_info)): 
            if idx == ref_idx:
                forward_transform = eye_like(ref_camera_to_world)
            else:
                qry_world_to_camera = self.bundle_info[idx]['world_to_camera']
                forward_transform = qry_world_to_camera @ ref_camera_to_world 

            forward_transforms.append(forward_transform)
            Tdash, Rdash, Zdash, Sdash = affines.decompose44(forward_transform)
            rz, rx, ry = euler.mat2euler(Rdash, 'rzxy')
            tx, ty, tz = Tdash
            rot = -np.array([rx, ry, rz])
            trans = -np.array([tx, ty, tz])
            trans_list.append(Tdash)
            rot_list.append(rot)

        trans = np.array(trans_list) # *np.array([1, -1, -1])
        rot = np.array(rot_list) * 180/np.pi

        traj = Trajectory6D(trans, rot)

        if show:
            self.plot_trajectory(traj, self.timestamps, **kargs)

        return traj

    
    def camera_intrinsics(self, ref_idx=None):
        """Retrieve camera intrinsics"""
        p_size = 1.4 * 1E-6     # pixel size for Iphone 12 pro max (arbitrary)
        # Default reference frame is the middle frame
        ref_idx = len(self.bundle_info) //2 + 1 if ref_idx == None else ref_idx
        # Read reference frame intrinsics
        ref_intrinsics = self.bundle_info[ref_idx]['intrinsics']
        params = {
            'focal_length': ref_intrinsics[0,0] * p_size,
            'principal_point':(ref_intrinsics[0,2] * p_size, ref_intrinsics[1,2] * p_size),
            'aspect_ratio': ref_intrinsics[1,1] / ref_intrinsics[0,0],
            'skew_coefficient': 0,
            'pixel_size': (p_size , p_size)

        }
        camera_intrinsics = CameraParameters(params)

        return camera_intrinsics

    @staticmethod
    def plot_trajectory(trajectory, timestamps=None, figsize=(8,6), path_to_save= None):
        """Plot read component-wise camera trajectory"""

        def xlabel(timestamps):
            return r'Time [s]' if timestamps is not None else r'Frame number'

        t = timestamps if timestamps is not None else np.arange(0, len(trajectory))
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.plot(t, trajectory.trans * 1E3)
        plt.xlabel(xlabel(timestamps))
        plt.ylabel(r'Translation [mm]')
        plt.xlim([0, t[-1]])
        plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
        plt.subplot(122)
        plt.plot(t, trajectory.rot)
        plt.xlabel(xlabel(timestamps))
        plt.ylabel(r'Rotation [$^\circ$]')
        plt.xlim([0, t[-1]])
        plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
        plt.tight_layout()
        if path_to_save is not None:
            savefig(path_to_save)


class VirtualCMB_CamLogs(object):
    """Class for reading bundle info in VirtualCMB dataset

    Args:
    -----
        camera_logs_dir (str):
            Path to camera logs directory
        sampling_freq (float):
            Sampling frequency of camera logs
    """
    def __init__(self,
                 camera_logs_dir: str,
                 sampling_freq: float):

        self.camera_logs_dir = camera_logs_dir
        # Read camera logs
        camera_logs_path = search_with_pattern(camera_logs_dir, 'camera_logs_*.txt')
        camera_logs_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.camera_logs = [read_log(log_path) for log_path in camera_logs_path]
        self.Fs = sampling_freq
        # Define timestamps
        self.timestamps = 1 / sampling_freq * np.arange(0, len(self.camera_logs))

    def camera_motion(self, ref_idx=None, show=False, **kargs):
        """Retrieve camera motion
        
        Args:
        -----
            ref_idx (int, optional):
                Index of reference frame. Default is the middle frame.
            show (bool, optional):
                Whether to plot the camera motion. Default is False.
            **kargs:
                Additional arguments for plotting function.
        
        Returns:
        --------
            traj (motion.trajectory.Trajectory6D):
                Camera motion trajectory
        """
        
        def eye_like(mat):
            """Return identity matrix of same shape as mat"""
            return np.eye(mat.shape[0])
        # Default reference frame is the middle frame
        ref_idx = len(self.camera_logs) //2 +1 if ref_idx == None else ref_idx
        assert ref_idx in range(len(self.camera_logs))
        # Read reference frame log
        ref_info = CameraLog().from_log(self.camera_logs[ref_idx])
        ref_camera_to_world = ref_info.camSet2WorldMat

        # Compute camera motion
        forward_transforms = []
        trans_list = []
        rot_list = []
        for idx, camera_log in enumerate(map(CameraLog().from_log, self.camera_logs)):
            if idx == ref_idx:
                forward_transform = eye_like(ref_camera_to_world)
            else:
                qry_world_to_camera = camera_log.world2CamSetMat
                forward_transform = qry_world_to_camera @ ref_camera_to_world

            forward_transforms.append(forward_transform)
            Tdash, Rdash, Zdash, Sdash = affines.decompose44(forward_transform)
            rz, rx, ry = euler.mat2euler(Rdash, 'rzxy')
            rot = np.array([rx, ry, rz])
            trans_list.append(Tdash)
            rot_list.append(rot)

        trans = np.array(trans_list) # *np.array([1, -1, -1])
        rot = np.array(rot_list) * 180/np.pi

        traj = Trajectory6D(trans, rot)

        if show:
            self.plot_trajectory(traj, self.timestamps, **kargs)

        return traj

    
    def camera_intrinsics(self, ref_idx=None):
        """Retrieve camera intrinsics"""
        ref_idx = len(self.camera_logs) //2 + 1 if ref_idx == None else ref_idx
        camera_intrinsics = CameraParameters().from_log(self.camera_logs[ref_idx]['cameraIntrinsics'])
        return camera_intrinsics

    def ref_cam_log(self, ref_idx=None):
        """Retrieve reference camera log"""
        ref_idx = len(self.camera_logs) //2 + 1 if ref_idx == None else ref_idx
        return self.camera_logs[ref_idx]


    @staticmethod
    def plot_trajectory(trajectory, timestamps=None, figsize=(8,6), path_to_save= None):
        """Plot read component-wise camera trajectory"""

        def xlabel(timestamps):
            return r'Time [s]' if timestamps is not None else r'Frame number'

        t = timestamps if timestamps is not None else np.arange(0, len(trajectory))
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.plot(t, trajectory.trans * 1E3)
        plt.xlabel(xlabel(timestamps))
        plt.ylabel(r'Translation [mm]')
        plt.xlim([0, t[-1]])
        plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
        plt.subplot(122)
        plt.plot(t, trajectory.rot)
        plt.xlabel(xlabel(timestamps))
        plt.ylabel(r'Rotation [$^\circ$]')
        plt.xlim([0, t[-1]])
        plt.legend([r'$x$-axis', r'$y$-axis', r'$z$-axis'])
        plt.tight_layout()
        if path_to_save is not None:
            savefig(path_to_save)
