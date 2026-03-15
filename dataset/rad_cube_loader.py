import sys

from torch.utils.data import Dataset, DataLoader
import os
import h5py
import scipy.io
from . import data_preparation
import numpy as np
import torch
from tqdm import tqdm

def collate_fn(batch):

    return

class RADCUBE_DATASET(Dataset):
    """
    Data Loader for the RaDelf dataset. It initialises a dictionary with the paths to the files of the radar
    camera and lidar.
    This is the version for single frame as input, no temporal information.

    Attributes:
        mode: train, val or test
        params: a dictionary with the parameters defined in data_preparation.py
    """
    def __init__(self, mode='train', params=None, multiclss=False):

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise ValueError("mode should be either train, val or test")

        self.dataset_path = params['dataset_path']
        self.train_val_scenes = params['train_val_scenes']
        self.test_scenes = params['test_scenes']

        self.params = params
        self.multiclass = multiclss

        # 太慢了，直接存储成npy文件，每次读取
        # files are named as ELE_Frame_xxx and Pow_Frame_xxx. Lets get all files that matches these
        if mode == 'train' or mode == 'val':
            scene_set = self.train_val_scenes
        else:
            scene_set = self.test_scenes

        # make a dictionary, indices are keys, elevation, power, and gt_paths are values
        self.data_dict = {}
        global_array_index = 0
        # IMPORTANT: It is assumed that the folders structure is as given in the dataset. If the folder
        # structure is changed this will not work.
        for scene_number in scene_set:

            # Here it is assumed the folders structure is as given in the dataset.
            # If modified, this lines have to be changed, specially "Scene" "and RadarCubes"
            scene_dir = self.dataset_path + '/Scene' + str(scene_number)
            cubes_dir = scene_dir + '/RadarCubes'
            all_files = os.listdir(cubes_dir)
            power_files = [file for file in all_files if "Pow_Frame" in file]
            power_numbers = [int(file.split("_")[-1].split(".")[0]) for file in power_files]

            power_numbers.sort()
            indices = power_numbers.copy()
            indices = np.array(indices)

            # if train: Take 9 indices and skip one. 90% training in the train_val dataset
            if mode == 'train':
                reminder = len(indices) % 10

                if reminder != 0:
                    indices_aux = indices[:-reminder]
                    indices_aux = indices_aux.reshape(-1, 10)[:, :9].reshape(-1)
                    indices = np.concatenate([indices_aux, indices[-reminder:]])
                else:
                    indices = indices.reshape(-1, 10)[:, :9].reshape(-1)

            # if val: Skip 9 indices and take the 10th. 10% val in the train_val dataset
            elif mode == 'val':
                reminder = len(indices) % 10
                if reminder != 0:
                    indices = indices[:-reminder]
                indices = indices.reshape(-1, 10)[:, -1].reshape(-1)

            # if test we keep all the indices

            # get timestamp mapping
            timestamps_path = cubes_dir + '/timestamps.mat'
            frame_num_to_timestamp = scipy.io.loadmat(timestamps_path)
            frame_num_to_timestamp = frame_num_to_timestamp["unixDateTime"]   # 自己生成的RadarCube里没有timestamps怎么回事？用rosDS/radar_ososs中的文件名作为它的timestamps
            frame_num_to_timestamp = frame_num_to_timestamp.reshape(-1,1)

            rosDS_path = scene_dir + '/rosDS'
            lidar_path = rosDS_path + '/rslidar_points_clean'
            camera_dir = rosDS_path + '/ueye_left_image_rect_color'
            if params['label_folder'] is not None:
                label_dir = scene_dir + '/' + params['label_folder']    #改成了自动生成的标签

            # get lidar timestamps
            lidar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(lidar_path)
            camera_timestamps_and_paths = data_preparation.get_timestamps_and_paths(camera_dir)
            if params['label_folder'] is not None:
                label_timestamps_and_paths = data_preparation.get_timestamps_and_paths(label_dir)
            for index in indices:
                self.data_dict[global_array_index] = {}

                ## handle radar
                self.data_dict[global_array_index]["elevation_path"] = os.path.join(cubes_dir,
                                                                                    "Ele_Frame_" + str(index) + ".mat")
                self.data_dict[global_array_index]["power_path"] = os.path.join(cubes_dir,
                                                                                "Pow_Frame_" + str(index) + ".mat")

                self.data_dict[global_array_index]["timestamp"] = (frame_num_to_timestamp[index - 1][0]) * 10 ** 9
                self.data_dict[global_array_index]["numpy_cube_path"] = os.path.join(cubes_dir,
                                                                                     "radar_cube_" + str(
                                                                                         index) + ".npy")
                ## handle LiDAR
                closest_lidar_time = data_preparation.closest_timestamp(self.data_dict[global_array_index]["timestamp"],
                                                                        lidar_timestamps_and_paths)
                self.data_dict[global_array_index]["gt_path"] = lidar_timestamps_and_paths[closest_lidar_time]
                self.data_dict[global_array_index]["gt_timestamp"] = closest_lidar_time

                ## handle camera
                closest_cam_time = data_preparation.closest_timestamp(self.data_dict[global_array_index]["timestamp"],
                                                                      camera_timestamps_and_paths)
                self.data_dict[global_array_index]["cam_path"] = camera_timestamps_and_paths[closest_cam_time]
                self.data_dict[global_array_index]["cam_timestamp"] = closest_cam_time

                ## handle label
                if params['label_folder'] is not None:
                    closest_label_time = data_preparation.closest_timestamp(
                        self.data_dict[global_array_index]["timestamp"],
                        label_timestamps_and_paths)
                    self.data_dict[global_array_index]["label_path"] = label_timestamps_and_paths[closest_label_time]
                    self.data_dict[global_array_index]["label_timestamp"] = closest_label_time

                global_array_index = global_array_index + 1

        # print division line
        print("--------------------------------------------------")

        # print(mode + " dataset loaded with " + str(len(self.data_dict)) + " samples")
        print("scenes used: " + str(scene_set))


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):

        # load elevation and power
        target_shape = (500, 128, 240)
        if not self.params['bev']:   
            ele_path = self.data_dict[idx]["elevation_path"]
            try:
                elevation = scipy.io.loadmat(ele_path)
                try:
                    elevation = elevation["elevationIndex"]
                except KeyError:
                    elevation = elevation["elevationIndex_cpu"]
            except NotImplementedError:
                with h5py.File(ele_path, 'r') as f:
                    try:
                        elevation = f['elevationIndex_cpu'][()]
                    except KeyError:
                        elevation = f['elevationIndex'][()]
            elevation = elevation.astype(np.single)
            elevation = np.nan_to_num(elevation, nan=17.0)
            elevation = elevation / 34
            if elevation.shape != target_shape:
                elevation = np.transpose(elevation, (2,1,0))
            # print("ele: ", elevation.shape)

        power_path = self.data_dict[idx]["power_path"]
        try:
            power = scipy.io.loadmat(power_path)["radarCube"]
        except NotImplementedError:
            with h5py.File(power_path, 'r') as f:
                power = f['radarCube'][()]
        power = power.astype(np.single)
        # Hardcoded maximum value after data exploration
        power = power / 8998.5576
        if power.shape != target_shape:
            power = np.transpose(power, (2,1,0))
        
        # print("pow: ", power.shape)
        # combine them into a single cube with 2 channels
        if not self.params['bev']:
            input_cube = np.stack((power, elevation))
        else:
            input_cube = power
        # print("cube: ", input_cube.shape)
        # load gt
        if not self.multiclass:
            gt_path = self.data_dict[idx]["gt_path"]
            gt_cloud = data_preparation.read_pointcloud(gt_path, mode="rs_lidar_clean")
            gt_cube = data_preparation.lidarpc_to_lidarcube(gt_cloud, self.params)
        else:
            label_path = self.data_dict[idx]["label_path"]
            gt_cloud = data_preparation.read_pointcloud(label_path, mode="radar_gt")   
            if self.params['bev']:
                gt_cube = np.max(gt_cloud, axis=0)  #改用生成的标签看看区别，这个是标注在spherical空间的

        item_params = self.data_dict[idx]  # this is a dictionary with all the paths and timestamps

        if not self.params['bev']:
            zero_pad = np.zeros([2, 12, 128, 240], dtype='single')
            input_cube = np.concatenate([input_cube, zero_pad], axis=1)
            zero_pad = np.zeros([2, 512, 128, 8], dtype='single')
            input_cube = np.concatenate([zero_pad, input_cube, zero_pad], axis=3)
            input_cube = np.transpose(input_cube, (0, 2, 1, 3))  # (C, H, W)

        else:
            zero_pad = np.zeros([12, 128, 240], dtype='single')
            input_cube = np.concatenate([input_cube, zero_pad])
            zero_pad = np.zeros([512, 128, 8], dtype='single')
            input_cube = np.concatenate([zero_pad, input_cube, zero_pad], 2)
            input_cube = np.transpose(input_cube, (1, 0, 2))  # (C, H, W)

        if self.multiclass:
            return input_cube, gt_cloud, item_params
        else:
            return input_cube, gt_cube, item_params

    
    #====================mean and std compute==================#
    def compute_mean_var(self, train_loader, val_loader, test_loader):
        n=0
        mean=torch.zeros([16,512,256],dtype=torch.complex128)
        M2=torch.zeros([16,512,256],dtype=torch.complex128)
        for dataloader in [train_loader,val_loader,test_loader]:
            for batch in tqdm(dataloader):
                batch = batch[0]
                batch_size = len(batch)
                batch_mean = torch.mean(batch, axis=0)
                diff = batch - batch_mean
                var_complex = torch.mean(diff ** 2, dim=0)
                print(batch_mean.shape, var_complex.shape)

                # 合并当前批次的统计量到全局统计量
                delta = batch_mean - mean
                mean = (n * mean + batch_size * batch_mean) / (n + batch_size)
                M2 += batch_size * var_complex + delta**2 * n * batch_size / (n + batch_size)
                n += batch_size

        # 最终方差
        variance = M2 / n  # 总体方差
        # sample_variance = M2 / (n - 1)  # 样本方差（无偏估计）
        np.save('mean.npy',mean)
        np.save('var.npy',variance)

    def compute_weights(self):
        num_0 = 0
        num_1 = 0 
        num_2 = 0
        num_3 = 0
        num_4 = 0
        for idx in tqdm(range(len(self.data_dict))):
            label_path = self.data_dict[idx]["label_path"]
            radar_gt = np.load(label_path)
            if self.params['bev']:
                radar_gt = np.max(radar_gt, axis=0)
            num_0 += np.sum(radar_gt == 0)
            num_1 += np.sum(radar_gt == 1)
            num_2 += np.sum(radar_gt == 2)
            num_3 += np.sum(radar_gt == 3)
            num_4 += np.sum(radar_gt == 4)
        num = np.array([num_0, num_1, num_2, num_3, num_4])
        total_num = np.sum(num)
        class_freq = num / total_num
        class_weights = 1.0 / class_freq
        class_weights_norm = class_weights / np.sum(class_weights)
        return class_weights_norm, num
