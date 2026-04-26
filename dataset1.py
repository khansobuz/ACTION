import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from scipy.ndimage import rotate

class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', augment=False):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.augment = augment
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def apply_augmentations(self, npy_data):
        if self.augment:
            # Apply rotation as an augmentation
            npy_data = self.random_rotation(npy_data)
        return npy_data

    def random_rotation(self, npy_data, max_angle=30):
        angle = random.uniform(-max_angle, max_angle)
        rotated_data = []
        for frame in npy_data:
            if len(frame.shape) < 2:  # Ensure the frame is at least 2D
                raise ValueError(f"Frame should be at least 2D, got {frame.shape} instead.")
            # Rotate the 2D frame
            rotated_frame = rotate(frame, angle, axes=(0, 1), reshape=False)
            rotated_data.append(rotated_frame)
        return np.stack(rotated_data)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            concat_npy = self.apply_augmentations(concat_npy)  # Apply augmentations here
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            concat_npy = self.apply_augmentations(concat_npy)  # Apply augmentations here
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            else:
                return concat_npy, gts, frames

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO', augment=False):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.augment = augment
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def apply_augmentations(self, npy_data):
        if self.augment:
            # Apply rotation as an augmentation
            npy_data = self.random_rotation(npy_data)
        return npy_data

    def random_rotation(self, npy_data, max_angle=30):
        angle = random.uniform(-max_angle, max_angle)
        rotated_data = []
        for frame in npy_data:
            if len(frame.shape) < 2:  # Ensure the frame is at least 2D
                raise ValueError(f"Frame should be at least 2D, got {frame.shape} instead.")
            # Rotate the 2D frame
            rotated_frame = rotate(frame, angle, axes=(0, 1), reshape=False)
            rotated_data.append(rotated_frame)
        return np.stack(rotated_data)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            concat_npy = self.apply_augmentations(concat_npy)  # Apply augmentations here
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            concat_npy = self.apply_augmentations(concat_npy)  # Apply augmentations here
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            else:
                return concat_npy, gts, frames

if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0, augment=True)  # Set augment=True to apply augmentations
    print(len(loader2))
    # print(loader[1], loader2[1])
