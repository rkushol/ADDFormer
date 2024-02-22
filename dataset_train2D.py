import os
import random
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage.transform import resize

#Partial implementation taken from https://github.com/Beckschen/TransUNet/blob/main/datasets/dataset_synapse.py

def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image= random_rot_flip(image)
        elif random.random() > 0.5:
            image= random_rotate(image)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample


class dataset_train(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
  

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        lst = self.sample_list[idx].split()
        img_name = lst[0]
        img_label = lst[1]
        image_path = os.path.join(self.data_dir, img_name)
        image = nib.load(image_path)
        
        if img_label == '0':
            label = 0
        elif img_label == '1':
            label = 1
        elif img_label == '2':
            label = 2
        
        #if the image is reconstructed with FreeSurfer, image dimension 256x256x256
        image = np.array(image.get_fdata()[16:240, 16:240]).squeeze().astype(np.float32)
        image = np.clip(image, -125, 275)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        #if the image is registered with MNI template, image dimension 182x218x182
        #image = np.array(image.get_fdata()[:, :]).squeeze().astype(np.float32)
        #image = (image - np.min(image)) / (np.max(image) - np.min(image))
        #image = resize(image, (224,224), mode='constant')
        

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        #sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
