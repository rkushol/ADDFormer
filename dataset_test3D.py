import torch
import torchvision
import numpy as np
import nibabel as nib
import os
from torch.utils.data import Dataset
from skimage.transform import resize



class dataset3D(Dataset):
    
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
        image = np.array(image.get_fdata()[16:240, 16:240, :]).squeeze().astype(np.float32)
        image = np.clip(image, -125, 275)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        #if the image is registered with MNI template, image dimension 182x218x182
        #image = np.array(image.get_fdata()[:, :, :]).squeeze().astype(np.float32)
        #image = (image - np.min(image)) / (np.max(image) - np.min(image))
        #image = resize(image, (224,218,224), mode='constant')
        
        image = customToTensor(image)

        return [image, label]

def customToTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic)
        img = torch.unsqueeze(img,0)
        return img.float()
