import os
import torch
import numpy as np
from torch.utils.data import (
    Dataset,
)
import random
import torchvision.transforms.functional as TF

class IRdropDataset(Dataset):
    def __init__(self, root_dir,transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.feature_files = os.listdir(self.root_dir+'/feature')
        self.label_files = os.listdir(self.root_dir+'/label')
    def __len__(self):
        return len(self.feature_files)
    def __getitem__(self, index):
        features_name = self.feature_files[index]
        feature = self.feature_data(features_name)
        label_name = self.label_files[index]
        label = self.label_data(label_name)
        if self.transform == True:
            if random.random() > 0.5:
                feature = TF.hflip(feature)
                label = TF.hflip(label)
            if random.random() > 0.5:
                feature = TF.vflip(feature)
                label = TF.vflip(label)
            # if random.random() > 0.5:
            #     feature = TF.rotate(feature, 90)
            #     label = TF.rotate(label, 90)
            # if random.random() > 0.5:
            #     feature = TF.rotate(feature, 270)
            #     label = TF.rotate(label, 270)
        return feature,label

    def feature_data(self,f_name):
        f = torch.transpose(torch.as_tensor(np.load(f"{self.root_dir}/feature/{f_name}")), 0, 2)
        f = torch.transpose(f, 1, 2)
        return f.type(torch.float32)

    def label_data(self,l_name):
        l = torch.as_tensor(np.load(f"{self.root_dir}/label/{l_name}")).squeeze().unsqueeze(1)
        l = torch.transpose(l , 0, 1)
        return l.type(torch.float32)
