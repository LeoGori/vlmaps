import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision


class ArenaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.filenames = [f_name for f_name in os.listdir(self.root_dir) if f_name.endswith('.jpg')]
        self._name = 'ArenaDataset'
        # self.transform = transform

    def get_name(self):
        return self._name

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.filenames[idx])
        # image = Image.open(img_name)
        # read image as tensor in c, h, w format
        image = torchvision.io.read_image(img_name)
        
        sample = {'image': image}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample