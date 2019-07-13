"""
Custom datasets for Age Estaimation
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from . import utils


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        labels = []
        for line in f.readlines():
            line = line.strip().split('\t')
            labels.append(line)
    return labels


class IMDBWIKI(Dataset):
    """
    IMDB wiki dataset
    """
    def __init__(self, data_dir, filename_path, transform, db ='imdb'):
        self.data_dir = data_dir
        self.transform = transform

        filename_list = get_list_from_filenames(filename_path)
        
        self.X_train = filename_list
        self.y_train = filename_list

        self.length = len(filename_list)

    def __getitem__(self, index):

        filepath, age, gender = self.X_train[index]
        img = Image.open(os.path.join(self.data_dir, str(filepath)))
        img = img.convert('RGB')

        age = np.array(age,np.float32)
        gender = np.array(gender,np.float32)

        if self.transform is not None:
            img = self. transform(img)       
        
        return img, age, gender


    def __len__(self):
        return self.length