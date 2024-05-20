from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms as T
import pandas as pd
import cv2
import torch

class Satellite_Data(Dataset):
    def __init__(self,path_csv,transform):
        self.df = pd.read_csv(path_csv)
        self.x = self.df["image_path"]
        self.y1 = self.df["digit_1"]
        self.y2 = self.df["digit_2"]
        self.y3 = self.df["digit_3"]
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx])
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        