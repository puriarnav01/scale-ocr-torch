from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms as T
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = self.transform(image)
        label_1 = torch.tensor(self.y1[idx]).long()
        label_2 = torch.tensor(self.y2[idx]).long()
        label_3 = torch.tensor(self.y3[idx]).long()
        return image, [label_1, label_2, label_3]
    

data = pd.read_csv("data/train_test.csv")
train, test = train_test_split(data, test_size=.1)
train.to_csv("data/train.csv")
test.to_csv("data/test.csv")
train_aug = T.Compose([T.ToPILImage(),T.Resize((228,228)),T.RandomHorizontalFlip(),T.RandomVerticalFlip(),
                        T.RandomRotation(90),T.ToTensor()])
test_aug = T.Compose([T.ToPILImage(),T.Resize((228,228)),T.ToTensor()])

train_data = Satellite_Data("data/train.csv",train_aug)
test_data = Satellite_Data("data/test.csv",test_aug)

train_loader = DataLoader(train_data, shuffle=True, drop_last=True, batch_size=32)
test_loader = DataLoader(test_data, shuffle=False, batch_size=len(test_data))
