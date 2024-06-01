import torch.nn as nn
import torch.nn.functional as F

class Scale_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cblock1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                               out_channels=16,
                                               kernel_size=3,
                                               padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2))
        self.cblock2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                               out_channels=32,
                                               kernel_size=3,
                                               padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2))
        self.expected_size = 32*57*57 
        self.fc1 = nn.Linear(self.expected_size,128)
        self.fc2 = nn.Linear(128,64)
        self.output = nn.Linear(64,30)

    def forward(self,x):
        x = self.cblock1(x)
        x = self.cblock2(x)
        x = x.view(-1, self.expected_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        x = x.view(-1,3,10)
        x = F.softmax(x,dim=2)
        return x

