import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from network import Scale_Model
from loader import train_loader, test_loader

def create_model():
    model = Scale_Model()
    criterion = CrossEntropyLoss()
    optim = torch.optim.Adam()
    
    return model,criterion,optim

def train_model(model,criterion,optim, train_loader,test_loader,epochs):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    model.train()
    for epoch in range(epochs):
        batch_loss = []
        batch_acc = []
        for X,y in train_loader:
            y_pred = model(X)
            loss = criterion(y_pred.view(-1,10),y.view(-1))
            batch_loss.append(loss.item())
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            break
    
    