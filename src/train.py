import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from copy import deepcopy
from src.network import Scale_Model

def create_model():
    model = Scale_Model()
    criterion = CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    
    return model,criterion,optim

def train_model(model,criterion,optim, train_loader,test_loader,epochs):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []
    
    for epoch in range(epochs):
        batch_loss = []
        batch_acc = []
        batch_f1 = []
        model.train()
        for X,y in train_loader:
            y = torch.stack(y, dim=1)
            y_pred = model(X)
            loss = criterion(y_pred.view(-1,10),y.view(-1))
            batch_loss.append(loss.item())
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            y_pred_classes = torch.argmax(y_pred,dim=2)
            correct = (y_pred_classes==y).float()
            accuracy = correct.sum()/correct.numel()
            batch_acc.append(accuracy.item())
            
            y_pred_flat = y_pred_classes.view(-1).cpu().numpy()
            y_true_flat = y.view(-1).cpu().numpy()
            f1 = f1_score(y_true_flat,y_pred_flat,average="macro")
            batch_f1.append(f1)
        
        train_loss.append(torch.tensor(batch_loss).mean().item())
        train_acc.append(torch.tensor(batch_acc).mean().item())
        train_f1.append(torch.tensor(batch_f1).mean().item())
        
    
        model.eval()
        X,y = next(iter(test_loader))
        y = torch.stack(y,dim=1)
        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred.view(-1,10),y.view(-1))
            test_loss.append(loss.item())
            
            y_pred_classes = torch.argmax(y_pred,dim=2)
            correct = (y_pred_classes==y).float()
            accuracy = correct.sum()/correct.numel()
            test_acc.append(accuracy.item())
            
            y_pred_flat = y_pred_classes.view(-1).cpu().numpy()
            y_true_flat = y.view(-1).cpu().numpy()
            f1 = f1_score(y_true_flat,y_pred_flat,average="macro")
            test_f1.append(f1)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Train F1: {train_f1[-1]:.4f} - "
              f"Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}, Test F1: {test_f1[-1]:.4f}")
        
        torch.save(model.state_dict(),"models/model_epoch_{}.pth".format(epoch))
    
    return train_loss, test_loss, train_acc, test_acc, train_f1, test_f1, model
        
        