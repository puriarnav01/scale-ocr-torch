from src.loader import train_loader, test_loader
from src.train import create_model, train_model

if __name__ == "__main__":
    model,criterion,optim = create_model()
    train_loss, test_loss, train_acc, test_acc, train_f1, test_f1, model = train_model(model,criterion,optim,train_loader,test_loader,30)
    