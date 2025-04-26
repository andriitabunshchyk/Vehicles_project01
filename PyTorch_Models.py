import torch
from torch import nn
import matplotlib.pyplot as plt
import optuna
from typing import Dict, List


class TinyVGG(nn.Module):
    def __init__(self,
                 input_shape :int,
                 hidden_units : int,
                 output_shape: int)->None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size =3,
                      stride =1,
                      padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size =3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2)
        )
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features = hidden_units*16*16,
                                                  out_features = output_shape))

    def forward(self,x):
        x =self.conv_block_1(x)
        #print(x.shape)
        x=self.conv_block_2(x)
        #print(x.shape)
        x=self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device = device):
    #Put the model in train mode
    model.train()
    #Setup train loss and train accuracy values
    train_loss, train_acc = 0,0

    #Loop through data loader batches
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #1. Forward pass
        y_pred = model(X)

        #2.Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #3.Optimizer zero grad
        optimizer.zero_grad()

        #4. Loss backward
        loss.backward()

        #5. Optimizer step
        optimizer.step()

        #Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += ((y_pred_class ==y).sum().item()/len(y_pred))
    #Adjust metrics to get the average loss and accuracy per batch
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):
    #Put model in eval mode
    model.eval()

    #Setup test loss and accuracy values
    test_loss, test_acc = 0,0

    #Turn on inference mode
    y_preds=[]  # for Confusion Matrix
    with torch.inference_mode():
        #Loop through dataloader batches
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            #1. Forward pass
            test_pred_logits = model(X)

            #2.Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss+= loss.item()

            #Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            y_preds.append(test_pred_labels.cpu())  # For confusion Matrix
            test_acc+=((test_pred_labels == y).sum().item()/len(test_pred_logits))

    #Adjust metrics to get average loss and accuracy
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    y_pred_tensor = torch.cat(y_preds) # For Confusion Matrix y_pred_tensor
    return test_loss, test_acc , y_pred_tensor    #y_pred_tensor for Confusion Matrix

#Create a train() function to combine train_step() and test_step()
from tqdm.auto import tqdm

#1. Create a train functions

def train(trial,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int=5,
          device = device):
    #2. Create empty results dictionary
    results ={'train_loss':[],
              'train_acc':[],
              'test_loss':[],
              'test_acc':[]}
    #3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer,
                                           device=device)
        test_loss, test_acc, y_pred_tensor = test_step(model=model,    #y_pred_tensor for Confusion Matrix
                                        dataloader=test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(results)
        trial.report(results['test_acc'][epoch], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return results, y_pred_tensor    #y_pred_tensor for Confusion Matrix

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plot training curves of a results dictionary."""
    #Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss=results['test_loss']

    #Get the accuracy values of the results dictionary (training and testing)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    #Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    #Setup a plot
    plt.figure(figsize=(15,7))
    #Plot the loss
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs,test_loss, label='test_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    #Plot the accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy, label='train_acc')
    plt.plot(epochs,test_accuracy, label='test_acc')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()