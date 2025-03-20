import torch
from torch import nn
from torchvision import transforms, datasets
import os
from torch.utils.data import DataLoader
from pathlib import Path
from PyTorch_Models import TinyVGG, train, plot_loss_curves
from timeit import default_timer as timer

train_transform = transforms.Compose([transforms.Resize(size=(64,64)),
                                      transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                     transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize(size=(64,64)),
                                     transforms.ToTensor()])

data_path = Path("Data")

train_dir = data_path / "Train"
test_dir = data_path / "Test"

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(dataset=train_data,
                              batch_size = BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)
NUM_EPOCHS = 30

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_0 = TinyVGG(input_shape=3,
                      hidden_units=15,
                      output_shape=len(train_data.classes)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =model_0.parameters(),
                                 lr = 0.001)
    start_time = timer()

    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer = optimizer,
                            loss_fn = loss_fn,
                            device = device,
                            epochs=NUM_EPOCHS)
    end_time = timer()
    print(f"Training time: {end_time-start_time}")
    plot_loss_curves(model_0_results)