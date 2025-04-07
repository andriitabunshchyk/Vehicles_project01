import torch
from torch import nn
from torchvision import transforms, datasets
import os
from torch.utils.data import DataLoader
from pathlib import Path
from PyTorch_Models import TinyVGG, train, plot_loss_curves
from timeit import default_timer as timer
import optuna
from optuna.trial import TrialState
from optuna_dashboard import run_server

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


def objective(trial):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    NUM_EPOCHS = trial.suggest_int("epochs", 10, 30)
    HIDDEN_UNITS = trial.suggest_int("hidden_units", 10, 30)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_0 = TinyVGG(input_shape=3,
                      hidden_units=HIDDEN_UNITS,
                      output_shape=len(train_data.classes)).to(device)

    LR = trial.suggest_float("lr", 1e-5, 1e-2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_name = trial.suggest_categorical("optimizer",["Adam", "SGD"])
    optimizer = getattr(torch.optim, optimizer_name)(params=model_0.parameters(),
                                 lr=LR)
    start_time = timer()

    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            device=device,
                            epochs=NUM_EPOCHS,
                            trial=trial)
    end_time = timer()
    print(f"Training time: {end_time - start_time}")
    return model_0_results['test_acc'][-1]
if __name__ == "__main__":
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction="maximize", storage=storage)
    study.optimize(objective, n_trials = 100, timeout = 2000)
    trial = study.best_trial
    print(f"Value: {trial.value} \n Parameters: {trial.params} \n {trial.system_attrs}")
    run_server(storage)
#    plot_loss_curves(???)
