# File: test_poison_detection.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from qubo_detection import detect_poisoned_samples
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


# def load_data():
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST('./data', train=False, transform=transform)
#
#     # Load indices of poisoned data
#     poisoned_indices = np.load('poisoned_indices.npy')
#
#     # Create data loaders
#     train_loader_original = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     train_loader_poisoned = DataLoader(Subset(train_dataset, poisoned_indices), batch_size=64, shuffle=True)
#
#     # Detect and remove poisoned samples using QUBO
#     features = [data.numpy() for data, _ in DataLoader(train_dataset, batch_size=len(train_dataset))]
#     labels = train_dataset.targets.numpy()
#     suspected_poisoned_indices = detect_poisoned_samples(features, labels, len(train_dataset[0][0].reshape(-1)))
#
#     # Indices of cleansed dataset
#     cleansed_indices = list(set(range(len(train_dataset))) - set(suspected_poisoned_indices))
#     train_loader_cleansed = DataLoader(Subset(train_dataset, cleansed_indices), batch_size=64, shuffle=True)
#
#     return train_loader_original, train_loader_poisoned, train_loader_cleansed, DataLoader(test_dataset, batch_size=1000, shuffle=False)
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Convert entire dataset to a tensor for manipulation
    full_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    data, targets = next(iter(full_loader))
    features = data.view(len(data), -1).numpy()  # Reshape data to [num_samples, num_features]

    # Load indices of poisoned data
    poisoned_indices = np.load('poisoned_indices.npy')

    # Create data loaders
    train_loader_original = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_loader_poisoned = DataLoader( Subset(train_dataset, poisoned_indices) , batch_size=64, shuffle=True)

    # Detect and remove poisoned samples using QUBO
    suspected_poisoned_indices = detect_poisoned_samples(features, targets.numpy())

    # Indices of cleansed dataset
    cleansed_indices = list(set(range(len(train_dataset))) - set(suspected_poisoned_indices))
    train_loader_cleansed = DataLoader(Subset(train_dataset, cleansed_indices), batch_size=64, shuffle=True)

    return train_loader_original, train_loader_poisoned, train_loader_cleansed, DataLoader(test_dataset, batch_size=2000, shuffle=False)


if __name__ == "__main__":
    from poison_mnist import MNISTPoisoner
    poisoner = MNISTPoisoner( poison_rate=0.1, target_labels=None, pattern_size=(5, 5) )
    train_data, test_data, poisoned_indices = poisoner.poison_dataset()
    poisoner.save_poisoned_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader_original, train_loader_poisoned, train_loader_cleansed, test_loader = load_data()

    # Initialize and train different models
    models = [Net().to(device) for _ in range(3)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.003) for model in models]

    for model, train_loader in zip(models, [train_loader_original, train_loader_poisoned, train_loader_cleansed]):
        for epoch in range(1, 2):
            # what model is being trained
            print(f"Training model {models.index(model)}")
            train(model, device, train_loader, optimizers[models.index(model)], epoch)
            test(model, device, test_loader)




