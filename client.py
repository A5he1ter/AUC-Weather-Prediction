import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def train(net, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = net(X_train)
        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()
        optimizer.step()

def test(net):
    """Validate the model on the test set."""
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        y_pred = net(X_test)
        y_pred = torch.squeeze(y_pred)
        loss = criterion(y_pred, y_test)
        loss = round(loss.item(), 3)
        y_pred = y_pred.ge(.5).view(-1).cpu()
        correct = (y_test == y_pred).sum().float()
    accuracy = correct / len(y_test)
    accuracy = round(accuracy.item(), 3)
    #print("accuracy:", accuracy)
    #print("loss:", loss)
    return loss, accuracy


RANDOM_SEED = 42
"""Load partition CIFAR10 data."""
    #fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    #partition = fds.load_partition(partition_id)
    ## Divide data on each node: 80% train, 20% test
    #partition_train_test = partition.train_test_split(test_size=0.2)
    #pytorch_transforms = Compose(
    #    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    #)

    #def apply_transforms(batch):
    #    """Apply transforms to the partition from FederatedDataset."""
    #    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    #    return batch

    #partition_train_test = partition_train_test.with_transform(apply_transforms)
    #trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    #testloader = DataLoader(partition_train_test["test"], batch_size=32)
    #return trainloader, testloader

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1, 2],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_args().partition_id

df = pd.DataFrame()
if partition_id == 0:
    df = pd.read_csv("./data/weatherAUS_1.csv")
elif partition_id == 1:
    df = pd.read_csv("./data/weatherAUS_2.csv")
elif partition_id == 2:
    df = pd.read_csv("./data/weatherAUS_0.csv.csv")
cols = ['Rainfall', 'Humidity3pm', 'Pressure9am', 'RainToday', 'RainTomorrow']
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)
df = df[cols]
df = df.dropna(how='any')
X = df[['Rainfall', 'Humidity3pm', 'RainToday', 'Pressure9am']]
y = df[['RainTomorrow']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

# Load model and data (simple CNN, CIFAR-10)
#net = Net().to(DEVICE)
net = Net(X_train.shape[1])
#trainloader, testloader = load_data(partition_id=partition_id)



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, epochs=10)
        return self.get_parameters(config={}), len(y_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net)
        return loss, len(y_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)
