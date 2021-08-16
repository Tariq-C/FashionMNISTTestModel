import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

# define Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.conv2d_maxpool_stack = nn.Sequential(
            nn.Conv2d(1, 6, 4),
            nn.ReLU(),
            nn.Conv2d(6, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, 4),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv2d_maxpool_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

print ("Loading Data")

model = NeuralNetwork()
model.load_state_dict(torch.load("model_99.95.pth"))


test_dataloader = DataLoader(test_data, batch_size=batch_size)

print("Done Loading")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("Testing Data")


# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model.eval()
size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
test_loss, correct = 0, 0
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        predicted = classes[pred[0].argmax(0)]
        actual = classes[y[0]]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")