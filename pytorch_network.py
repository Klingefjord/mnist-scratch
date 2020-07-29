import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from mnist_loader import load_data_wrapper

class PytorchNetwork(nn.Module):
    def __init__(self):
        super(PytorchNetwork, self).__init__()
        self.linear1 = nn.Linear(784, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 30)
        self.linear4 = nn.Linear(30, 10)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return x

def train(number_of_epochs=30, batch_size=64, learning_rate=.1):
    training_data, _, test_data = load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    random.shuffle(training_data)

    net = PytorchNetwork()
    criterion = nn.MSELoss()
    accuracy = 0.0

    for e in range(number_of_epochs):
        batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
        for batch in batches: 
            x = torch.stack([torch.flatten(torch.tensor(training_values)).float() for training_values, _ in batch])
            y_hat = torch.stack([torch.flatten(torch.tensor(labels)).float() for _, labels in batch])
            y = net(x)
            loss = criterion(y, y_hat)
            accuracy = 1 - loss.item()

            net.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in net.parameters():
                    param -= (learning_rate * param.grad) / batch_size

        print(f"Epoch {e} finished, acc: {accuracy}")
    
if __name__ == "__main__": 
    train()