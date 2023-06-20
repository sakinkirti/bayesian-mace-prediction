import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class TorchModel(nn.Module):
    """
    author: Sakin Kirti
    date: 11/26/2022
    
    class to define a simple neural network to classify the iris dataset
    """

    def __init__(self, num_inputs: int):
        """
        constructor to make the model
        
        params:
        inputs: the number of features being input to the model
        
        return:
        the model object
        """

        # call super
        super(TorchModel, self).__init__()

        # define the hidden layers
        self.layer1 = nn.Linear(num_inputs, 50)
        self.layer2 = nn.Linear(50, 40)
        self.layer3 = nn.Linear(40, 20)
        self.layer4 = nn.Linear(20, 5)
        self.layer5 = nn.Linear(5, 2)

    def forward(self, x):
        """
        method to define the forward propogation
        
        params:
        x: np.array - the input data
        
        return:
        np.array: the predicted values
        """

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.softmax(self.layer5(x), dim=1)
        return x

def train(data: np.array, labels: np.array, num_epochs: int):
    """
    method to train the neural network
    
    params:
    data: np.array - the training data
    labels: np.array - the true class labels
    
    return:
    torch.model - a properly trained network on the given data
    """

    # convert the input data
    data = torch.Tensor(torch.from_numpy(data)).float()
    labels = torch.Tensor(torch.from_numpy(labels)).long().ravel()

    # initialize the model and training stuff
    model = TorchModel(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_f = nn.CrossEntropyLoss()

    # for recording
    epoch_cache = []
    loss_cache = []

    # train
    for epoch in range(num_epochs):
        # forward pass
        pred = model(data)
        loss = loss_f(pred, labels)

        # cache
        epoch_cache.append(epoch)
        loss_cache.append(loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return
    print(f"Final Loss: {loss_cache[-1]}")
    return model, epoch_cache, loss_cache

def test(model, data, labels):
    """
    method to test the previously trained network
    
    params:
    model: TorchModel - the pretrained model
    data: np.array - the testing data
    labels: np.array - the true class labels

    return:
    None
    """

    # convert the input data
    data = torch.Tensor(torch.from_numpy(data)).float()
    labels = torch.Tensor(torch.from_numpy(labels)).long().ravel()

    # test
    pred = model(data)
    correct = (torch.argmax(pred, dim=1) == labels).type(torch.FloatTensor)
    return correct.mean(), pred.detach().numpy()
