import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
import sys
import socket
import pickle
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable

num_epochs = 10
learning_rate = 0.0001
batch_size = 128

def get_data(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_" + str(id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_" + str(id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples


class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Flattens input by reshaping it into a one-dimensional tensor. 
        x = torch.flatten(x, 1)
        # Apply linear transformation
        x = self.fc1(x)
        # Apply a softmax followed by a logarithm
        output = F.log_softmax(x, dim=1)
        return output


model = MCLR()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 


def train(num_epochs, model, loader, opt_method):

    model.train()

    # train the model using minibatch GD
    if opt_method:
        total_step = len(loader['train']) 
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loader['train']):
                
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = model(b_x)
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step
                optimizer.zero_grad()
                # backpropagation, compute gradients 
                loss.backward()
                # apply gradients
                optimizer.step()

                if (i+1) % 5 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # train the model using GD
    else:
        for epoch in range(num_epochs):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(X_train)   # batch x
            b_y = Variable(y_train)   # batch y
            output = model(b_x)
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()
            # backpropagation, compute gradients 
            loss.backward()
            # apply gradients
            optimizer.step()

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    return loss.data

def test(model, loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        test_counter = 0
        test_accuracy = 0
        for images, labels in loader['test']:
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            test_accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
            test_counter += 1
        print('Test accuracy of the model on the test images: {:.2f}%'.format(test_accuracy * 100/test_counter))
        return test_accuracy/test_counter


IP = '127.0.0.1'
port_server = 6000
client_id = sys.argv[1]
port_client = int(sys.argv[2])
opt_method = int(sys.argv[3])
server_address = (IP, port_server)

X_train, y_train, X_test, y_test, _, _ = get_data(client_id)
train_data = [(x, y) for x, y in zip(X_train, y_train)]
test_data = [(x, y) for x, y in zip(X_test, y_test)]

# create client_socket for sending client information and the local model to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.sendto(pickle.dumps(('handshake', client_id, list(X_train.shape)[0], port_client)), server_address)

# create server_socket for receiving the global model from the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((IP, port_client))

# dataloader for the minibatch GD
minibatch_loader = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                          shuffle=True),
}


## debug train() and test()...
# train(num_epochs, model, minibatch_loader)
# test()


# keep listening to the server 
while True:
    # receive global model from the server
    received_data = server_socket.recv(65507)
    global_model = pickle.loads(received_data)

    # upadte local model parameters 
    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        local_param.data = global_param.data.clone()

    # train the local model using the parameters from the global model
    local_loss = train(num_epochs, model, minibatch_loader, opt_method)

    # test using the local model (after training), and calculate testing accuracy
    local_accuracy = test(model, minibatch_loader)

    # print the client information
    print("I am {}".format(client_id))
    print("Receiving new global model")
    print("Training loss: {:2f}".format(local_loss))
    print("Testing accuracy: {:.2f}%".format(local_accuracy))
    print("Local training...")
    print("Sending new local model\n")

    local_model_with_id = ['model', client_id, model, local_accuracy, local_loss]

    # send local model (after training) back to the server
    client_socket.sendto(pickle.dumps(local_model_with_id), (IP, port_server))
    print('client msg sended')
