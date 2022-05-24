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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
    loss = 0    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    return loss.data()

def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        test_counter = 0
        test_accuracy = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            test_accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
            test_counter += 1
        print('Test Accuracy of the model on the test images: %.2f' % (accuracy/test_counter))
        return test_accuracy

def send_parameters(server_model, users):
    for user in users:
        user.set_parameters(server_model)

def aggregate_parameters(server_model, users, total_train_samples):
    # Clear global model before aggregation
    for param in server_model.parameters():
        param.data = torch.zeros_like(param.data)
        
    for user in users:
        for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
    return server_model

def evaluate(user):
    total_accurancy = 0
    for user in users:
        total_accurancy += user.test()
    return total_accurancy/len(users)

    
IP = '127.0.0.1'
port_server = 6000
client_id = sys.argv[1]
port_client = int(sys.argv[2])
opt_method = int(sys.argv[3])

num_epochs = 10
learning_rate = 0.001

X_train, y_train, X_test, y_test, _, _ = get_data(client_id)
train_data = [(x, y) for x, y in zip(X_train, y_train)]
test_data = [(x, y) for x, y in zip(X_test, y_test)]

# Create a UDP socket for sending global model / training dataset shape to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, ord(client_id[-1]))
addr = (IP, port_server)
client_socket.sendto(pickle.dumps((client_id, list(X_train.shape))), addr)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, ord(client_id[-1]))
server_socket.bind(('', port_client))

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
}

#cnn = CNN()
#loss_func = nn.CrossEntropyLoss()   
#optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate) 
#train()
#test()





