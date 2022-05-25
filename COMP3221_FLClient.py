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
learning_rate = 0.001


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

model = CNN()
loss_func = nn.CrossEntropyLoss()   
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

def train(num_epochs, model, loader, opt_method):
    
    model.train()
        
    # Train the model
    total_step = len(loader['train'])
    
    if opt_method:    
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loader['train']):
                
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = model(b_x)[0]               
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()    
                # apply gradients             
                optimizer.step()                
                
                if (i+1) % 10 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    else:
        for epoch in range(num_epochs):              
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(X_train)   # batch x
            b_y = Variable(y_train)   # batch y
            output = model(b_x)[0]               
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
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            test_accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
            test_counter += 1
        print('Test accuracy of the model on the test images: %.2f' % (test_accuracy/test_counter))
        return test_accuracy

def set_parameters(model):
    for old_param, new_param in zip(model.parameters(), model.parameters()):
        old_param.data = new_param.data.clone()

IP = '127.0.0.1'
port_server = 6000
client_id = sys.argv[1]
port_client = int(sys.argv[2])
opt_method = int(sys.argv[3])

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


# train(num_epochs, model, loaders)
# test()


while True:
    # receive global model from the server
    received_data, adr = server_socket.recvfrom(65507)
    global_model = pickle.loads(received_data)
    # training loss
    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        local_param.data = global_param.data.clone()

    local_loss = train(num_epochs, model, loaders, opt_method)

    # make the prediction using global model, and calculate accuracy
    local_accuracy = test(model, loaders)


    # output information
    print("I am {}".format(client_id))
    print("Receiving new global model")
    print("Training loss: {:2f}".format(local_loss))
    print("Testing accuracy: {:.2f}%".format(local_accuracy))
    print("Local training...")
    print("Sending new local model")
    print()

    local_model_with_id = [client_id, model]

    # sending new local model to the server
    addr = (IP, port_server)
    client_socket.sendto(pickle.dumps(local_model_with_id), addr)





