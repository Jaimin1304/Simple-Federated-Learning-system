import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
from time import sleep, time
from sys import argv
import _thread
import pickle
from random import randint


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

def send_parameters(server_model, clients_lst):
    for user in clients_lst:
        user.set_parameters(server_model)

def aggregate_parameters(server_model, clients_lst, total_train_samples):
    # Clear global model before aggregation
    for param in server_model.parameters():
        param.data = torch.zeros_like(param.data)
        
    for user in clients_lst:
        for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
    return server_model

def evaluate(clients_lst):
    total_accurancy = 0
    for user in clients_lst:
        total_accurancy += user.test()
    return total_accurancy/len(clients_lst)


# Init parameters
port_server = int(argv[1])
sub_client = int(argv[2])
IP = '127.0.0.1'
clients_lst = []
server_model = MCLR()
learning_rate = 0.001
num_glob_iters = 100 # No. of global rounds
curr_round = 0
w = randint(0, 10)

# wait for the first connection 
# and the following connections in the following 30s
first_conn_time = 0
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind((IP, port_server)) # Bind to the port
    while True:
        data = s.recv(2048)
        data = pickle.loads(data)
        # check if this is the first connection
        if len(clients_lst) == 0:
            first_conn_time = time()
        # add new client to the lst: [client_id, client_port, data_size]
        if len(clients_lst) < 5:
            clients_lst.append([int(data[0]), int(data[1]), int(data[2])])
        # stop receiving handshaking msg 30s after the first handshake
        if time() - first_conn_time >= 30 and first_conn_time != 0:
            break
    s.close()

# Runing FedAvg
loss = []
acc = []

for glob_iter in range(num_glob_iters):
    # Broadcast global model to all clients
    send_parameters(server_model,clients_lst)
    
    # Evaluate the global model across all clients
    avg_acc = evaluate(clients_lst)
    acc.append(avg_acc)
    print("Global Round:", glob_iter + 1, "Average accuracy across all clients : ", avg_acc)
    
    # Each client keeps training process to  obtain new local model from the global model 
    avgLoss = 0
    for user in clients_lst:
        avgLoss += user.train(1)
    # Above process training all clients and all client paricipate to server, how can we just select subset of user for aggregation
    loss.append(avgLoss)
    
    # TODO:  Aggregate all clients model to obtain new global model 
    aggregate_parameters(server_model, clients_lst, total_train_samples)
