import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
from time import sleep, time
from sys import argv
import _thread
import pickle
from random import randint


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

def aggregate_parameters(gl_model, clients_lst, total_train_samples):
    # Clear global model before aggregation
    for param in gl_model.parameters():
        param.data_recv = torch.zeros_like(param.data_recv)
        
    for user in clients_lst:
        for server_param, user_param in zip(gl_model.parameters(), user.model.parameters()):
            server_param.data_recv = server_param.data_recv + user_param.data_recv.clone() * user.train_samples / total_train_samples
    return gl_model

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
gl_model = CNN()
round_limit = 100 # No. of global rounds
curr_round = 0

# wait for the first connection 
# and the following connections in the following 30s
first_conn_time = 0
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind((IP, port_server)) # Bind to the port
    while True:
        data_recv, addr = s.recvfrom(4096)
        data_recv = pickle.loads(data_recv)
        # check if this is the first connection
        if len(clients_lst) == 0:
            first_conn_time = time()
        # add new client to the lst: [client_id, client_addr, data_recv_size]
        if len(clients_lst) < 5:
            clients_lst.append([int(data_recv[0]), addr, int(data_recv[1])])
        # stop receiving handshaking msg 30s after the first handshake
        if time() - first_conn_time >= 30 and first_conn_time > 0:
            break
    s.close()

# Runing FedAvg
loss = []
acc = []

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((IP, port_server))
for round in range(round_limit):
    # broadcast the global model to all clients
    for client in clients_lst:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_bcast:
            s_bcast.sendto(gl_model.encode(), client[1])
            s_bcast.close()

    # receive local models from all clients
    responded_clients = []
    while len(responded_clients) < len(clients_lst):
        responded_clients += 1
        data_recv, addr = s.recvfrom(65507)
        data_recv = pickle.loads(data_recv)
        # check if this msg is a handshaking msg from a late-joinned client

    # Evaluate the global model across all clients
    avg_acc = evaluate(clients_lst)
    acc.append(avg_acc)
    print("Global Round:", round + 1, "Average accuracy across all clients : ", avg_acc)

    # Each client keeps training process to obtain new local model from the global model 
    avgLoss = 0
    for user in clients_lst:
        avgLoss += user.train(1)
    # Above process training all clients and all client paricipate to server, how can we just select subset of user for aggregation
    loss.append(avgLoss)

    # Aggregate all clients model to obtain new global model 
    aggregate_parameters(gl_model, clients_lst, total_train_samples)
