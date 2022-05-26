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
        # Create a linear transformation to the incoming data_recv
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

def aggregate_parameters(server_model, clients_lst, total_train_samples):
    # Clear global model before aggregation
    for param in server_model.parameters():
        param.data_recv = torch.zeros_like(param.data_recv)
        
    for user in clients_lst:
        for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
            server_param.data_recv = server_param.data_recv + user_param.data_recv.clone() * user.train_samples / total_train_samples
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
round_limit = 100 # No. of global rounds
curr_round = 0
gl_model = randint(0, 10)

# wait for the first connection 
# and the following connections in the following 30s
first_conn_time = 0
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind((IP, port_server)) # Bind to the port
    while True:
        data_recv, addr = s.recvfrom(2048)
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

    # TODO:  Aggregate all clients model to obtain new global model 
    aggregate_parameters(server_model, clients_lst, total_train_samples)
