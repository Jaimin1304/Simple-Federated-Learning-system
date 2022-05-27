from ctypes import sizeof
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
from time import sleep, time
from sys import argv
import pickle
from random import randint
from sys import getsizeof
from socket import error as socket_error

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

def aggregate_parameters(gl_model, clients_lst, total_train_samples, sub_client):
    print('Aggregating new global model')
    # Clear global model before aggregation
    for param in gl_model.parameters():
        param.data = torch.zeros_like(param.data)

    for client in clients_lst:
        if client[3] != None:
            for server_param, client_param in zip(gl_model.parameters(), client[3].parameters()):
                server_param.data = server_param.data + client_param.data.clone() * client[2] / total_train_samples
    return gl_model

def evaluate(clients_lst):
    total_accurancy = 0
    for client in clients_lst:
        if client[4] != None:
            total_accurancy += client[4]
    return total_accurancy/len(clients_lst)


# Init parameters
port_server = int(argv[1])
sub_client = int(argv[2])
IP = '127.0.0.1'
clients_lst = []
gl_model = MCLR()
round_limit = 100 # No. of global rounds


class Handshakes_handler(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        global clients_lst
        global s_hh
        # wait for the first connection
        # and the following connections in the following 30s
        while True:
            try:
                print('Handshakes_handler')
                data_recv = s_hh.recv(2048)
                data_recv = pickle.loads(data_recv)
                # add new client to the lst: [client_id, client_addr, data_recv_size, model, accuracy, loss]
                if len(clients_lst) < 5:
                    clients_lst.append([data_recv[1], data_recv[3], int(data_recv[2]), None, None, None])
            except socket_error as e:
                break


# wait for the first connection 
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind((IP, port_server)) # Bind to the port
    # handle the first handshake outside the handshake_handler thread
    data_recv = s.recv(2048)
    print('connection!')
    data_recv = pickle.loads(data_recv)
    s.close()

# add new client to the lst: [client_id, client_addr, data_recv_size, model, accuracy, loss]
clients_lst.append([data_recv[1], data_recv[3], int(data_recv[2]), None, None, None])
# start handshake_handler thread to deal with the rest of the handshakes
# and count down 30s
s_hh = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_hh.bind((IP, port_server)) # Bind to the port
handshakes_handler = Handshakes_handler()
handshakes_handler.start()
print('start')
# stop receiving handshaking msg 30s after the first handshake
handshakes_handler.join(5)
print('join')
s_hh.close()

# broadcast the initial global model to all clients
for client in clients_lst:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_bcast:
        print((IP, client[1]))
        print(getsizeof(pickle.dumps(gl_model)))
        s_bcast.sendto(pickle.dumps(gl_model), (IP, client[1]))
        s_bcast.close()

# Runing FedAvg
loss = []
acc = []
total_train_samples = sum([i[2] for i in clients_lst])

# create server socket for listening
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((IP, port_server))

for round in range(round_limit):
    print(f'Global iteration {round+1}:')
    print(f'Total number of clients: {len(clients_lst)}')

    # receive local models from all clients
    responded_clients = 0
    while responded_clients < len(clients_lst):
        responded_clients += 1
        print("waiting client's msg")
        data_recv = s.recv(65507)
        print("client's msg received")
        data_recv = pickle.loads(data_recv)
        # check if this msg is a handshaking msg from a late-joinned client
        if data_recv[0] == 'handshake':
            if len(clients_lst) < 5:
                clients_lst.append([data_recv[1], data_recv[3], int(data_recv[2]), None, None, None])
            continue

        client_id = data_recv[1]
        for c in clients_lst:
            if client_id == c[1]:
                c[3] = data_recv[2] # model
                c[4] = data_recv[3] # local accuracy
                c[5] = data_recv[4] # local loss
                print(f'Getting local model from client {c[0]}')

    # Evaluate the global model across all clients
    avg_acc = evaluate(clients_lst)
    acc.append(avg_acc)
    print("Global Round:", round + 1, "Average accuracy across all clients : ", avg_acc)

    # Each client keeps training process to obtain new local model from the global model 
    avgLoss = 0
    for client in clients_lst:
        if client[5] != None:
            avgLoss += client[5]
    # Above process training all clients and all client paricipate to server, how can we just select subset of client for aggregation
    loss.append(avgLoss)

    # update total_train_samples
    total_train_samples = 0
    for i in clients_lst:
        if i[3] != None and i[4] != None:
            total_train_samples += i[2]

    # Aggregate all clients model to obtain new global model 
    gl_model = aggregate_parameters(gl_model, clients_lst, total_train_samples, sub_client)

    # broadcast the global model to all clients
    print('Broadcasting new global model')
    print()
    for client in clients_lst:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_bcast:
            s_bcast.sendto(pickle.dumps(gl_model), (IP, client[1]))
            s_bcast.close()
