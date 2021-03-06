from ctypes import sizeof
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
from sys import argv
import pickle
from sys import getsizeof
from socket import error as socket_error
import random
import matplotlib
import matplotlib.pyplot as plt

class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)
        nn.init.xavier_uniform_(self.fc1.weight)

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

    clean_clients_lst = [i for i in clients_lst if i[3] != None]

    if not sub_client or len(clean_clients_lst) < 2:
        for client in clean_clients_lst:
            if client[3] != None:
                for server_param, client_param in zip(gl_model.parameters(), client[3].parameters()):
                    server_param.data = server_param.data + client_param.data.clone() * client[2] / total_train_samples
    else:
        if len(clean_clients_lst) >= 2:
            client_1 = random.choice(clean_clients_lst)
            other_clients = clean_clients_lst.copy()
            other_clients.remove(client_1)
            client_2 = random.choice(other_clients)
            if client_1[3] != None and client_2[3] != None:
                for server_param, client_param in zip(gl_model.parameters(), client_1[3].parameters()):
                    server_param.data = server_param.data + client_param.data.clone() * client_1[2] / total_train_samples
                for server_param, client_param in zip(gl_model.parameters(), client_2[3].parameters()):
                    server_param.data = server_param.data + client_param.data.clone() * client_2[2] / total_train_samples
            else:
                print('wrong client chosen!')
    return gl_model

def evaluate(clients_lst):
    clean_client_lst = [i for i in clients_lst if i[4] != None]
    total_accurancy = sum([i[4] for i in clean_client_lst])
    return total_accurancy/len(clean_client_lst)


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
        global clients_lst, s_hh
        # wait for the first connection and the following connections in the following 30s
        while True:
            try:
                #print('Handshakes_handler')
                data_recv = s_hh.recv(2048)
                data_recv = pickle.loads(data_recv)
                if len(clients_lst) < 5:
                    # add new client: [client_id, client_addr, data_recv_size, model, accuracy, loss, alive_flag]
                    clients_lst.append([data_recv[1], data_recv[3], int(data_recv[2]), None, None, None, False])
                    print(f'new connection from {data_recv[1]}')
            except socket_error as e:
                break


class Datapackages_handler(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        global clients_lst, s
        data_recv = s.recv(65507)
        #print("client's msg received")
        data_recv = pickle.loads(data_recv)
        # check if this msg is a handshaking msg from a late-joinned client
        if data_recv[0] == 'handshake':
            if len(clients_lst) < 5:
                # add new client: [client_id, client_addr, data_recv_size, model, accuracy, loss, alive_flag]
                clients_lst.append([data_recv[1], data_recv[3], int(data_recv[2]), None, None, None, False])
                print(f'new connection from {data_recv[1]}')
            return

        client_id = data_recv[1]
        for c in clients_lst:
            if client_id == c[0]:
                c[3] = data_recv[2] # model
                c[4] = data_recv[3] # local accuracy
                c[5] = data_recv[4] # local loss
                c[6] = True # set the alive_flag to ture, indicating this client is alive
                print(f'Getting local model from client {c[0]}')


# wait for the first connection 
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind((IP, port_server)) # Bind to the port
    # handle the first handshake outside the handshake_handler thread
    data_recv = s.recv(2048)
    #print('connection!')
    data_recv = pickle.loads(data_recv)
    s.close()

# add new client: [client_id, client_addr, data_recv_size, model, accuracy, loss, alive_flag]
clients_lst.append([data_recv[1], data_recv[3], int(data_recv[2]), None, None, None, False])
print(f'new connection from {data_recv[1]}')
# start handshake_handler thread to deal with the rest of the handshakes
# and count down 30s
s_hh = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_hh.bind((IP, port_server)) # Bind to the port
handshakes_handler = Handshakes_handler()
handshakes_handler.start()
#print('start')
# stop receiving handshaking msg 30s after the first handshake
handshakes_handler.join(10)
#print('join')
s_hh.close()

# broadcast the initial global model to all clients
for client in clients_lst:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_bcast:
        #print((IP, client[1]))
        #print(getsizeof(pickle.dumps(gl_model)))
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
    # reset alive_flag for every clinet records
    for i in clients_lst:
        i[6] = False

    print(f'Global iteration {round+1}:')
    print(f'Total number of clients: {len(clients_lst)}')
    # receive local models from all clients
    responded_clients = 0
    while responded_clients < len(clients_lst):
        print(responded_clients)
        responded_clients += 1
        #print("waiting client's msg")
        datapackages_handler = Datapackages_handler()
        datapackages_handler.start()
        datapackages_handler.join(1)

    # delete clients from the clients_lst who failed to send data to the server
    for i in clients_lst[::-1]:
        if not i[6] and i[5] != None and i[4] != None and i[3] != None:
            print(f'{i[0]} disconnected!')
            clients_lst.remove(i)
    #print(clients_lst)

    # Evaluate the global model across all clients
    avg_acc = evaluate(clients_lst)
    acc.append(avg_acc)
    print("Global Round:", round + 1, "\nAverage accuracy across all clients : {:.2f}%".format(avg_acc * 100))

    avgLoss = sum([i[5] for i in clients_lst if i[5] != None])

    loss.append(avgLoss)

    # update total_train_samples
    total_train_samples = sum([i[2] for i in clients_lst if i[3] != None and i[4] != None])

    # Aggregate all clients model to obtain new global model 
    gl_model = aggregate_parameters(gl_model, clients_lst, total_train_samples, sub_client)

    # broadcast the global model to all clients
    print('Broadcasting new global model')
    #print(clients_lst)
    print()
    for client in clients_lst:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_bcast:
            s_bcast.sendto(pickle.dumps(gl_model), (IP, client[1]))
            s_bcast.close()

with open('global_accuracy.txt', 'w') as f:
    f.writelines([str(i)+'\n' for i in acc])

with open('global_loss.txt', 'w') as f:
    f.writelines([str(i.item())+'\n' for i in loss])

print('Training process complete! final accuracy: {:0.4f}, final loss: {:0.4f}\n'.format(acc[-1], loss[-1]))

# send a terminate signal to all clients
for client in clients_lst:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_bcast:
        s_bcast.sendto(pickle.dumps('mission complete'), (IP, client[1]))
        s_bcast.close()
