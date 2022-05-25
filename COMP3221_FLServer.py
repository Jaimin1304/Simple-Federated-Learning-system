import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

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

# Init parameters 
num_user = 5
users = []
server_model = MCLR()
batch_size = 20
learning_rate = 0.01
num_glob_iters = 100 # No. of global rounds

# TODO:  Create a federate learning network with 5 clients and append it to users list.
total_train_samples = 0
for i in range(1,num_user+1):
    user = UserAVG(i, server_model, learning_rate, batch_size)
    users.append(user)
    total_train_samples += user.train_samples
    send_parameters(server_model, users)

# Runing FedAvg
loss = []
acc = []

for glob_iter in range(num_glob_iters):
    
    
    # TODO: Broadcast global model to all clients
    send_parameters(server_model,users)
    
    # Evaluate the global model across all clients
    avg_acc = evaluate(users)
    acc.append(avg_acc)
    print("Global Round:", glob_iter + 1, "Average accuracy across all clients : ", avg_acc)
    
    # Each client keeps training process to  obtain new local model from the global model 
    avgLoss = 0
    for user in users:
        avgLoss += user.train(1)
    # Above process training all clients and all client paricipate to server, how can we just select subset of user for aggregation
    loss.append(avgLoss)
    
    # TODO:  Aggregate all clients model to obtain new global model 
    aggregate_parameters(server_model, users, total_train_samples)
