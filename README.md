# COMP3221 ASM3
This is a simple Federated Learning (FL) system for COMP3221 Assignment 3, developed by Group3.

## Run the server
In a separate terminal, type: `python COMP3221_FLServer.py <Port-Server> <Sub-client>`

### Disable clients subsampling with flag 0 <M=K>:
e.g. `python3 COMP3221_FLServer.py 6000 0`

### Enable clients subsampling with flag 1 <M<K>>:
e.g. `python3 COMP3221_FLServer.py 6000 1`

## Run the clients
In the format of `python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>`
### Example of running single client with the optimization method of (batch) gradient descent
`python3 COMP3221_FLClient.py client1 6001 0`
### Example of running single client with the optimization method of minibatch gradient descent
`python3 COMP3221_FLClient.py client1 6001 1`

### You can run each client in a separate terminal:
The following is an example where client1 uses GD, client2 uses minibatch GD, client3 uses GD, client4 uses minibatch GD, client5 uses GD.
`python3 COMP3221_FLClient.py client1 6001 0`

`python3 COMP3221_FLClient.py client2 6002 1`

`python3 COMP3221_FLClient.py client3 6003 0`

`python3 COMP3221_FLClient.py client4 6004 1`

`python3 COMP3221_FLClient.py client5 6005 0`

### You can also run all clients at once using make_clients.sh
To do this, type`bash make_clients.sh`,
then use `screen -r <Client-id>` to access the virtual terminal of each client.
