# COMP3221 ASM3

## Run the server
In the format of `python COMP3221_FLServer.py <Port-Server> <Sub-client>`

### Disable client sampling with flag 0 <M=K>:
`python3 COMP3221_FLServer.py 6000 0` 

### Disable client sampling with flag 1 <M<K>>:
`python3 COMP3221_FLServer.py 6000 1` 

## Run the clients
In the format of `python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>`
### Example of running single client with the optimization method of (batch) gradient descent
`python3 COMP3221_FLClient.py client1 6001 0` 
### Example of running single client with the optimization method of minibatch gradient descent
`python3 COMP3221_FLClient.py client1 6001 1` 

### You can run each client in a sequence:
`python3 COMP3221_FLClient.py client1 6001 0` 

`python3 COMP3221_FLClient.py client2 6002 0`

`python3 COMP3221_FLClient.py client3 6003 0`

`python3 COMP3221_FLClient.py client4 6004 1`

`python3 COMP3221_FLClient.py client5 6005 1`

### You can also run all clients using make_clients.sh
`make_clients.sh`
Then use `screen -r <Client-id>` to access the virtual terminal of each client

