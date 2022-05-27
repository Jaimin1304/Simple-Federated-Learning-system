# Use screen to run the following processes,
# where we can then choose to open any number of virtual terminals by using 'screen -r <node id>'
#screen -dmS "server" python3 COMP3221_FLServer.py 6000 0
screen -dmS "client1" python3 COMP3221_FLClient.py client1 6001 0
screen -dmS "client2" python3 COMP3221_FLClient.py client2 6002 0
screen -dmS "client3" python3 COMP3221_FLClient.py client3 6003 0
screen -dmS "client4" python3 COMP3221_FLClient.py client4 6004 0
screen -dmS "client5" python3 COMP3221_FLClient.py client5 6005 0

# Run the following to monitor a node
# screen -r <node id>

# Run the following to kill all python processes
# kill -9 $(ps -A | grep python | awk '{print $1}')