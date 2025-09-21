# grpc_worker.py

- This file contains the implementation of a gRPC worker that communicates with a master server to receive and execute tasks. The worker registers itself with the master, listens for incoming tasks, executes them
- This file defines a class called GRPCWorker, which:
  - Connects to the Master node over gRPC
  - Tells the Master: "I'm a worker — here’s what I can do"
  - Starts sending heartbeat messages every 30 seconds
  - Stays alive and ready for future tasks (like training jobs)
- It's like a worker joining a team and checking in regularly with the manager (Master).

## init

- The `__init__` method initializes the GRPCWorker instance with the master address, worker address, and capabilities.
- This sets up the worker with:
  - worker_id: a unique name like worker-1
  - config: configuration like which host/port to connect to
- It also prepares:
  - A gRPC channel (for communication)
  - A stub (a tool to call Master’s functions)

## start

- The `start` method:
  - Connects to the Master server
  - Registers the worker with its capabilities
  - Starts a heartbeat thread to send regular updates to the Master in background
  - Enters a loop to listen for incoming tasks from the Master. Keeps running (doesn't exit)

## stop

If needed, this method will:

- Stop the heartbeat loop
- Close the gRPC connection
