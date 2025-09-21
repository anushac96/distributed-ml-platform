# grpc_coordinator.py

This module implements a gRPC server that listens for incoming requests from worker nodes and coordinates distributed machine learning tasks. It uses Protocol Buffers to define the service and message formats.
This file contains:

- A gRPC server (GRPCCoordinator) that listens for messages from worker nodes
- A service handler (MasterServicer) that defines how to respond to:
  - Workers registering
  - Heartbeats (status check-ins)
- A monitoring system to remove workers that stop responding
- This is how the Master node keeps track of active workers in your distributed training setup

## Class: MasterServicer

This class defines what happens when a worker connects to the Master.

- **RegisterWorker:** Called when a worker node registers itself with the Master. It adds the worker to the active workers list.
  - Extracts worker info (host, port, capabilities)
  - Stores it in the master's self.coordinator.workers dictionary
  - Returns a response saying “successfully registered”
- **Heartbeat:** Called periodically when a worker sends a heartbeat message to indicate it's still alive. It updates the last seen timestamp for that worker.
  - Updates the worker's last_heartbeat timestamp
  - If the worker is unknown, returns acknowledged = False

## Class: GRPCCoordinator

This is the main gRPC server manager. It:

- Starts the gRPC server
- Hosts the MasterService
- Keeps a dictionary of workers
- Monitors if any worker has stopped responding (no heartbeat in 60 seconds)

### init

Initializes the GRPCCoordinator with:

- host and port to listen on
- a dictionary to track workers
- a threading lock for safe concurrent access
- a flag to indicate if the server is running
- self.workers: where worker info is stored
- self.running: controls server loop
- self.server: the actual gRPC server

### start

Starts the gRPC server and begins listening for worker connections.

- Creates a gRPC server with a thread pool on the configured host:port
- Registers the MasterServicer to handle incoming requests
- Binds to the specified host and port
- Starts a background task to monitor worker health
- Starts the server and a monitoring thread
- Keeps the server running

### register_worker

This adds a new worker to the dictionary manually. (Used internally if needed.)

### monitor_workers

This runs in the background and periodically every 30 seconds checks if any worker has not sent a heartbeat in the last 60 seconds. If so, it removes that worker from the active list.

### stop

Stops the gRPC server gracefully.
