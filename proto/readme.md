# What Is This?

This .proto file defines:

- A service called MasterService — the central controller
- Messages (data formats) used for communication between workers and the master
- This is used in a distributed ML system where:
- A master assigns training jobs
- Workers register themselves, send heartbeats, request jobs, and report job completions

## master.proto

### Service Definition

This defines 4 remote procedure calls (RPCs) — like "functions" the worker can call on the master:
Function What it does

- **RegisterWorker:** Worker says “Hi, I’m ready to work!”
- **SendHeartbeat:** Worker checks in regularly to show it’s alive and healthy
- **RequestJob:** Worker asks, “Do you have any job for me?”
- **CompleteJob:** Worker says “I finished the job” and reports results

### Message Definitions

#### RegisterWorkerRequest & RegisterWorkerResponse

**Worker says:**

- "I’m worker ABC"
- "Here are my capabilities" (e.g., has GPU, supports certain models)
- "I live at this host and port"

**Master replies:**

- Whether registration was successful
- Optional message ("Registered successfully", "Duplicate ID", etc.)
- Worker ID assigned (could be a modified version of the submitted one)

#### WorkerCapabilities

- has_gpu (bool): Does the worker have a GPU?
- supported_models (repeated string): List of model types the worker can handle (e.g., "resnet", "bert")
- max_concurrent_jobs (int32): Maximum number of jobs the worker can handle at once

#### HeartbeatRequest & HeartbeatResponse

**Worker says:**

- "I’m worker ABC"
- "Here’s my current status" (e.g., CPU load, memory usage)
- "I’m still alive!"

**Master replies:**

- Whether heartbeat was received successfully
- Optional message ("Heartbeat received", "Worker not recognized", etc.)

#### WorkerStatus

- cpu_load (float): Current CPU load percentage (0.0 to 100.0)
- memory_usage (float): Current memory usage percentage (0.0 to 100.0)
- current_jobs (int32): Number of jobs currently being processed

#### JobRequest & JobResponse

**Worker says:**

- "I’m worker ABC"
- "Do you have any job for me?"

**Master replies:**

- Whether a job is assigned
- Job ID if assigned
- Job parameters (e.g., dataset path, model type, hyperparameters)
- Optional message ("Job assigned", "No jobs available", etc.)

#### JobParameters

- dataset_path (string): Path to the training dataset
- model_type (string): Type of model to train (e.g., "resnet", "bert")
- hyperparameters (map<string, string>): Key-value pairs of hyperparameters (e.g., "learning_rate": "0.001", "batch_size": "32")

#### TrainingJob

- job_id (string): Unique identifier for the job
- parameters (JobParameters): Parameters for the training job

#### TrainingConfig

- default_learning_rate (float): Default learning rate if not specified
- default_batch_size (int32): Default batch size if not specified
- max_epochs (int32): Maximum number of training epochs
- Optimizer to use (e.g., "Adam", "SGD")

#### JobCompleteRequest & JobCompleteResponse

**Worker says:**

- "I’m worker ABC"
- "I finished job XYZ"
- "Here are the results" (e.g., accuracy, loss)

**Master replies:**

- Whether job completion was recorded successfully
- Optional message ("Job completion recorded", "Job ID not recognized", etc.)

#### JobResults

- accuracy (float): Final accuracy achieved
- loss (float): Final loss value
- metrics (map<string, float>): Additional metrics (e.g., "precision": 0.95, "recall": 0.90)
- Whether it was successful
- Any errors that happened

## worker.proto

### Service Name

This defines 3 actions the master can request from a worker:

Function

- **StartTraining:** Tells the worker to start training a model
- **StopTraining:** Tells the worker to stop a running job
- **GetStatus:** Asks the worker: “How’s it going?”

### Message Definitions

#### StartTrainingRequest & StartTrainingResponse

**Master says:**

- "Please start training job XYZ"

**Worker replies:**

- Whether the job was started successfully
- Optional message ("Job started", "Worker busy", etc.)

#### StopTrainingRequest & StopTrainingResponse

**Master says:**

- "Please stop job XYZ"

**Worker replies:**

- Whether the job was stopped successfully
- Optional message ("Job stopped", "Job not found", etc.)

#### GetStatusRequest & GetStatusResponse

**Master says:**

- "How are you doing, worker ABC?"

**Worker replies:**

- Current status (e.g., CPU load, memory usage, current jobs)
- Optional message ("Status retrieved", "Worker not recognized", etc.)

#### WorkerStatus

- cpu_load (float): Current CPU load percentage (0.0 to 100.0)
- memory_usage (float): Current memory usage percentage (0.0 to 100.0)
- current_jobs (int32): Number of jobs currently being processed

<!-- ### Why Use This?

- **Standardized Communication:** Everyone speaks the same language (defined by .proto)
- **Efficiency:** Binary format is compact and fast
- **Scalability:** Easy to add more workers or extend functionality

### How to Use This?

1. Install Protocol Buffers and gRPC libraries
2. Compile the .proto file to generate code in your preferred language (Python, Go, etc.)
3. Implement the server (master) and client (worker) logic using the generated code

### Example Commands to Compile and Run

```bash
#Install basic dependencies first
python -m pip install --upgrade pip
python -m pip install grpcio grpcio-tools protobuf
pip install torch torchvision pyyaml
#Create the config directory and file
mkdir -p config/local
#Compile the proto file
python -m grpc_tools.protoc --proto_path=proto --python_out=src/generated --grpc_python_out=src/generated proto/master.proto
python scripts/generate_proto.py
#Run the basic example
pip install -e .
python examples/basic_example.py
```

### Next Steps -->

## parameter_server.proto

- This file defines a ParameterServerService for managing model parameters in a distributed ML setup.
- This defines the communication between workers and a Parameter Server in a Distributed Machine Learning system.

## What Is a Parameter Server?

In distributed ML, especially with large models:

- Each worker trains a part of the model using a batch of data
- But the model parameters (weights) are stored and updated centrally on a Parameter Server

The idea is:

- Workers send gradients (updates) → 📨 Push
- They receive updated parameters (weights) → 📤 Pull

### Service Definition

This defines 3 remote procedure calls (RPCs) — like "functions" the worker can call on the parameter server:
Function

- **PushGradients:** Worker sends updated model parameters to the parameter server
- **PullParameters:** Worker requests the latest model parameters from the parameter server
- **GetParameterVersion:** Worker asks for the current version of the model parameters
- **RegisterWorker:** Worker says "Hey, I'm online" to the parameter server

### Message Definitions

#### PushGradientsRequest & PushGradientsResponse

**Worker says:**

- "Here are my updated gradients for job XYZ"
- worker_id: Who is sending this?
- layer_name: Which part of the model (e.g., "layer1", "output")?
- gradient_data: The actual gradient (binary blob – efficient format)
- version: The version of parameters this was based on

**Parameter Server replies:**

- Whether the gradients were received successfully. Whether it successfully applied the gradients
- Optional message ("Gradients received", "Worker not recognized", etc.)
- What the new version of the model is

#### PullParametersRequest & PullParametersResponse

**Worker says:**

- "I need the latest parameters for job XYZ"
- worker_id: Who is requesting?
- layer_name: Which part of the model?
- version: The version of parameters the worker currently has (to avoid redundant data transfer)
  **Parameter Server replies:**
- Whether the parameters were sent successfully
- Optional message ("Parameters sent", "Worker not recognized", etc.)
- parameter_data: The actual model parameters (binary blob)
- version: The version of the parameters being sent

#### GetParameterVersionRequest & GetParameterVersionResponse

**Worker says:**

- "What’s the current version of parameters for job XYZ?"
- worker_id: Who is asking?
- layer_name: Which part of the model?
  **Parameter Server replies:**
- The current version of the parameters
- Optional message ("Version retrieved", "Worker not recognized", etc.)

#### RegisterWorkerRequest & RegisterWorkerResponse

**Worker says:**

- "I’m worker ABC, ready to push/pull parameters"
- worker_id: Unique ID of the worker
  **Parameter Server replies:**
- Whether registration was successful
- Optional message ("Registered successfully", "Duplicate ID", etc.)
- Worker ID assigned (could be a modified version of the submitted one)

#### Imagine You're Teaching a Class

Let’s say you're a teacher (the Parameter Server), and you have a class of students (the Workers). You're all working together on building a model, like solving a math problem or training an AI.

#### What is a Model?

In Machine Learning, a model is like a smart calculator that learns from data.
Example:

- You show it a lot of pictures of cats and dogs, and it learns to tell them apart.
- The model “learns” by adjusting its internal settings called weights.

#### What are Weights?

Think of weights as the "knobs" or "settings" inside the model.

- They decide how the model reacts to different inputs.
- During training, we keep tweaking these weights until the model gives the right answers.

**Analogy:** If your brain was a model, then weights are like how strongly you believe certain things — and you change them as you learn.

#### What are Gradients?

Gradients are the suggestions for how to change the weights to make the model better.

They're like saying:

- “This weight is too low, increase it a bit,”
  or
- “That one is too high, reduce it.”

**Analogy:** A gradient is like feedback after a test — it tells you what you got wrong and how to improve.

#### Why Do We Need a Parameter Server?

Now, imagine you have many students (workers) working on the same assignment in parallel.

- Each student trains the model on their own small dataset.
- But we want to combine all their learnings into one master model.

This is where the Parameter Server comes in.

#### The Parameter Server’s Job:

The Parameter Server acts like the central brain or teacher.

It does 3 things:

**Push:** Workers send their gradients (learning suggestions) to the parameter server.
This is like students sending their notes to the teacher.

**Pull:** Workers request the latest weights from the parameter server.
Like students asking: “What’s the latest answer key so I can keep studying with the most updated info?”

**Register:** When a new worker starts, it tells the parameter server, “Hey, I’m ready to learn and contribute!”

#### Workflow in Simple Steps:

- Worker starts with some weights (model settings).

- It trains on its own data and calculates gradients (how to improve).

- It pushes those gradients to the parameter server.

- The parameter server updates the global weights using those gradients.

- Workers then pull the latest weights and continue training.

This loop continues until the model is well-trained.

#### Simple Real-Life Analogy

- Team Project in School

- Everyone writes a part of the project

- You send your updates (gradients) to the team leader (parameter server)

- The team leader updates the master copy (weights)

- You ask the leader for the newest version to keep working (pull parameters)

You do more work and repeat the cycle
