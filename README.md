# DistributedML Training Platform

A scalable distributed machine learning training platform that reduces training time by 75% through custom parameter server architecture and fault-tolerant scaling.

## Features

- **Distributed Training**: Scale across multiple nodes seamlessly
- **Fault Tolerance**: Automatic recovery from node failures
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Performance Optimized**: Custom gradient compression and async updates
- **Production Ready**: Full monitoring and observability

## Quick Start
```bash
# Clone repository
git clone https://github.com/anushahadagali/distributed-ml-platform.git
cd distributed-ml-platform

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run example
python examples/mnist_distributed.py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install