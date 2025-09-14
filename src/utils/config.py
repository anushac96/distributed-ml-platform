# What: Configuration management system
# Why: Centralized config prevents hardcoded values
# When: Called at startup by all components to load settings

import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MasterConfig:
    host: str = "localhost"
    port: int = 50051
    max_workers: int = 100
    heartbeat_interval: int = 30

@dataclass
class WorkerConfig:
    master_host: str = "localhost"
    master_port: int = 50051
    gpu_enabled: bool = True
    max_batch_size: int = 32

@dataclass
class ParameterServerConfig:
    host: str = "localhost"
    port: int = 50052
    num_shards: int = 4
    compression_enabled: bool = True

class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config/local/config.yaml')
        self._load_config()
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}
        
        self.master = MasterConfig(**config_data.get('master', {}))
        self.worker = WorkerConfig(**config_data.get('worker', {}))
        self.parameter_server = ParameterServerConfig(**config_data.get('parameter_server', {}))