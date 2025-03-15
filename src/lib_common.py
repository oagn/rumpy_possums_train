import os, re, ast, yaml
from jax import devices
from keras import distribution
from contextlib import contextmanager

# Reads a yaml file and returns a dictionary
def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Updates the configuration from environment variables
def update_config_from_env(config):
    for conf_key, value in config.items():
        if conf_key in os.environ:
            env_val = os.environ[conf_key]
            if isinstance(value, int):  # If the default is an integer
                config[conf_key] = int(env_val)
            elif isinstance(value, list) and all(isinstance(item, int) for item in value):
                config[conf_key] = [int(x) for x in env_val.split(',')]
            else:
                try:
                    config[conf_key] = ast.literal_eval(env_val)
                except (ValueError, SyntaxError):
                    config[conf_key] = env_val
    return config
  
# Maps model names to image sizes
def model_img_size_mapping(model_name):
    size_mapping = {
        r'(?i)^EN0$': 224,  # ENB0
        r'(?i)^EN2$': 260,  # ENB2
        r'(?i)^ENS$': 384,   # ENS
        r'(?i)^EN[ML]$': 480, # ENM, ENL
        r'(?i)^ENX$': 512,  # ENXL
        r'(?i)^CN[PN]$': 288, # CNP, CNN
        r'(?i)^CN[TSBL]$': 384, # CNT, CNS, CNB, CNBL
        r'(?i)^VT[TBSL]$': 384 # ViTT, ViTS, ViTB, ViTL
    }
    # Check the model name against each regex pattern and return the corresponding image size
    for pattern, size in size_mapping.items():
        if re.match(pattern, model_name[:3]):
            return size
    return 384 # Default value of 384px if no match is found

# Define NullStrategy within the module level so it can be easily accessed
class NullStrategy:
    def scope(self):
        @contextmanager
        def null_scope():
            yield
        return null_scope()

# Sets up the strategy for TensorFlow/JAX training with GPU (single or multiple) or CPU
def setup_strategy():
    gpus = devices()
    if any('cuda' in str(device).lower() for device in gpus):
        strategy = distribution.DataParallel(devices=gpus)
        print(str(len(gpus)) + ' x GPU activated' + '\n')
    else:
        strategy = NullStrategy()
        print('CPU-only training activated' + '\n')
    return strategy