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

# Define a wrapper for JAX sharding to make it compatible with the scope() API
class JaxShardingWrapper:
    def __init__(self, sharding):
        self.sharding = sharding
    
    def scope(self):
        @contextmanager
        def jax_scope():
            yield
        return jax_scope()

# Sets up the strategy for TensorFlow/JAX training with GPU (single or multiple) or CPU
def setup_strategy():
    from jax import devices
    gpus = devices()
    if any('cuda' in str(device).lower() or 'gpu' in str(device).lower() for device in gpus):
        try:
            # Try the new approach first
            strategy = distribution.DataParallel(devices=gpus)
            print(str(len(gpus)) + ' x GPU activated' + '\n')
        except (AttributeError, TypeError) as e:
            # Fall back to a simpler approach if the DataParallel initialization fails
            print(f"Warning: DataParallel initialization failed with error: {e}")
            print("Falling back to single-device strategy")
            
            # For JAX backend
            if os.environ.get("KERAS_BACKEND", "").lower() == "jax":
                from jax.sharding import PositionalSharding
                import jax
                
                # Create a simple positional sharding for JAX
                devices = jax.devices()
                if len(devices) > 0:
                    # Wrap the PositionalSharding in our compatible wrapper
                    sharding = PositionalSharding(devices)
                    strategy = JaxShardingWrapper(sharding)
                    print(f"Using JAX PositionalSharding with {len(devices)} devices\n")
                else:
                    strategy = NullStrategy()
                    print("No JAX devices found, using CPU-only training\n")
            else:
                # For TensorFlow backend, use MirroredStrategy if multiple GPUs are available
                import tensorflow as tf
                if len(gpus) > 1:
                    strategy = tf.distribute.MirroredStrategy(devices=gpus)
                    print(f"Using TensorFlow MirroredStrategy with {len(gpus)} GPUs\n")
                else:
                    strategy = tf.distribute.OneDeviceStrategy(device=gpus[0] if gpus else "/cpu:0")
                    print(f"Using TensorFlow OneDeviceStrategy on {gpus[0] if gpus else 'CPU'}\n")
    else:
        strategy = NullStrategy()
        print('CPU-only training activated' + '\n')
    return strategy