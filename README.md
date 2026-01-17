# CCL Python Bindings

**pyccl** provides experimental Python bindings for the communication library using `ctypes`. It is designed to facilitate distributed operations on XPU/HPU devices, leveraging the low-level C APIs for high-performance communication.

> **Note**: This project is currently in an experimental state and is built upon the [oneCCL C API](https://uxlfoundation.github.io/oneCCL/v2/index.html).

## Features

- **XPU Support**: Integrates with the SYCL runtime managed by [PyTorch XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).
- **Stateless Process Group**: utility efficiently maintains multiple communication groups, leveraged by frameworks like vLLM.

## Installation

### From Source

To install the package from source, run:

```bash
python setup.py install
```

### via Pip

You can also install directly from the repository:

```bash
pip install git+https://github.com/xinyu-intel/pyhccl.git
```

## Usage

### Initialization

`pyccl` relies on a stateless process group for initializing the communication backend.

```python
from pyccl.utils import StatelessProcessGroup

def stateless_init_process_group(master_address, master_port, rank, world_size):
    """
    Initialize a stateless process group.
    
    Args:
        master_address (str): IP address of the master node.
        master_port (int): Port for communication.
        rank (int): Global rank of the current process.
        world_size (int): Total number of processes.
    """
    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size
    )
    return pg
```

### Communication Example (AllReduce)

Below is an example of performing an AllReduce operation using `PyCCLCommunicator`.

```python
import torch
from pyccl import PyCCLCommunicator

# Ensure you have initialized the process group (see above)
# pg = stateless_init_process_group(...)

# Initialize the communicator
comm = PyCCLCommunicator(pg)

# Prepare tensor on the appropriate device (e.g., XPU or HPU)
t = torch.ones((4096), device='xpu', dtype=torch.bfloat16)

# Perform AllReduce
comm.all_reduce(t)

# Synchronize the device
torch.xpu.synchronize()
```

For a complete working example, please refer to [examples/allreduce.py](examples/allreduce.py).
