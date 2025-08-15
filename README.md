# pyhccl

Habana Communication Library Python bindings using ctypes.

Note that the project is experimental purpose only.


## Install

from source,

```
python setup.py install
```

or just,

```
pip install https://github.com/xinyu-intel/pyhccl.git
```

## Examples

* StatelessProcessGroup

Leverage from vLLM which help maintain multiple communication groups.

```python
from pyhccl.utils import StatelessProcessGroup

def stateless_init_process_group(master_address, master_port, rank, world_size):
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    return pg

```

* pyhccl.PyHcclCommunicator

```python
import torch
from pyhccl import PyHcclCommunicator

t = torch.ones((4096), device='hpu', dtype=torch.bfloat16)

pg = stateless_init_process_group(master_ip, master_port, global_rank, nproc_per_node * node_size)
pyhccl = PyHcclCommunicator(pg)
comm.all_reduce(t)
torch.hpu.synchronize()
```

This piece of code is an example of AllReduce. See the complete code [here](https://github.com/xinyu-intel/pyhccl/blob/main/examples/allreduce.py)
