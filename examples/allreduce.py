import torch
import habana_frameworks.torch as ht
from pyhccl import PyHcclCommunicator
from pyhccl.utils import StatelessProcessGroup


def stateless_init_process_group(master_address, master_port, rank, world_size):
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pyhccl = PyHcclCommunicator(pg)
    return pyhccl

def main(node_size, nproc_per_node, local_rank, global_rank, master_ip, master_port):
    t = torch.ones((4096), device='hpu', dtype=torch.bfloat16)

    comm = stateless_init_process_group(master_ip, master_port, global_rank, nproc_per_node * node_size)

    startEv =ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    startEv.record()

    for i in range(10000):
        comm.all_reduce(t)

    endEv.record()
    endEv.synchronize()

    total_time = startEv.elapsed_time(endEv)
    avg_time = total_time / 10000
    
    print(f"[Rank {global_rank}] AllReduce benchmark completed: {total_time:.4f} ms")
    print(f"[Rank {global_rank}] Average time per iteration: {avg_time:.4f} ms")

    return t

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pyhccl test")
    parser.add_argument("--nproc-per-node",
                        type=int,
                        default=2,
                        help="Num devices per node")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=4400,
                        help="Master node port")
    args = parser.parse_args()

    node_size = args.node_size
    node_rank = args.node_rank
    nproc_per_node = args.nproc_per_node

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = args.master_port
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port


    from multiprocessing import Process

    procs = []
    for local_rank, global_rank in enumerate(
            range(node_rank * nproc_per_node, (node_rank + 1) * nproc_per_node)):
        proc = Process(target=main,
                       args=(node_size, nproc_per_node, local_rank,
                             global_rank, dp_master_ip, dp_master_port))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
