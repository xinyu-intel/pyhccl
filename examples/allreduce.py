import torch
import habana_frameworks.torch as ht
from pyhccl import PyHcclCommunicator
from pyhccl.utils import StatelessProcessGroup
import csv
import os

def stateless_init_process_group(master_address, master_port, rank, world_size):
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pyhccl = PyHcclCommunicator(pg)
    return pyhccl

def main(node_size, nproc_per_node, local_rank, global_rank, master_ip, master_port, output_dir):
    
    results = []
    t = torch.ones(1, device='hpu', dtype=torch.bfloat16)
    comm = stateless_init_process_group(master_ip, master_port, global_rank, nproc_per_node * node_size)
    # Iterate over powers of 2 for tensor sizes
    for power in range(10, 25):  # From 2^10 to 2^24 elements
        size = 2 ** power
        
        t = torch.ones(size, device='hpu', dtype=torch.bfloat16)
        
        # Warm up
        for _ in range(10):
            comm.all_reduce(t)
        
        startEv = ht.hpu.Event(enable_timing=True)
        endEv = ht.hpu.Event(enable_timing=True)
        
        iterations = 20000
        total_time = 0 
        for i in range(iterations):
            startEv.record()
            comm.all_reduce(t)
        
            endEv.record()
            endEv.synchronize()
            iter_time = startEv.elapsed_time(endEv)
            total_time += iter_time
        avg_time = total_time / iterations
        
        # Calculate data size in bytes (bfloat16 = 2 bytes)
        data_size_bytes = size * 2
        
        result = {
            'rank': global_rank,
            'tensor_size': size,
            'data_size_bytes': data_size_bytes,
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'world_size': nproc_per_node * node_size
        }
        results.append(result)
        if global_rank == 0:
            print(f"[Rank {global_rank}] Size: {size:>12} elements ({data_size_bytes:>12} bytes) - "
                f"Avg time: {avg_time:>8.4f} ms")
    
    # Save results to CSV
    csv_filename = os.path.join(output_dir, f"allreduce_results_rank_{global_rank}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['rank', 'tensor_size', 'data_size_bytes', 'total_time_ms', 'avg_time_ms', 'world_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[Rank {global_rank}] Results saved to {csv_filename}")

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
    parser.add_argument("--output-dir",
                        type=str,
                        default="./results",
                        help="Output directory for CSV results")
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


    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    from multiprocessing import Process

    procs = []
    for local_rank, global_rank in enumerate(
            range(node_rank * nproc_per_node, (node_rank + 1) * nproc_per_node)):
        proc = Process(target=main,
                       args=(node_size, nproc_per_node, local_rank,
                             global_rank, dp_master_ip, dp_master_port, args.output_dir))
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

    # Merge all CSV files into a single summary file with averages
    summary_file = os.path.join(args.output_dir, "allreduce_summary.csv")
    all_results = []
    
    for local_rank, global_rank in enumerate(
            range(node_rank * nproc_per_node, (node_rank + 1) * nproc_per_node)):
        csv_filename = os.path.join(args.output_dir, f"allreduce_results_rank_{global_rank}.csv")
        if os.path.exists(csv_filename):
            with open(csv_filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                all_results.extend(list(reader))
    
    if all_results:
        # Group by tensor_size and calculate averages
        from collections import defaultdict
        grouped_results = defaultdict(list)
        for result in all_results:
            tensor_size = int(result['tensor_size'])
            grouped_results[tensor_size].append({
                'data_size_bytes': int(result['data_size_bytes']),
                'total_time_ms': float(result['total_time_ms']),
                'avg_time_ms': float(result['avg_time_ms']),
                'world_size': int(result['world_size'])
            })
        
        # Calculate averages for each tensor size
        averaged_results = []
        for tensor_size, results_list in grouped_results.items():
            avg_result = {
                'tensor_size': tensor_size,
                'data_size_bytes': results_list[0]['data_size_bytes'],  # Same for all ranks
                'avg_total_time_ms': sum(r['total_time_ms'] for r in results_list) / len(results_list),
                'avg_avg_time_ms': sum(r['avg_time_ms'] for r in results_list) / len(results_list),
                'world_size': results_list[0]['world_size'],  # Same for all ranks
                'num_ranks': len(results_list)
            }
            averaged_results.append(avg_result)
        
        # Sort by tensor size
        averaged_results.sort(key=lambda x: x['tensor_size'])
        
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = ['tensor_size', 'data_size_bytes', 'avg_total_time_ms', 'avg_avg_time_ms', 'world_size', 'num_ranks']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_results)
        print(f"Summary results saved to {summary_file}")

    exit(exit_code)
