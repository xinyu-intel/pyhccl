#!/usr/bin/env python3
"""
General-purpose CCL collective benchmark using pyccl.

Benchmarks individual CCL primitives (all_reduce, all_gather, reduce_scatter,
broadcast, send_recv) across configurable tensor shapes, dtypes, and devices.

Benchmark methodology:
  - Cold data: pre-allocates N independent input buffers (one per timed iter)
    so the collective never hits warm device cache from a previous iteration.
  - GEMM pre-fire: a large matmul is launched before the timed region to keep
    the device execution pipeline in a steady compute-active state, preventing
    cold-start / idle-power effects from inflating measured latency.
  - Single event pair: one start/end event around all iterations with a single
    synchronize at the end — measures true pipelined throughput, not per-call
    round-trip.

Usage (single node):
  python bench_ccl.py --nproc-per-node 2 --op all_reduce \
      --shapes 1024x4096,2048x4096 --dtype bfloat16

Usage (multi-node):
  python bench_ccl.py --nproc-per-node 4 --node-size 2 --node-rank 0 \
      --master-addr 10.0.0.1 --op all_reduce all_gather \
      --shapes 4096x7168 --dtype float16
"""

import argparse
import csv
import os
import time
from multiprocessing import Process

import torch
from pyccl import PyCCLCommunicator
from pyccl.utils import StatelessProcessGroup


SUPPORTED_OPS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "send_recv"]

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int8": torch.int8,
}


def parse_shapes(shapes_str):
    """Parse shape string like '1024x4096,2048x7168' into list of tuples."""
    shapes = []
    for s in shapes_str.split(","):
        dims = tuple(int(d) for d in s.strip().split("x"))
        shapes.append(dims)
    return shapes


def format_bytes(num_bytes):
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def shape_str(shape):
    return "x".join(str(d) for d in shape)


def init_comm(master_addr, master_port, rank, world_size):
    pg = StatelessProcessGroup.create(
        host=master_addr, port=master_port, rank=rank, world_size=world_size
    )
    return PyCCLCommunicator(pg)


def make_gemm_fixture(shape, device):
    """Create a large GEMM pair to prime the execution pipeline before timing."""
    m = shape[0]
    k = shape[-1]
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(k, k, dtype=torch.bfloat16, device=device)
    return a, b


def make_cold_buffers(shape, dtype, device, count):
    """Pre-allocate independent input buffers for cold-cache measurement."""
    if dtype.is_floating_point:
        return [torch.randn(shape, device=device, dtype=dtype) for _ in range(count)]
    return [torch.randint(0, 127, shape, device=device, dtype=dtype) for _ in range(count)]


def bench_all_reduce(comm, shape, dtype, device, warmup, iters):
    bufs = make_cold_buffers(shape, dtype, device, iters)
    warmup_t = bufs[0]
    for _ in range(warmup):
        comm.all_reduce(warmup_t)

    gemm_a, gemm_b = make_gemm_fixture(shape, device)
    return _timed_loop(lambda i: comm.all_reduce(bufs[i]), device, iters, gemm_a, gemm_b)


def bench_all_gather(comm, shape, dtype, device, warmup, iters):
    recv_shape = (shape[0] * comm.world_size, *shape[1:])
    bufs = make_cold_buffers(shape, dtype, device, iters)
    recv_t = torch.empty(recv_shape, device=device, dtype=dtype)
    for _ in range(warmup):
        comm.all_gather(recv_t, bufs[0])

    gemm_a, gemm_b = make_gemm_fixture(shape, device)
    return _timed_loop(lambda i: comm.all_gather(recv_t, bufs[i]), device, iters, gemm_a, gemm_b)


def bench_reduce_scatter(comm, shape, dtype, device, warmup, iters):
    send_shape = (shape[0] * comm.world_size, *shape[1:])
    bufs = make_cold_buffers(send_shape, dtype, device, iters)
    recv_t = torch.empty(shape, device=device, dtype=dtype)
    for _ in range(warmup):
        comm.reduce_scatter(recv_t, bufs[0])

    gemm_a, gemm_b = make_gemm_fixture(send_shape, device)
    return _timed_loop(lambda i: comm.reduce_scatter(recv_t, bufs[i]), device, iters, gemm_a, gemm_b)


def bench_broadcast(comm, shape, dtype, device, warmup, iters):
    bufs = make_cold_buffers(shape, dtype, device, iters)
    for _ in range(warmup):
        comm.broadcast(bufs[0], src=0)

    gemm_a, gemm_b = make_gemm_fixture(shape, device)
    return _timed_loop(lambda i: comm.broadcast(bufs[i], src=0), device, iters, gemm_a, gemm_b)


def bench_send_recv(comm, shape, dtype, device, warmup, iters):
    bufs = make_cold_buffers(shape, dtype, device, iters)
    rank = comm.rank
    world_size = comm.world_size
    if world_size % 2 != 0:
        raise ValueError(f"send_recv benchmark requires even world_size, got {world_size}")
    peer = rank + 1 if rank % 2 == 0 else rank - 1

    def _op(i):
        if rank % 2 == 0:
            comm.send(bufs[i], dst=peer)
            comm.recv(bufs[i], src=peer)
        else:
            comm.recv(bufs[i], src=peer)
            comm.send(bufs[i], dst=peer)

    for _ in range(warmup):
        _op(0)

    gemm_a, gemm_b = make_gemm_fixture(shape, device)
    return _timed_loop(_op, device, iters, gemm_a, gemm_b)


def _timed_loop(op_fn, device, iters, gemm_a, gemm_b):
    """
    Time `iters` calls of op_fn(i) with:
      - A GEMM pre-fire to prime the pipeline
      - A single start/end event pair (no per-iter sync)
    """
    if "xpu" in device:
        start_ev = torch.Event(enable_timing=True)
        end_ev = torch.Event(enable_timing=True)

        torch.mm(gemm_a, gemm_b)
        start_ev.record()
        for i in range(iters):
            op_fn(i)
        end_ev.record()
        end_ev.synchronize()
        total_ms = start_ev.elapsed_time(end_ev)
    else:
        torch.mm(gemm_a, gemm_b)
        start = time.perf_counter()
        for i in range(iters):
            op_fn(i)
        end = time.perf_counter()
        total_ms = (end - start) * 1000
    return total_ms / iters


def compute_bandwidths(op, data_bytes, world_size, time_s):
    """
    Compute algBw and busBw following the oneCCL benchmark convention
    (uxlfoundation/oneCCL tests/benchmark/).

    algBw = total_data_volume / time
    busBw = algBw * bus_factor

    For all_gather/reduce_scatter, the "count" passed to the collective is
    per-rank, so total volume = count * nranks * typesize = data_bytes * nranks.
    For all_reduce/broadcast/send_recv, total volume = data_bytes (the full tensor).
    """
    n = world_size
    if time_s <= 0:
        return 0.0, 0.0

    if op == "all_reduce":
        alg_bw = data_bytes / time_s
        bus_factor = 2 * (n - 1) / n if n > 1 else 0
    elif op == "all_gather":
        alg_bw = (data_bytes * n) / time_s
        bus_factor = (n - 1) / n
    elif op == "reduce_scatter":
        alg_bw = (data_bytes * n) / time_s
        bus_factor = (n - 1) / n
    elif op == "broadcast":
        alg_bw = data_bytes / time_s
        bus_factor = 1
    elif op == "send_recv":
        alg_bw = data_bytes / time_s
        bus_factor = 1
    else:
        alg_bw = data_bytes / time_s
        bus_factor = 1

    bus_bw = alg_bw * bus_factor
    return alg_bw, bus_bw


BENCH_FNS = {
    "all_reduce": bench_all_reduce,
    "all_gather": bench_all_gather,
    "reduce_scatter": bench_reduce_scatter,
    "broadcast": bench_broadcast,
    "send_recv": bench_send_recv,
}


def worker(
    local_rank, global_rank, world_size, master_addr, master_port,
    ops, shapes, dtype_name, device, warmup, iters, output_dir,
):
    if device == "xpu":
        torch.xpu.set_device(local_rank)
        device_str = f"xpu:{local_rank}"
    else:
        device_str = "cpu"

    comm = init_comm(master_addr, master_port, global_rank, world_size)
    torch_dtype = DTYPE_MAP[dtype_name]
    elem_size = torch_dtype.itemsize

    results = []

    col = (f"  {'op':<16}  {'dtype':<10}  {'shape':<14}"
           f"  {'xfer_MB':>8}  {'time_us':>10}  {'algbw_GBps':>11}  {'busbw_GBps':>11}")
    sep = "  " + "-" * (len(col) - 2)
    if global_rank == 0:
        print("\n" + sep)
        print(col)
        print(sep)

    for op in ops:
        bench_fn = BENCH_FNS[op]
        for shape in shapes:
            avg_ms = bench_fn(comm, shape, torch_dtype, device_str, warmup, iters)
            numel = 1
            for d in shape:
                numel *= d
            data_bytes = numel * elem_size
            alg_bw, bus_bw = compute_bandwidths(op, data_bytes, world_size, avg_ms * 1e-3)
            alg_bw_gbps = alg_bw / 1e9
            bus_bw_gbps = bus_bw / 1e9

            result = {
                "op": op,
                "shape": shape_str(shape),
                "dtype": dtype_name,
                "data_bytes": data_bytes,
                "avg_latency_us": round(avg_ms * 1000, 2),
                "algbw_GBps": round(alg_bw_gbps, 2),
                "busbw_GBps": round(bus_bw_gbps, 2),
                "world_size": world_size,
                "rank": global_rank,
            }
            results.append(result)

            if global_rank == 0:
                print(
                    f"  {op:<16}  {dtype_name:<10}  {shape_str(shape):<14}"
                    f"  {data_bytes/1e6:>8.3f}  {avg_ms * 1000:>10.2f}"
                    f"  {alg_bw_gbps:>11.2f}  {bus_bw_gbps:>11.2f}"
                )

    if global_rank == 0:
        print(sep)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"bench_rank_{global_rank}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="General-purpose CCL collective benchmark using pyccl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shape format:
  --shapes 1024x4096              single shape
  --shapes 1024x4096,4096x7168    multiple shapes (comma-separated)
  --shapes 8192                   1-D tensor

Available operations:
  all_reduce, all_gather, reduce_scatter, broadcast, send_recv
  Use 'all' to run all operations.

Examples:
  python bench_ccl.py --nproc-per-node 2 --op all_reduce --shapes 2048x4096
  python bench_ccl.py --nproc-per-node 4 --op all --shapes 1024x7168,4096x7168 --dtype float16
""",
    )
    parser.add_argument(
        "--op", type=str, nargs="+", default=["all_reduce"],
        choices=SUPPORTED_OPS + ["all"],
        help="Operations to benchmark (default: all_reduce)",
    )
    parser.add_argument(
        "--shapes", type=str, default="1024x1024,1024x4096,4096x4096,4096x7168",
        help="Comma-separated tensor shapes, e.g. '1024x4096,4096x7168'",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()),
        help="Tensor dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device", type=str, default="xpu", choices=["cpu", "xpu"],
        help="Device (default: xpu)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Timed iterations (default: 100)",
    )
    parser.add_argument(
        "--nproc-per-node", type=int, default=2,
        help="Number of processes per node (default: 2)",
    )
    parser.add_argument(
        "--node-size", type=int, default=1,
        help="Total number of nodes (default: 1)",
    )
    parser.add_argument(
        "--node-rank", type=int, default=0,
        help="Rank of this node (default: 0)",
    )
    parser.add_argument(
        "--master-addr", type=str, default="127.0.0.1",
        help="Master address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master-port", type=int, default=4400,
        help="Master port (default: 4400)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Directory to save CSV results (optional)",
    )

    args = parser.parse_args()

    if "all" in args.op:
        ops = SUPPORTED_OPS
    else:
        ops = args.op

    shapes = parse_shapes(args.shapes)
    world_size = args.nproc_per_node * args.node_size
    torch_dtype = DTYPE_MAP[args.dtype]

    print(f"pyccl collective benchmark  world={world_size}  device={args.device}  "
          f"dtype={args.dtype}")
    print(f"ops={ops}  shapes={[shape_str(s) for s in shapes]}  "
          f"warmup={args.warmup}  iters={args.iters}")

    procs = []
    for local_rank in range(args.nproc_per_node):
        global_rank = args.node_rank * args.nproc_per_node + local_rank
        proc = Process(
            target=worker,
            args=(
                local_rank, global_rank, world_size,
                args.master_addr, args.master_port,
                ops, shapes, args.dtype, args.device,
                args.warmup, args.iters, args.output_dir,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=600)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} (timed out after 10 minutes)")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if args.output_dir and exit_code == 0:
        _merge_results(args.output_dir, args.nproc_per_node, args.node_rank)

    exit(exit_code)


def _merge_results(output_dir, nproc_per_node, node_rank):
    """Merge per-rank CSVs into a summary with averaged latencies."""
    from collections import defaultdict

    all_rows = []
    for local_rank in range(nproc_per_node):
        global_rank = node_rank * nproc_per_node + local_rank
        path = os.path.join(output_dir, f"bench_rank_{global_rank}.csv")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            all_rows.extend(list(reader))

    if not all_rows:
        return

    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["op"], row["shape"], row["dtype"])
        grouped[key].append(row)

    summary_path = os.path.join(output_dir, "bench_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "op", "shape", "dtype", "data_bytes",
            "avg_latency_us", "algbw_GBps", "busbw_GBps", "world_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (op, shape, dtype), rows in sorted(grouped.items()):
            avg_lat = sum(float(r["avg_latency_us"]) for r in rows) / len(rows)
            avg_alg = sum(float(r["algbw_GBps"]) for r in rows) / len(rows)
            avg_bus = sum(float(r["busbw_GBps"]) for r in rows) / len(rows)
            writer.writerow({
                "op": op,
                "shape": shape,
                "dtype": dtype,
                "data_bytes": rows[0]["data_bytes"],
                "avg_latency_us": round(avg_lat, 2),
                "algbw_GBps": round(avg_alg, 2),
                "busbw_GBps": round(avg_bus, 2),
                "world_size": rows[0]["world_size"],
            })
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
