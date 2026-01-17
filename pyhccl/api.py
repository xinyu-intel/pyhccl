from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from .binding import (ONECCLLibrary, buffer_type, onecclComm_t, onecclDataTypeEnum,
                      onecclRedOpTypeEnum, onecclUniqueId, xpuStream_t)
from .utils import StatelessProcessGroup


class PyHcclCommunicator:

    def __init__(
        self, group: StatelessProcessGroup, library_path: Optional[str] = None
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyHcclCommunicator to.
            library_path: the path to the HCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        self.rank = group.rank
        self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            self.oneccl = ONECCLLibrary(library_path)
        except Exception:
            # disable because of missing HCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        if self.rank == 0:
            # get the unique id from HCCL
            self.unique_id = self.oneccl.onecclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = onecclUniqueId()

        self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        self.comm: onecclComm_t = self.oneccl.onecclCommInitRank(
            self.world_size, self.unique_id, self.rank
        )
        
        self.oneccl.onecclSetDevice(self.rank)
        
        self.stream = torch.xpu.current_stream().sycl_queue

        # A small all_reduce for warmup.
        data = torch.ones(1, device="xpu")
        self.all_reduce(data)
        torch.xpu.synchronize()
        del data

    def all_reduce(
        self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> torch.Tensor:
        if self.disabled:
            return None
        assert in_tensor.device.type == "xpu", f"the input tensor should be on xpu"

        self.oneccl.onecclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(in_tensor.data_ptr()),
            in_tensor.numel(),
            onecclDataTypeEnum.from_torch(in_tensor.dtype),
            onecclRedOpTypeEnum.from_torch(op),
            self.comm,
            xpuStream_t(self.stream),
        )
        return in_tensor

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        if self.disabled:
            return
        assert input_tensor.device.type == "xpu", f"the input tensor should be on xpu"

        self.oneccl.onecclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            onecclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            xpuStream_t(self.stream),
        )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        if self.disabled:
            return
        assert input_tensor.device.type == "xpu", f"the input tensor should be on xpu"
        assert output_tensor.device.type == "xpu", f"the output tensor should be on xpu"

        self.oneccl.onecclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            onecclDataTypeEnum.from_torch(input_tensor.dtype),
            onecclRedOpTypeEnum.from_torch(op),
            self.comm,
            xpuStream_t(self.stream),
        )

    def send(self, tensor: torch.Tensor, dst: int):
        if self.disabled:
            return
        assert tensor.device.type == "xpu", f"the input tensor should be on xpu"

        self.oneccl.onecclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            xpuStream_t(self.stream),
        )

    def recv(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        assert tensor.device.type == "xpu", f"the input tensor should be on xpu"

        self.oneccl.onecclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            xpuStream_t(self.stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        assert tensor.device.type == "xpu", f"the input tensor should be on xpu"
        sendbuff = buffer_type(tensor.data_ptr())
        recvbuff = buffer_type(tensor.data_ptr())

        self.oneccl.onecclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            xpuStream_t(self.stream),
        )
