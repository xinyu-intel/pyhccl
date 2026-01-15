import ctypes
import logging
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


onecclResult_t = ctypes.c_int
onecclComm_t = ctypes.c_void_p


class onecclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_uint8 * 1024), ("length", ctypes.c_size_t)]


xpuStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

onecclDataType_t = ctypes.c_int


class onecclDataTypeEnum:
    onecclInt8 = 0
    onecclChar = 0
    onecclUint8 = 1
    onecclInt32 = 2
    onecclInt = 2
    onecclUint32 = 3
    onecclInt64 = 4
    onecclUint64 = 5
    onecclFloat16 = 6
    onecclHalf = 6
    onecclFloat32 = 7
    onecclFloat = 7
    onecclFloat64 = 8
    onecclDouble = 8
    onecclBfloat16 = 9
    onecclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.onecclInt8
        if dtype == torch.uint8:
            return cls.onecclUint8
        if dtype == torch.int32:
            return cls.onecclInt32
        if dtype == torch.int64:
            return cls.onecclInt64
        if dtype == torch.float16:
            return cls.onecclFloat16
        if dtype == torch.float32:
            return cls.onecclFloat32
        if dtype == torch.float64:
            return cls.onecclFloat64
        if dtype == torch.bfloat16:
            return cls.onecclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


onecclRedOp_t = ctypes.c_int


class onecclRedOpTypeEnum:
    onecclSum = 0
    onecclProd = 1
    onecclMin = 2
    onecclMax = 3
    onecclAvg = 4
    onecclOpNone = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.onecclSum
        if op == ReduceOp.PRODUCT:
            return cls.onecclProd
        if op == ReduceOp.MAX:
            return cls.onecclMax
        if op == ReduceOp.MIN:
            return cls.onecclMin
        if op == ReduceOp.AVG:
            return cls.onecclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class ONECCLLibrary:
    exported_functions = [
        # const char* onecclGetErrorString(onecclResult_t result);
        Function("onecclGetErrorString", ctypes.c_char_p, [onecclResult_t]),
        # onecclResult_t onecclGetVersion(int* version);
        Function("onecclGetVersion", onecclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # onecclResult_t onecclGetUniqueId(onecclUniqueId* uniqueId);
        Function("onecclGetUniqueId", onecclResult_t, [ctypes.POINTER(onecclUniqueId)]),
        # onecclResult_t CCL_C_API onecclSetDevice(uint32_t index);
        Function("onecclSetDevice", onecclResult_t, [ctypes.c_uint32]),
        # onecclResult_t onecclCommInitRank(onecclComm_t* comm, int nranks, onecclUniqueId commId, int rank);
        # note that onecclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "onecclCommInitRank",
            onecclResult_t,
            [ctypes.POINTER(onecclComm_t), ctypes.c_int, onecclUniqueId, ctypes.c_int],
        ),
        # onecclResult_t onecclAllReduce(const void*    sendbuff,
        #                    void*          recvbuff,
        #                    size_t         count,
        #                    onecclDataType_t datatype,
        #                    onecclRedOp_t    reduceOp,
        #                    onecclComm_t     comm,
        #                    void*          stream_handle);
        # note that ctypes.c_void_p is a pointer type, so the last argument
        # is a pointer
        Function(
            "onecclAllReduce",
            onecclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                onecclDataType_t,
                onecclRedOp_t,
                onecclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # onecclResult_t onecclAllGather(const void*    sendbuff,
        #                    void*          recvbuff,
        #                    size_t         sendcount,
        #                    onecclDataType_t datatype,
        #                    onecclComm_t     comm,
        #                    void*          stream_handle);
        # note that ctypes.c_void_p is a pointer type, so the last argument
        # is a pointer
        Function(
            "onecclAllGather",
            onecclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                onecclDataType_t,
                onecclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # onecclResult_t onecclReduceScatter(const void*    sendbuff,
        #                        void*          recvbuff,
        #                        size_t         recvcount,
        #                        onecclDataType_t datatype,
        #                        onecclRedOp_t    reduceOp,
        #                        onecclComm_t     comm,
        #                        void*          stream_handle);
        # note that ctypes.c_void_p is a pointer type, so the last argument
        # is a pointer
        Function(
            "onecclReduceScatter",
            onecclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                onecclDataType_t,
                onecclRedOp_t,
                onecclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # onecclResult_t onecclSend(const void* sendbuff,
        #                       size_t count,
        #                       onecclDataType_t datatype,
        #                       int peer,
        #                       onecclComm_t comm,
        #                       void* stream);
        Function(
            "onecclSend",
            onecclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                onecclDataType_t,
                ctypes.c_int,
                onecclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # onecclResult_t onecclRecv(void* recvbuff,
        #                       size_t count,
        #                       onecclDataType_t datatype,
        #                       int peer,
        #                       onecclComm_t comm,
        #                       void* stream);
        Function(
            "onecclRecv",
            onecclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                onecclDataType_t,
                ctypes.c_int,
                onecclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # onecclResult_t onecclBroadcast(const void*    sendbuff,
        #                    void*          recvbuff,
        #                    size_t         count,
        #                    onecclDataType_t datatype,
        #                    int            root,
        #                    onecclComm_t     comm,
        #                    void*          stream_handle);
        Function(
            "onecclBroadcast",
            onecclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                onecclDataType_t,
                ctypes.c_int,
                onecclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # onecclResult_t onecclCommDestroy(onecclComm_t comm);
        Function("onecclCommDestroy", onecclResult_t, [onecclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        self.streamHandle = ctypes.c_void_p()

        so_file = so_file or "libccl.so"

        try:
            if so_file not in ONECCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                ONECCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = ONECCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load ONECCL library from %s ."
                "It is expected if you are not running on Gaudi."
                "Otherwise, the oneccl library might not exist, be corrupted "
                "or it does not support the current platform %s.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in ONECCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in ONECCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            ONECCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = ONECCLLibrary.path_to_dict_mapping[so_file]

    def onecclGetErrorString(self, result: onecclResult_t) -> str:
        return self._funcs["onecclGetErrorString"](result).decode("utf-8")

    def ONECCL_CHECK(self, result: onecclResult_t) -> None:
        if result != 0:
            error_str = self.onecclGetErrorString(result)
            raise RuntimeError(f"ONECCL error: {error_str}")

    def onecclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.ONECCL_CHECK(self._funcs["onecclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 2604 --> "2.6.4"
        major = version_str[0].lstrip("0")
        minor = version_str[1].lstrip("0")
        patch = version_str[2:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def onecclGetUniqueId(self) -> onecclUniqueId:
        unique_id = onecclUniqueId()
        self.ONECCL_CHECK(self._funcs["onecclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def onecclSetDevice(self, index: int) -> None:
        self.ONECCL_CHECK(self._funcs["onecclSetDevice"](index))

    def onecclCommInitRank(
        self, world_size: int, unique_id: onecclUniqueId, rank: int
    ) -> onecclComm_t:
        comm = onecclComm_t()
        self.ONECCL_CHECK(
            self._funcs["onecclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def onecclAllReduce(
        self,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        count: int,
        datatype: int,
        op: int,
        comm: onecclComm_t,
        stream: ctypes.c_void_p,
    ) -> None:
        # `datatype` actually should be `onecclDataType_t`
        # and `op` should be `onecclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.ONECCL_CHECK(
            self._funcs["onecclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def onecclReduceScatter(
        self,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        count: int,
        datatype: int,
        op: int,
        comm: onecclComm_t,
        stream: ctypes.c_void_p,
    ) -> None:
        # `datatype` actually should be `onecclDataType_t`
        # and `op` should be `onecclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.ONECCL_CHECK(
            self._funcs["onecclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def onecclAllGather(
        self,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        count: int,
        datatype: int,
        comm: onecclComm_t,
        stream: ctypes.c_void_p,
    ) -> None:
        # `datatype` actually should be `onecclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.ONECCL_CHECK(
            self._funcs["onecclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def onecclSend(
        self,
        sendbuff: ctypes.c_void_p,
        count: int,
        datatype: int,
        dest: int,
        comm: onecclComm_t,
        stream: ctypes.c_void_p,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def onecclRecv(
        self,
        recvbuff: ctypes.c_void_p,
        count: int,
        datatype: int,
        src: int,
        comm: onecclComm_t,
        stream: ctypes.c_void_p,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def onecclBroadcast(
        self,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        count: int,
        datatype: int,
        root: int,
        comm: onecclComm_t,
        stream: ctypes.c_void_p,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def onecclCommDestroy(self, comm: onecclComm_t) -> None:
        self.ONECCL_CHECK(self._funcs["onecclCommDestroy"](comm))


__all__ = [
    "ONECCLLibrary",
    "onecclDataTypeEnum",
    "onecclRedOpTypeEnum",
    "onecclUniqueId",
    "onecclComm_t",
    "ctypes.c_void_p",
    "ctypes.c_void_p",
]
