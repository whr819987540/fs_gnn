import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool
from torch import distributed as dist
from threading import Lock

class Reducer(object):

    def __init__(self):
        super(Reducer, self).__init__()
        self._data_cpu = {}
        self._pool = None
        self._handles = []
        self._stream = None
        self.communication_volume = 0 # rank 0的进程统计梯度的通信量
        self.lock = Lock() # 互斥锁访问communication_volume
        self.partitions = None # 总的进程数
        self.rank = None # 当前进程的rank
        
    def init(self, model):
        cnt = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            print(f"{name}")
            cnt += 1
            # 规约时的临时数据放在cpu上
            # 可以提高异步传输时计算与通信的效率，但是会增大对主存的占用
            # 根据内存情况选择是否开启
            self._data_cpu[name] = (torch.zeros_like(param.data, pin_memory=False, device='cpu'), dist.new_group())
            print("dist.new_group() ok")
        self._pool = ThreadPool(processes=cnt) # 不一定能创建这么多线程
        # self._pool = ThreadPool(processes=2) 
        self._stream = torch.cuda.Stream()
        self.rank, self.partitions = dist.get_rank(), dist.get_world_size()

    def reduce(self, param, name, data, n_train):
        def create_stream():
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                data.div_(n_train)
                data_cpu, group = self._data_cpu[name]
                data_cpu.copy_(data)
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group)
                param.grad.copy_(data_cpu, non_blocking=True)

            if self.rank == 0:
                with self.lock:
                    # *2是因为all_reduce被认为有两个过程
                    # reduce: 其它process向rank 0发送
                    # braodcase: rank 0向其它process发送
                    self.communication_volume += data_cpu.numel() * data.element_size() * (self.partitions - 1) * 2

        self._handles.append(self._pool.apply_async(create_stream))

    def synchronize(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
