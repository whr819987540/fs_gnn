import torch
import torch.distributed as dist
from torch.multiprocessing import Process 
import os

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # 发送tensor到进程1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 sent data to rank 1')
    else:
        # 接收tensor从进程0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 received data from rank 0')

    req.wait()

    if rank == 1:
        tensor += 1
        # 发送tensor到进程0
        req = dist.isend(tensor=tensor, dst=0)
        print('Rank 1 sent data to rank 0')
    else:
        # 接收tensor从进程1
        req = dist.irecv(tensor=tensor, src=1)
        print('Rank 0 received data from rank 1')

    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "18118"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2 
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
