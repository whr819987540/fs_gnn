import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from time import sleep


def init_processes(rank,size):
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "18118"
    dist.init_process_group(backend='gloo', rank=rank, world_size=size)

    run(rank,size)
    
def run(rank,size):
    rank, size = dist.get_rank(), dist.get_world_size()
    # 定义需要通信的数据
    tensor = torch.tensor([rank+1],dtype=torch.int)
    tensor = torch.square(tensor)
    # received_tensor = torch.tensor([rank,100])
    # 待接收的数据与接收数据的tensor的类型必须一致
    received_tensor = torch.zeros(1,dtype=torch.int)
    print(tensor,received_tensor)
    
    if rank == 0:
        dist.broadcast(tensor, src=rank)
        print(f"{rank} broadcast send {tensor}")

        dist.broadcast(received_tensor,src=1)    
        print(f"{rank} broadcast recv {received_tensor}")

    else:
        dist.broadcast(received_tensor,src=0)    
        print(f"{rank} broadcast recv {received_tensor}")

        dist.broadcast(tensor, src=rank)
        print(f"{rank} broadcast send {tensor}")

# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # 发送tensor到进程1
#         dist.send(tensor=tensor, dst=1)
#         print('Rank 0 sent data to rank 1')
#         # 接收tensor从进程1        
#         dist.recv(tensor=tensor, src=1)
#         print('Rank 0 received data from rank 1')
#     else:
#         # 接收tensor从进程0
#         dist.recv(tensor=tensor, src=0)
#         print('Rank 1 received data from rank 0')
#         tensor += 1
#         # 发送tensor到进程0
#         dist.send(tensor=tensor, dst=0)
#         print('Rank 1 sent data to rank 0')

#     print('Rank ', rank, ' has data ', tensor[0])

if __name__ == "__main__":
    # use gloo as the communication backend
    
    size = 2
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_processes,args=(rank,size))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
        