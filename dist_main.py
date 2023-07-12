# 分布式上使用pubmed数据集训练fs-gnn

from helper.parser import *
import random
import torch.multiprocessing as mp
from helper.utils import *
import dist_train
import warnings
from time import time


if __name__ == '__main__':
    # parse args
    args = create_parser()
    print(args)

    # fix the seed
    if args.fix_seed is False:
        if args.parts_per_node < args.n_partitions:
            warnings.warn('Please enable `--fix-seed` for multi-node training.')
        args.seed = random.randint(0, 1 << 31)
    print(args)

    # set the graph name
    if args.graph_name == '':
        if args.inductive:
            args.graph_name = '%s-%d-%s-%s-induc' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
        else:
            args.graph_name = '%s-%d-%s-%s-trans' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
    print(args)

    # divide the graph
    # 1. the main process load the dataset, divive the graph and save the partition
    # 2. different process load corresponding partition by its process rank
    if args.skip_partition:
        if args.n_feat == 0 or args.n_class == 0 or args.n_train == 0:
            warnings.warn('Specifying `--n-feat`, `--n-class` and `--n-train` saves data loading time.')
            g, n_feat, n_class = load_data(args.dataset)
            args.n_feat = n_feat
            args.n_class = n_class
            args.n_train = g.ndata['train_mask'].int().sum().item()
    else:
        # load the graph
        g, n_feat, n_class = load_data(args.dataset)
        if args.node_rank == 0:
            # divide the graph and save the partition
            if args.inductive:
                graph_partition(g.subgraph(g.ndata['train_mask']), args)
            else:
                graph_partition(g, args)
        args.n_class = n_class
        args.n_feat = n_feat
        args.n_train = g.ndata['train_mask'].int().sum().item()
    print(args)

    # start multi-process and run the distributed training
    if args.backend == 'gloo':
        processes = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = torch.cuda.device_count()
            devices = [f'{i}' for i in range(n)]
        # set how to start a new process
        mp.set_start_method('spawn', force=True)
        start_id = args.node_rank * args.parts_per_node

        # 子进程向主进程传输执行结果
        queue = mp.Queue()

        # 统计达到某一个准确率target_acc所用的时间
        # 从启动partition个子进程开始到所有子进程退出
        start_time = time()
        for i in range(start_id, min(start_id + args.parts_per_node, args.n_partitions)):
            # maybe different workers use the same gpu
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
            p = mp.Process(target=dist_train.init_processes, args=(i, args.n_partitions, queue, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        end_time = time()
        print("time: ", end_time - start_time)

        # 获取子进程的执行结果
        model_param_grad_communication_volume = 0
        feature_embedding_communication_volume = 0
        epoch = 0
        while not queue.empty():
            ret = queue.get()
            if ret['rank'] == 0:
                model_param_grad_communication_volume = ret['model_param_grad_communication_volume']
                epoch = ret['epoch']

            tmp = ret['feature_embedding_communication_volume']
            feature_embedding_communication_volume += tmp
            print(f"[{ret['rank']}] feature and embedding communication volume {tmp}")

        print(f"args: update_freq {args.log_every}, {'fs' if args.fs else 'no-fs'}, lr {args.lr}, target-acc {args.target_acc},epoch {epoch}")
        print(f"model param grad communication volume\t{model_param_grad_communication_volume}")
        print(f"feature and embedding communication volume\t{feature_embedding_communication_volume}")
    elif args.backend == 'nccl':
        raise NotImplementedError
    elif args.backend == 'mpi':
        raise NotImplementedError
    else:
        raise ValueError
