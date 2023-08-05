# 分布式上使用pubmed数据集训练fs-gnn

from helper.parser import *
import random
import torch.multiprocessing as mp
from helper.utils import *
import dist_gcn_train
import warnings
from time import time
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from os.path import join
import logging
from module import fs_layer



if __name__ == '__main__':
    # parse args
    args = create_parser()

    # init logging
    log_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    init_logging(args,log_id)
    logger = logging.getLogger("main")
    logger.info(args)

    # fix the seed
    if args.fix_seed is False:
        if args.parts_per_node < args.n_partitions:
            warnings.warn('Please enable `--fix-seed` for multi-node training.')
        args.seed = random.randint(0, 1 << 31)
    logger.info(args)

    # set the graph name
    if args.graph_name == '':
        if args.inductive:
            args.graph_name = '%s-%d-%s-%s-induc' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
        else:
            args.graph_name = '%s-%d-%s-%s-trans' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
    logger.info(args)

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
    logger.info(args)

    # set the real dimension of the feature
    # model为gcn_first
    # pretrain为true，fs必为true（因为pretrain就是在训练fs），表示在pretrain阶段训练fs, 只有在这种情况下model中才有fs, 并且需要导出fs的参数
    # pretrain为flase，fs为true，表示在offline阶段，需要加载已经训练好的fs, 然后处理feature, 但model中没有fs层
    # pretrain为false，fs为false，表示使用gcn_first，但model中没有fs层true
    # pretrain为flase，fs为true时模型结构会因为fs的处理而改变，所以需要重新设置n_feat
    if args.pretrain==False and args.fs==True:
        args.n_feat = int(args.n_feat*args.fsratio)

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

        # FIXME: 因为rank之间feature的通信存在问题，所以临时用本地存储的全图feature来模拟点对点通信
        # DONE: 修复了feature的通信问题，不再需要这个临时方案
        # all_feat = g.ndata['feat'].cuda()

        # 建立node globalid到partition id的映射
        # 建立globalid与index的映射
        all_partition_detail, mapper_manager = get_all_partition_detail_and_globalid_index_mapper_manager(g.num_nodes(), min(start_id + args.parts_per_node, args.n_partitions)-start_id,args)
            
        # 统计训练一定轮数所用的时间
        # 实际训练轮数由rank=0的进程返回
        # 从启动partition个子进程开始到所有子进程退出
        start_time = time()
        for i in range(start_id, min(start_id + args.parts_per_node, args.n_partitions)):
            # maybe different workers use the same gpu
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
            p = mp.Process(target=dist_gcn_train.init_processes, args=(
                i, args.n_partitions, queue, args, all_partition_detail, mapper_manager,log_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        end_time = time()
        logger.info(f"time: {end_time - start_time}")
        print(f"time: {end_time - start_time}")

        # 获取子进程的执行结果
        model_param_grad_communication_volume = 0
        feature_embedding_communication_volume = 0
        feature_communication_volume = 0
        sampled_nodes_num = 0
        feature_embedding_communication_volume_list = [0]*args.n_partitions
        epoch = 0
        
        while not queue.empty():
            ret = queue.get()
            if ret['rank'] == 0:
                model_param_grad_communication_volume = ret['model_param_grad_communication_volume']
                epoch = ret['epoch']
                writer_path = ret['writer_path']
                fs_run_time = ret['fs_run_time']
                normal_run_time = ret['normal_run_time']

            tmp = ret['feature_embedding_communication_volume']
            feature_embedding_communication_volume_list[ret['rank']] = tmp
            feature_embedding_communication_volume += tmp
            logger.info(f"[{ret['rank']}] feature and embedding communication volume {tmp}")
            print(f"[{ret['rank']}] feature and embedding communication volume {tmp}")
            
            a = ret['feature_communication_volume']
            feature_communication_volume += a
            b = ret['sampled_nodes_num']
            sampled_nodes_num += b
            logger.info(f"[{ret['rank']}] feature communication volume {a}, sampled_nodes_num {b}")
            print(f"[{ret['rank']}] feature communication volume {a}, sampled_nodes_num {b}")
            

        print(f"args: update_freq {args.log_every}, {'fs' if args.fs else 'no-fs'}, lr {args.lr}, training times {epoch+1}")
        print(f"model param grad communication volume\t{model_param_grad_communication_volume}")
        print(f"feature and embedding communication volume\t{feature_embedding_communication_volume}")
        print(f"AVG(feature communication volume)\t{feature_communication_volume/sampled_nodes_num}")

        writer = SummaryWriter(f"{ret['writer_path']} result")
        
        writer.add_text("args", f"update_freq {args.log_every}, {'fs' if args.fs else 'no-fs'}, lr {args.lr}, training times {epoch+1}", 0)
        writer.add_text("model_param_grad_communication_volume", f"{model_param_grad_communication_volume}", 0)
        
        for i in range(args.n_partitions):
            writer.add_text(f"[{i}]feature and embedding communication volume", f"{feature_embedding_communication_volume_list[i]}", 0)
            
        writer.add_text("feature_and_embedding_communication_volume", f"{feature_embedding_communication_volume}", 0)
        
    elif args.backend == 'nccl':
        raise NotImplementedError
    elif args.backend == 'mpi':
        raise NotImplementedError
    else:
        raise ValueError
