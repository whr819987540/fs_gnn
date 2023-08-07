import torch.nn.functional as F
from module.fs_model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score
from new_layer import feature_importance_gini, loss_func
from torch.utils.tensorboard import SummaryWriter
from module.gcn_module import gcn_model
from helper.sampler import layer_wise_sampling, layer_importance_sampling, node_wise_sampling
from module.gcn_module.first_method_model import GCN_first
import logging
from helper.inference import sample_full

def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, adj_matrix, layer_size, args, mode, random_selection_mask=None, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    logger = logging.getLogger("evaluate_trans")
    logger.debug("start evaluate_trans")
    # gpu
    model.eval()

    feat, labels = g.ndata['feat'], g.ndata['label']
    feat = select_feature(args, feat, random_selection_mask)

    mask = g.ndata[mode + '_mask']

    if args.model == 'gcn_first':
        gnn_layer_num = get_gnn_layer_num(layer_size,args)
        adjs, previous_indices = sample_full(adj_matrix, gnn_layer_num, args.sampling_method)
        logits = model(feat, adjs)
    elif args.model == 'gcn_second':
        pass
    elif args.model == 'gcn':
        logits = model(g, feat)
        
    acc = calc_acc(logits[mask], labels[mask])
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
    logger.debug(buf)

    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, adj_matrix, layer_size, args, epoch:int, writer:SummaryWriter, random_selection_mask=None, result_file_name=None):
    logger = logging.getLogger("evaluate_trans")
    logger.debug("start evaluate_trans")
    model.eval()
    
    feat, labels = g.ndata['feat'], g.ndata['label']
    feat = select_feature(args, feat, random_selection_mask)
        
    train_mask, test_mask, val_mask = g.ndata['train_mask'], g.ndata['test_mask'], g.ndata['val_mask']

    if args.model == 'gcn_first':
        # logger.debug("before get adj")
        # adj = get_adj_matrix_from_graph(g)
        # logger.debug("get adj")
        # adj = adj.to(matrix_value_type).to_dense().t()
        gnn_layer_num = get_gnn_layer_num(layer_size,args)
        adjs, previous_indices = sample_full(adj_matrix, gnn_layer_num, args.sampling_method)
        logits = model(feat, adjs)
    elif args.model == 'gcn_second':
        pass
    else:
        logits = model(g, feat)

    train_acc = calc_acc(logits[train_mask], labels[train_mask])
    test_acc = calc_acc(logits[test_mask], labels[test_mask])
    val_acc = calc_acc(logits[val_mask], labels[val_mask])

    train_loss = loss_func(
                results=logits[train_mask],
                labels=labels[train_mask],
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            ) / train_mask.int().sum().item()
    test_loss = loss_func(
                results=logits[test_mask],
                labels=labels[test_mask],
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            ) / train_mask.int().sum().item()
    val_loss = loss_func(
                results=logits[val_mask],
                labels=labels[val_mask],
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            ) / val_mask.int().sum().item()
    
    buf = "{:s} | Train Accuracy {:.2%} | Test Accuracy {:.2%} | Validation Accuracy {:.2%} | Train Loss {:.5f} | Test Loss {:.5f} | Validation Loss {:.5f}".format(name, train_acc, test_acc, val_acc, train_loss, test_loss, val_loss)
    
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            logger.info(f"evaluate result: {buf}")
    else:
        print(buf)
    writer.add_scalar("train_acc", train_acc, epoch)
    writer.add_scalar("test_acc", test_acc, epoch)
    writer.add_scalar("val_acc", val_acc, epoch)
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("val_loss", val_loss, epoch)

    logger.debug("stop evaluating")
    return model, train_acc, test_acc, val_acc


def average_gradients(model, n_train):
    reduce_time = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        t0 = time.time()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= n_train
        reduce_time += time.time() - t0
    return reduce_time


def move_to_cuda(graph, part, node_dict):
    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))

    return graph, part, node_dict


def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    # for the present process, the shape is None
    # for the other process, the shape is the number of boundary nodes in both the other process and the present process
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def create_inner_graph(graph, node_dict):
    # inner graph是边的起点与终点都是inner node的图
    # 也就是说边完全在这个图里面，没有被子图划分给切割开
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return construct(part, graph, pos, one_hops)


def move_train_first(graph, node_dict, boundary):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary


def create_graph_train(graph, node_dict):
    u, v = graph.edges()
    num_u = graph.num_nodes('_U')
    sel = nonzero_idx(node_dict['train_mask'][v.long()])
    u, v = u[sel], v[sel]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if graph.num_nodes('_U') < num_u:
        graph.add_nodes(num_u - graph.num_nodes('_U'), ntype='_U')
    return graph, node_dict['in_degree'][node_dict['train_mask']]


def precompute(graph, node_dict, boundary, recv_shape, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum()
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])
    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(
                # fn.copy_src(src='h', out='m'),
                fn.copy_u(u='h', out='m'),
                fn.sum(msg='m', out='h'),
                etype='_E',
            )
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[0:in_size]], dim=1)
    else:
        raise Exception


def create_model(layer_size,mu, args):
    """
        在CPU上创建模型
    """  
    if args.model == 'graphsage':
        return GraphSAGEWithFS(
            layer_size=layer_size,
            activation=F.relu,
            use_pp=args.use_pp,
            sigma=args.sigma,
            mu=mu,
            fs=args.fs,
            dropout=args.dropout,
            norm=args.norm,
            train_size=args.n_train,
            n_linear=args.n_linear
        )
    elif args.model == 'gcn':
        return gcn_model.GCN(
            layer_size=layer_size,
            dropout=args.dropout,
            sigma=args.sigma,
            mu=mu,
            fs=args.fs,
        )
    elif args.model == 'gcn_first':
        # model为gcn_first
        # pretrain为true，fs必为true（因为pretrain就是在训练fs），表示在pretrain阶段训练fs, 只有在这种情况下model中才有fs, 并且需要导出fs的参数
        # pretrain为flase，fs为true，表示在offline阶段，需要加载已经训练好的fs, 然后处理feature, 但model中没有fs层
        # pretrain为false，fs为false，表示使用gcn_first，但model中没有fs层true
        if args.pretrain == True and args.fs == False:
            raise ValueError
        elif args.sampling_method  == "layer_importance_sampling" and args.pretrain == True and args.fs == True:
            if args.fs_init_method=="gini":
                path = join('gini',f"{args.dataset}_gini_impurity.pth")
                weights = torch.load(path)
            elif args.fs_init_method=="random":
                weights = None
            else:
                raise ValueError
        else:
            weights = None
            
        return GCN_first(
            layer_size=layer_size,
            dropout=args.dropout,
            weights=weights,
            fs=args.fs,
            pretrain=args.pretrain,
        )
    else:
        raise NotImplementedError


def reduce_hook(param, name, args, optimizer:torch.optim.Optimizer):
    def fn(grad):
        # OPT3
        epoch = optimizer.state['step']['epoch']
        rank = dist.get_rank()
        if rank == 0:
            print(f"epoch: {epoch}, name: {name}, time: {time.time()}")
        # print("epoch",name,epoch)
        # epoch layers.3.linear2.bias 98
        # epoch layers.3.linear2.weight 98
        # epoch layers.3.linear1.bias 98
        # epoch layers.3.linear1.weight 98
        # epoch norm.2.weight 98
        # epoch norm.2.bias 98
        # epoch layers.2.linear2.bias 98
        # epoch layers.2.linear2.weight 98
        # epoch layers.2.linear1.bias 98
        # epoch layers.2.linear1.weight 98

        if get_update_flag(epoch, args):
            ctx.reducer.reduce(param, name, grad, args.n_train)
    return fn


def construct(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g


def extract(graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    sel = (node_dict['part_id'] < size)
    for key in node_dict.keys():
        if node_dict[key].shape[0] == sel.shape[0]:
            node_dict[key] = node_dict[key][sel]
    graph = dgl.node_subgraph(graph, sel, store_ids=False)
    return graph, node_dict


def get_update_flag(epoch, args):
    return epoch % args.log_every == 0


def run(graph, node_dict, gpb, queue, args, all_partition_detail, mapper_manager, all_feat):
    """
        graph is the subgraph
        node_dict:
        part_id (int): the partition id of each node, from 0 to n_partitions-1
            including the inner nodes and boundary nodes
        inner_node: whether the node is inner node
        torch.unique((node_dict['part_id']==0)==(node_dict['inner_node']))
        tensor([True], device='cuda:0')
        label, feat, in_degree: only for inner nodes
        train_mask, val_mask, test_mask: only for inner nodes
    """
    index_type = torch.int64
    matrix_value_type = torch.int32

    rank, size = dist.get_rank(), dist.get_world_size()

    logger = logging.getLogger(f"[{rank}]")
    logger.info("start")

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # load the whole graph
    if rank == 0:
        full_g, n_feat, n_class = load_data(args.dataset)
        full_g = full_g.to(torch.device('cuda'))
        # full_g_adj_matrix 在gcn_first模型中被用到
        # 用来evaluate
        full_g_adj_matrix = get_adj_matrix_from_graph(full_g)
        full_g_adj_matrix = full_g_adj_matrix.to(matrix_value_type).to_dense().t()
        
        # if args.inductive:
        #     logger.info("inductive split")
        #     _, val_g, test_g = inductive_split(full_g)
        # else:
        #     val_g, test_g = full_g.clone(), full_g.clone()

    # record the training results
    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)

    # inner graph是边的起点与终点都是inner node的图
    # dgl对于边界节点采取的策略是复制，即某条边连接的两个子图都拥有该边界节点
    part = create_inner_graph(graph.clone(), node_dict)
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()
    logger.info(f"num_in: {num_in}") # equal to part.num_nodes()
    logger.info(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges ,{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict)
    logger.info("move_to_cuda")

    # 获取边界节点
    boundary = get_boundary(node_dict, gpb)

    layer_size = get_layer_size(args)
    logger.info(f"layer_size: {layer_size}")

    pos = get_pos(node_dict, gpb)
    graph = order_graph(part, graph, gpb, node_dict, pos)
    # store the in_deg for inner nodes
    in_deg = node_dict['in_degree']
    logger.info("order_graph")

    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)
    logger.info("move_train_first")

    # recv_shape有size个分量
    # 对于第i个分量，如果i==rank，那么该分量为None
    # 否则，该分量为第i个进程与当前进程的边界节点数，也就是待传输节点数
    recv_shape = get_recv_shape(node_dict)
    logger.info(f"[{rank}] get_recv_shape {recv_shape}")

    ctx.buffer.init_buffer(        
        num_in=num_in,
        num_all=graph.num_nodes('_U'), 
        boundary=boundary,
        f_recv_shape=recv_shape, 
        # layer_size=layer_size[:args.n_layers-args.n_linear],
        layer_size=layer_size[:len(layer_size) - args.n_linear - 1],
        use_pp=args.use_pp, 
        backend=args.backend,
        pipeline=args.enable_pipeline, 
        corr_feat=args.feat_corr,
        corr_grad=args.grad_corr, 
        corr_momentum=args.corr_momentum
    )
    logger.info("init buffer to reduce model gradinets")

    if args.use_pp:
        logger.info(node_dict['feat'].shape)
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)
        logger.info(node_dict['feat'].shape)

    del boundary
    del part
    del pos

    torch.manual_seed(args.seed)

    feat = node_dict['feat']
    random_selection_mask = None
    if args.model == 'gcn_first' and (args.sampling_method=="layer_importance_sampling" or args.sampling_method=="layer_wise_sampling") and args.pretrain == False and args.fs == True and args.fs_init_method=="seed":
        # 传输中的类型不能是bool
        random_selection_mask = torch.zeros(feat.shape[1],dtype=matrix_value_type)
        if rank==0:
            index = torch.randperm(feat.shape[1],dtype=index_type)[:args.n_feat]
            random_selection_mask[index]=1
            dist.broadcast(random_selection_mask,src=0)
        else:
            dist.broadcast(random_selection_mask,src=0)
        # 充当索引, 类型必须是bool
        random_selection_mask = random_selection_mask.bool().cuda()
    
    feat = select_feature(args, feat, random_selection_mask)
    labels = node_dict['label']
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()

    # model
    train_x = feat[train_mask]
    train_y = labels[train_mask]
    mu = feature_importance_gini(train_x, train_y)
    with torch.no_grad():
        tmp_logger = logging.getLogger(f"[{rank}] model init")
        if rank == 0:
            # 对于rank 0即实际完成初始化的worker，random_init_fs为args值
            model = create_model(copy.deepcopy(layer_size), mu, args)    
            model.cuda()
            # TODO: 模型分发, rank 0负责将初始化模型发送给其它rank
            # DONE: 模型分发
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
                tmp_logger.debug(f"[{rank}] send param {param} {param.shape}")
        else:
            # 对于rank 不为0的worker，random_init_fs恒为True
            # 以免在初始化时使用全图train来初始化fs
            model = create_model(copy.deepcopy(layer_size), mu, args)   
            model.cuda()
            # TODO: 接收模型参数
            # DONE: 接收模型参数
            for param in model.parameters():
                # [64, 500] [64]
                # [16,64] [16]
                # [3,16] [3]
                dist.broadcast(param.data,src=0)
                tmp_logger.debug(f"[{rank}] recv param {param} {param.shape}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # create the same type tensor with model parameters for parameter reducing on cpu
    ctx.reducer.init(model)
    logger.info(f"init reducer")

    # OPT3
    # reduce_hook在每次训练使用backward计算出梯度后都会被调用
    # 但是目前希望在间隔几个epoch才进行梯度聚合
    # 所以需要将epoch作为参数传进去
    # 由于reduce_hook的参数需要在注册时绑定，所以需要将epoch放到一个类中
    # 可以放到optimizer中
    # tell optimizer the epoch
    optimizer.state['step']['epoch'] = 0
    # register the hook for gradient reducing on cpu
    # the hook will be called after the local gradient is computed
    for i, (name, param) in enumerate(model.named_parameters()):
        
        # param.register_hook(reduce_hook(param, name, args.n_train))
        param.register_hook(reduce_hook(param, name, args, optimizer))
        
    best_model, best_acc = None, 0

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (
            args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    logger.debug(f"result_file_name: {result_file_name}")
    
    # loss function
    # loss = loss_func(
    #     results=logits[train_mask],
    #     labels=labels[train_mask],
    #     lamda=0.1,
    #     sigma=1,
    #     model=model,
    #     fs=args.fs,
    #     args=args,
    # )

    # if args.dataset == 'yelp':
    #     loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    # else:
    #     loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')

    train_dur, comm_dur, reduce_dur = [], [], []
    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    # node_dict.pop('train_mask')
    # node_dict.pop('inner_node')
    # node_dict.pop('part_id')
    # node_dict.pop(dgl.NID)

    # if not args.eval:
    #     node_dict.pop('val_mask')
    #     node_dict.pop('test_mask')

    if args.model=="gcn_first":
        if args.sampling_method=="full_graph_sampling":
            tag=1
        elif args.sampling_method=="layer_wise_sampling":
            if args.fs==False:
                tag=2
            else:
                tag=3
        elif args.sampling_method=="layer_importance_sampling":
            if args.pretrain==True:
                tag=8
            else:
                if args.fs==False:
                    tag=4
                else:
                    if args.fs_init_method=="seed":
                        tag=5
                    else:
                        tag=8
        else:
            raise ValueError
    else:
        raise ValueError
                

    writer = get_writer(
        "dist", "gpu", "fs" if args.fs else "no fs",
        f"dataset={args.dataset}",
        f"model={args.model},sampling_method={args.sampling_method},pretrain={args.pretrain},fs={args.fs},fs_init_method={args.fs_init_method}",
        f"tag={tag}",
        f"layer_size={layer_size} lr={args.lr} period={args.log_every}",
        f"partition={args.n_partitions}",
        now_str()
    )

    backward_time = 0

    # worker m 的inner node集合Vm，GNN模型L层
    # for each iteration i:
    # 	一个batch B属于Vm //这些节点相当于是最后一层即第L-1层采样的节点
    # 	previous nodes=batch
    # 	adjs=[] //用于采样后保存每一层的邻接矩阵
    # 	for 层数=[L-1,L-2,...,0]: //从最后一层往前采样
    # 		for each node j in previous nodes:
    # 			if node j是本地inner node:
    # 				直接获取邻接矩阵node的那一行Nj
    #           else:从node i属于的那个worker上传输邻接知阵node i的那行Nj
    #       取Nj的并集，构成一个新的邻接电阵A，行是previous nodes，列是所有previous nodes的邻居节点的并集，即待选节点集合
    #       采样函数 /输入:邻接矩阵A 输出: 该层采样的节点sampled nodes，previous nodes和sampled nodes的邻接矩阵Aprevious nodes=sampled nodes //F始下一层的采样
    #       adis+=[A,] //保存邻接矩阵用于后面的训练
    # input nodes=previous nodes //采样到第0层的节点就是输入节点，需要它们的feature
    # model(featlinput nodes], adis,labels[batch]) //横型训练

    gnn_layer_num = get_gnn_layer_num(layer_size,args)

    # 对于需要经常进行通信的数据，将他们从GPU移动到CPU上

    # 初始时，worker通过node_dict['_ID']只知道inner nodes 和 boundary nodes属于哪个worker，但并不知道其它nodes属于哪个worker
    # 只能通过上一层的邻接矩阵知道该节点A是节点B的邻居，然后在B所属的worker上查找根据节点A查找node_dict['part_id']，从而知道节点A属于哪个worker
    # 或者直接从文件中将所有partition读进来，一次建立所有节点到所属worker的映射
    # 这个操作在主进程中进行，然后将映射传递给各个worker
    # all_partition_detail
    
    # const in sampling
    # node id of inner nodes that belong to the subgraph of the present process
    # global id
    inner_nodes = node_dict['_ID'][node_dict['inner_node']]

    # adjecent matrix of inner nodes and boundary nodes
    # inner nodes as the rows
    # inner nodes and boundary nodes as the columns
    # TODO: matrix部分是对称的
    # DONE: 是对称的
    # torch.unique(torch.eq(inner_boundary_nodes_adj_matrix[:inner_boundary_nodes_adj_matrix.shape[0],:inner_boundary_nodes_adj_matrix.shape[0]],inner_boundary_nodes_adj_matrix[:inner_boundary_nodes_adj_matrix.shape [0],:inner_boundary_nodes_adj_matrix.shape[0]].t()))
    # tensor([True])
    # dense matrix
    # index
    # on gpu, so move it to cpu
    inner_boundary_nodes_adj_matrix = get_adj_matrix_from_graph(graph)
    inner_boundary_nodes_adj_matrix = inner_boundary_nodes_adj_matrix.to(matrix_value_type).to_dense().cpu().t()

    # communicate adj line with other ranks
    # TODO: 测试通信（adj_line, feature）
    # DONE: adj_line ok
    # DONE: feature ok

    # feature中global_id到index的映射
    # gpu
    globalid_index_mapper_in_feature = GlobalidIndexMapper(node_dict['_ID'][node_dict['inner_node']])

    # 最后一层的节点是训练集的，但中间节点不一定是训练集的，所以是整个子图的feature
    # 也不一定属于本地worker，所以需要通信
    swapper = Swapper(feat,inner_boundary_nodes_adj_matrix,args.sample_num, mapper_manager, globalid_index_mapper_in_feature, index_type, matrix_value_type, feat.dtype,all_feat)
    swapper.start_listening()

    for epoch in range(args.n_epochs):
        logger.info(f"epoch: {epoch}")
        epoch_logger = logging.getLogger(f"[{rank},{epoch}]")
        
        # adj of each layer
        adjs = [None for _ in range(gnn_layer_num)]

        # global id
        batch = get_sampled_batch_from_inner_nodes(inner_nodes, node_dict['train_mask'], args) # L-1(output)
        # on gpu, so move it to cpu
        batch = batch.cpu()
        batch_index = mapper_manager[rank].globalid_to_index(batch)
        # global id
        previous_nodes = batch # L-j层的上一层是L-j+1层
        
        # batch节点作为行，batch节点的所有邻居节点作为列
        for layer in range(gnn_layer_num - 1, -1, -1): # L-1, L-2, L-3, ..., 0(input)
            epoch_logger.info(f"epoch: {epoch} layer: {layer}")
            layer_logger = logging.getLogger(f"[{rank},{epoch},{layer}]")
                        
            # tasks 需要进行通信的任务
            swapper_tasks = [torch.tensor([], dtype=index_type) for _ in range(size)]
            adj_line_result = []
            # 将与某个worker的多次通信聚合为一次通信，for循环结束后统一处理
            # global id
            for node in previous_nodes:
                # index
                node_rank = all_partition_detail[node.item()]
                index = mapper_manager[node_rank].globalid_to_index(node.unsqueeze(0))
                # inner node
                if node_rank == rank:
                    # 直接获取本地邻接矩阵[node, :], 记为Nj
                    # index
                    # adj_line只是记录了某个inner node的邻居节点在邻接矩阵中的index
                    # index, shape:[1,x]
                    try:
                        adj_line = inner_boundary_nodes_adj_matrix[index,:]
                    except:
                        layer_logger.exception(f"global id {node}, index {index}, rank {node_rank} {dist.get_rank()}")
                        raise Exception
                    adj_line_result.append({
                        "node":node, # globalid
                        "adj_line":adj_line, # index
                        "rank":node_rank,
                    })
                else:
                    # node j所在worker传输邻接矩阵[node j, :], 记为Nj
                    # the partition id of node j is the rank of the worker which the node j belongs to
                    # index
                    swapper_tasks[node_rank] = torch.cat([swapper_tasks[node_rank], node.unsqueeze(0).to(index_type)])
                    # print("neighbor node in other workers")

            layer_logger.debug(f"swapper_tasks: {swapper_tasks}")

            # collect the adj_line from other workers through communication
            # 如果是异步执行，可以避免同步执行时与某个rank的通信时间过长而阻碍其它通信
            # 但是只有当通信完成时，才能开始后面的工作（具体来说是构建adj_matrix时），因此需要join等待子线程全部完成
            adj_line_thread_pool = ThreadPool(processes=size-1)
            for i in range(size):
                # TODO: adjline通信方案选择
                # 异步，线程池
            #     if swapper_tasks[i].shape[0]:
            #         layer_logger.debug(f"[{rank}] send to [{i}]: {swapper_tasks[i]}")
            #         adj_line_thread_pool.apply_async(swapper.get_adj_line_from_worker, args=(i, swapper_tasks[i]), callback=lambda x:adj_line_result.extend(x))
            # # wait for all subprocesses to finish
            # adj_line_thread_pool.close()
            # adj_line_thread_pool.join()

                # 同步
                if swapper_tasks[i].shape[0]:
                    layer_logger.debug(f"[{rank}] send to [{i}]: {swapper_tasks[i]}")
                    res = swapper.get_adj_line_from_worker(i,swapper_tasks[i])
                    adj_line_result.extend(res)
                    layer_logger.debug(f"adj lines from {i}: {res}")

            layer_logger.debug(f"adj lines {adj_line_result}")

            
            # adj_line 中为1元素的index => global id
            # >>> a=torch.Tensor([1,2,3,4,5,1,2,3])
            # >>> a
            # tensor([1., 2., 3., 4., 5., 1., 2., 3.])
            # >>> torch.where(a == 1)
            # (tensor([0, 5]),)
            # >>> torch.where(a == 1)[0]
            # tensor([0, 5])
            # >>> b=torch.Tensor([11,12,13,14,15,16,17,18])
            # >>> b[torch.where(a == 1)[0]]
            # tensor([11., 16.])
            merged_nodes_global_id = torch.tensor([],dtype=index_type)
            for item in adj_line_result:
                node = item["node"]
                adj_line = item["adj_line"]
                tmp_rank = item["rank"]

                nodes_in_adj_line_global_id = mapper_manager[tmp_rank].index_to_globalid(torch.where(adj_line == 1)[1])
                merged_nodes_global_id = torch.cat([merged_nodes_global_id, nodes_in_adj_line_global_id])
                
            merged_nodes_global_id = torch.unique(merged_nodes_global_id)
            layer_logger.debug(f"merged_nodes_global_id.shape {merged_nodes_global_id.shape}")
            # previous nodes肯定在备选的merged_nodes_global_id里面
            # torch.unique(torch.isin(previous_nodes,merged_nodes_global_id))
            # tensor([True])
            # rank 0, 第L-1层的input中出现boundary nodes
            # torch.unique(torch.isin(merged_nodes_global_id.cuda(),node_dict['_ID'][node_dict['inner_node']]))
            # tensor([False,  True], device='cuda:0')
            
            # merged_nodes_global_id 作为行列
            adj_matrix = torch.zeros((merged_nodes_global_id.shape[0], merged_nodes_global_id.shape[0]),dtype=matrix_value_type)
            merged_nodes_mapper = GlobalidIndexMapper(merged_nodes_global_id)

            for item in adj_line_result:
                node = item["node"] # globalid
                index = merged_nodes_mapper.globalid_to_index(node.unsqueeze(0)) # index
                adj_line = item["adj_line"]
                tmp_rank = item["rank"]

                nodes_in_adj_line_global_id = mapper_manager[tmp_rank].index_to_globalid(torch.where(adj_line == 1)[1])

                adj_matrix[index, merged_nodes_mapper.globalid_to_index(nodes_in_adj_line_global_id)] = 1

            adj_matrix_clone = adj_matrix.clone()
            adj_matrix = adj_matrix_clone + adj_matrix_clone.T
            # 将对角线置为1
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i,i] = 1

            # update previous_nodes from layer-wise sampling function
            # global id => index
            if args.sampling_method == "layer_wise_sampling":
                adj, sampled_nodes = layer_wise_sampling(adj_matrix, merged_nodes_mapper.globalid_to_index(previous_nodes), args.sample_num)
            elif args.sampling_method == "layer_importance_sampling":
                adj, sampled_nodes = layer_importance_sampling(adj_matrix, merged_nodes_mapper.globalid_to_index(previous_nodes), args.sample_num)
            else:
                raise ValueError
            # if merged_nodes_mapper.index_to_globalid(sampled_nodes).shape[0] != 

            layer_logger.debug(
                f"""
                    previous_nodes.shape {previous_nodes.shape}, 
                    torch.unique(previous_nodes).shape {torch.unique(previous_nodes).shape},
                    merged_nodes_mapper.globalid.shape {merged_nodes_mapper.globalid.shape},
                    merged_nodes_mapper.globalid_to_index(previous_nodes).shape {merged_nodes_mapper.globalid_to_index(previous_nodes).shape} ,
                    adj.shape {adj.shape},
                    adj_matrix.shape {adj_matrix.shape}, 
                    sampled_nodes.shape {sampled_nodes.shape}
                """
                )

            # index => global id
            previous_nodes=merged_nodes_mapper.index_to_globalid(sampled_nodes)
            # adj与feature一起运算，放到GPU里面
            adjs[layer] = adj.cuda()

        # glolbal id
        input_nodes = previous_nodes
        # input_nodes的feature可能在别的worker上
        # 通过all_partition_detail(global id=>rank)找到input_nodes所在的worker
        # global id
        # input_nodes 与 input_nodes_rank 在相同index的情况下一一对应
        input_nodes_rank = torch.tensor(all_partition_detail, dtype=matrix_value_type)[input_nodes]

        input_nodes_feat = torch.zeros((input_nodes.shape[0],feat.shape[1]), dtype=feat.dtype,device=feat.device)

        ranks = torch.unique(input_nodes_rank).tolist()
        feature_thread_pool = ThreadPool(processes=size-1)
        res = [] # 存放异步通信的结果
        for tmp_rank in ranks:
            idx = torch.where(input_nodes_rank == tmp_rank)[0]
            if tmp_rank == rank:
                device = globalid_index_mapper_in_feature.globalid.device
                input_nodes_feat[idx.to(device),:] = feat[globalid_index_mapper_in_feature.globalid_to_index(input_nodes[idx].to(device)),:]
            else:
                # 通信
                # input_nodes[idx] global id，访问tmp_rank worker
                # 点对点访问
                layer_logger.debug("get_feature_from_worker")

                # TODO: feature通信方案选择
                # 异步，线程池
                # feature_thread_pool.apply_async(
                #     swapper.get_feature_from_worker, 
                #     args=(tmp_rank, idx, input_nodes[idx].to(index_type)), 
                #     callback=lambda x:res.append(x),
                # )

                # 同步，线程池
                # item = feature_thread_pool.apply(
                #     swapper.get_feature_from_worker, 
                #     args=(tmp_rank, idx, input_nodes[idx].to(index_type)), 
                # )

                # 同步
                # item = swapper.get_feature_from_worker(tmp_rank, idx, input_nodes[idx].to(index_type))

                # 本地获取
                item = swapper.get_feature_from_worker(tmp_rank, idx, input_nodes[idx].to(index_type))
                input_nodes_feat[item["idx"],:] = item["feat"]
                layer_logger.debug(f"input_nodes_feat {item} {item['feat'].shape}")


        # TODO: fix
        # wait for all subprocesses to finish
        # feature_thread_pool.close()
        # feature_thread_pool.join()
        # 处理异步通信结果
        # for item in res:
        #     print(f"[{rank}] epoch:{epoch} layer:{i} idx:{item['idx'].shape} feat:{item['feat'].shape}")
        #     input_nodes_feat[item["idx"],:] = item["feat"]
        
        # OPT3: update the step in optimizer
        optimizer.state['step']["epoch"] = epoch
        
        update_flag = get_update_flag(epoch, args)
            
        t0 = time.time()
        model.train()

        # forward
        if args.model == 'graphsage':
            logits = model(graph, feat, in_deg, update_flag)
            # print(type(graph)) # <class 'dgl.heterograph.DGLGraph'>
            # logits = model(graph, feat)
        elif args.model == 'gcn':
            # input_nodes' feat as input
            # batch label as wanted output
            logits = model(graph,feat[input_nodes,:],adjs,update_flag)
            loss = loss_func(
                results=logits,
                labels=labels[batch_index.cuda()],
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            )
        elif args.model == 'gcn_first':
            for adj in adjs:
                # [408, 512]
                # [56, 408]
                # [16, 56]
                epoch_logger.debug(f"adj.shape {adj.shape}")
            # [512, 500]
            logits = model(input_nodes_feat, adjs)
            
            loss = loss_func(
                results=logits,
                labels=labels[batch_index.cuda()],
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            )
        elif args.model == 'gcn_second':
            pass
        else:
            raise NotImplementedError

        # # loss
        # if args.inductive:
        #     loss = loss_func(
        #         results=logits,
        #         labels=labels,
        #         lamda=args.lamda,
        #         sigma=args.sigma,
        #         model=model,
        #         fs=args.fs,
        #         args=args,
        #     )
        #     # loss = loss_fcn(logits, labels)
        # else:
        #     if args.model == 'gcn':
        #         loss = loss_func(
        #             results=logits,
        #             labels=labels[batch],
        #             lamda=args.lamda,
        #             sigma=args.sigma,
        #             model=model,
        #             fs=args.fs,
        #             args=args,
        #         )
        #     elif args.model == 'gcn_first':
        #         loss = None
        #     elif args.model == 'gcn_second':
        #         loss = None
        #     else:
        #         loss = loss_func(
        #             results=logits[train_mask],
        #             labels=labels[train_mask],
        #             lamda=args.lamda,
        #             sigma=args.sigma,
        #             model=model,
        #             fs=args.fs,
        #             args=args,
        #         )
        #         # loss = loss_fcn(logits[train_mask], labels[train_mask])

        del logits

        optimizer.zero_grad(set_to_none=True)
        # OPT3: calculate the gradients but aggregate the gradients periodically by reduce_hook and optimizer.state['step']
        # reduce_hook will aggregate the gradients of the same parameter on different processes
        torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        end = time.time()
        backward_time += end - start
        # 这里实际上是在等待同步完成
        # 需要更新时才会统计时间
        if update_flag:
            pre_reduce = time.time()
            ctx.reducer.synchronize()
            reduce_time = time.time() - pre_reduce

        ctx.buffer.next_epoch()

        # gradient aggregation ok
        # update the parameters
        optimizer.step()

        # if epoch >= 5 and epoch % args.log_every != 0 :
        if update_flag:
            # 总的训练时间
            train_dur.append(time.time() - t0)
            # 传输embedding的时间
            comm_dur.append(ctx.comm_timer.tot_time())
            # 传输梯度的时间
            reduce_dur.append(reduce_time)

        # if (epoch + 1) % 10 == 0:
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

        ctx.comm_timer.clear()

        del loss

        # find the best model by validation accuracy
        # rank 0 process 使用全局模型在全图上计算acc与loss
        # 模型通过reduce_hook更新后再计算train、test、val的准确率
        # 好处：减少模型的拷贝次数；在全局模型上使用全局图进行计算，而不是在子图、局部模型上计算
        async_test_flag = True
        if rank == 0 and args.eval and get_update_flag(epoch, args):
            # 在一个子线程中测试
            if async_test_flag:
                if thread is not None:
                    model_copy, train_acc, test_acc, val_acc = thread.get()
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model = model_copy

                # OP3: 在model里面加入了一个list[None|torch.Tensor]和一个torch.Tensor作为成员后
                # torch.Tensor默认不支持深度拷贝，所以选择state_dict来拷贝
                # model_copy = copy.deepcopy(model)
                # copy只能在主线程里面进行
                model_copy  = create_model(copy.deepcopy(layer_size),mu, args)
                model_copy.cuda()
                model_copy.load_state_dict(model.state_dict())

                # submit the accuracy and loss calculation task on full graph to subthread
                if not args.inductive:
                    thread = pool.apply_async(
                        evaluate_trans,
                        args=(
                            'Epoch %05d' % epoch,
                            model_copy,
                            # val_g, # full_g
                            full_g,
                            full_g_adj_matrix,
                            layer_size,
                            args,
                            epoch,
                            writer,
                            random_selection_mask,
                            result_file_name
                        )
                    )
                else:
                    thread = pool.apply_async(
                        evaluate_induc,
                        args=(
                            'Epoch %05d' % epoch,
                            model_copy,
                            # val_g, # full_g
                            full_g,
                            'val',
                            result_file_name
                        )
                    )
            # 在主线程中测试
            else:
                _, train_acc, test_acc, val_acc = evaluate_trans(
                    'Epoch %05d' % epoch,
                    model,
                    full_g,
                    full_g_adj_matrix,
                    layer_size,
                    args,
                    epoch,
                    writer,
                    random_selection_mask,
                    result_file_name
                )

    ret = {
        "rank":rank,
    }

    # rank 0 process save the best model
    if args.eval and rank == 0:
        if thread is not None:
            model_copy, train_acc, test_acc, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
            
        os.makedirs('model/', exist_ok=True)
        torch.save(
            best_model.state_dict(),
            'model/' + args.graph_name + '_final.pth.tar'
        )
        print('model saved')
        
        # model为gcn_first
        # pretrain为true，fs必为true（因为pretrain就是在训练fs），表示在pretrain阶段训练fs, 只有在这种情况下model中才有fs, 并且需要导出fs的参数
        # pretrain为flase，fs为true，表示在offline阶段，需要加载已经训练好的fs, 然后处理feature, 但model中没有fs层
        # pretrain为false，fs为false，表示使用gcn_first，但model中没有fs层true
        if args.model == "gcn_first" and args.pretrain==True and args.fs==True:
            torch.save(
                {
                    "fs_layer.weights":model.state_dict()['fs_layer.weights'],
                },
                'model/' + args.graph_name + '_fs_layer_final.pth.tar'
            )
            print(f'model fs layer for {args.dataset} saved')
        
        with open(result_file_name, 'a+') as f:
            buf = str(args)+ "\n" + "Validation accuracy {:.2%}".format(best_acc)
            f.write(buf + '\n')
            print(buf)
        # _, acc = evaluate_induc('Test Result', best_model, test_g, 'test')
        _, acc = evaluate_induc(
            'Test Result', 
            best_model, 
            full_g,
            full_g_adj_matrix,
            layer_size,
            args,
            'test',
            random_selection_mask,
        )
        
        # 所有进程传梯度的通信量
        ret['model_param_grad_communication_volume'] = ctx.reducer.communication_volume
        # 实际执行的轮数
        ret['epoch'] = epoch
        # writer的thread_lock不可序列化，所以不能将writer放在字典中
        # 然后通过队列传给主进程
        ret['writer_path'] = writer.log_dir
        ret['fs_run_time'] = model.fs_run_time
        ret['normal_run_time'] = model.normal_run_time
        print("fs_run_time", model.fs_run_time)
        print("normal_run_time", model.normal_run_time)
        print("backward_time", backward_time)
        writer.close()
        
    # 当前进程传输embedding的通信量
    ret['feature_embedding_communication_volume'] = ctx.buffer.communication_volume
    # 当前进程传输feature的通信量
    ret['feature_communication_volume'] = swapper.feature_communication_volume
    # 将结果从子进程发送给主进程
    queue.put(ret)


def single_run(args):
    matrix_value_type = torch.int32
    logger = logging.getLogger(f"[single]")
    
    # 加载全图
    full_g, n_feat, n_class = load_data(args.dataset)
    full_g = full_g.to(torch.device('cuda'))
    full_g_adj_matrix = get_adj_matrix_from_graph(full_g)
    full_g_adj_matrix = full_g_adj_matrix.to(matrix_value_type).to_dense().t()
    # full_g_adj_matrix = full_g_adj_matrix.to(matrix_value_type)

    feat, labels = full_g.ndata['feat'], full_g.ndata['label']
    train_mask, test_mask, val_mask = full_g.ndata['train_mask'], full_g.ndata['test_mask'], full_g.ndata['val_mask']
    print(train_mask.sum(), test_mask.sum(), val_mask.sum())
    
    # 加载模型
    layer_size = get_layer_size(args)
    logger.info(f"layer_size: {layer_size}")
    gnn_layer_num = get_gnn_layer_num(layer_size,args)
    model = create_model(copy.deepcopy(layer_size), None, args)
    model.cuda()

    # 邻接矩阵
    adjs, previous_indices = sample_full(full_g_adj_matrix, gnn_layer_num, args.sampling_method)

    # writer 
    if args.model=="gcn_first":
        if args.sampling_method=="full_graph_sampling":
            tag=1
        elif args.sampling_method=="layer_wise_sampling":
            if args.fs==False:
                tag=2
            else:
                tag=3
        elif args.sampling_method=="layer_importance_sampling":
            if args.pretrain==True:
                tag=8
            else:
                if args.fs==False:
                    tag=4
                else:
                    if args.fs_init_method=="seed":
                        tag=5
                    else:
                        tag=8
        else:
            raise ValueError
    else:
        raise ValueError

    writer = get_writer(
        "dist", "gpu", "fs" if args.fs else "no fs",
        f"dataset={args.dataset}",
        f"model={args.model},sampling_method={args.sampling_method},pretrain={args.pretrain},fs={args.fs},fs_init_method={args.fs_init_method}",
        f"tag={tag}",
        f"layer_size={layer_size} lr={args.lr} period={args.log_every}",
        f"partition={args.n_partitions}",
        now_str()
    )

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (
            args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    logger.debug(f"result_file_name: {result_file_name}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    for epoch in range(args.n_epochs):
        logits = model(feat, adjs)

        loss = loss_func(
            results=logits[train_mask],
            labels=labels[train_mask],
            lamda=args.lamda,
            sigma=args.sigma,
            model=model,
            fs=args.fs,
            args=args,
        )

        optimizer.zero_grad(set_to_none=True)
        # OPT3: calculate the gradients but aggregate the gradients periodically by reduce_hook and optimizer.state['step']
        # reduce_hook will aggregate the gradients of the same parameter on different processes
        torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        end = time.time()

        optimizer.step()

        train_acc = calc_acc(logits[train_mask], labels[train_mask])
        test_acc = calc_acc(logits[test_mask], labels[test_mask])
        val_acc = calc_acc(logits[val_mask], labels[val_mask])

        train_loss = loss_func(
                    results=logits[train_mask],
                    labels=labels[train_mask],
                    lamda=args.lamda,
                    sigma=args.sigma,
                    model=model,
                    fs=args.fs,
                    args=args,
                ) / train_mask.int().sum().item()
        test_loss = loss_func(
                    results=logits[test_mask],
                    labels=labels[test_mask],
                    lamda=args.lamda,
                    sigma=args.sigma,
                    model=model,
                    fs=args.fs,
                    args=args,
                ) / train_mask.int().sum().item()
        val_loss = loss_func(
                    results=logits[val_mask],
                    labels=labels[val_mask],
                    lamda=args.lamda,
                    sigma=args.sigma,
                    model=model,
                    fs=args.fs,
                    args=args,
                ) / val_mask.int().sum().item()

        print(f"epoch {epoch} train_acc {train_acc} test_acc {test_acc} val_acc {val_acc}")
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        
def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def broadcast_test_acc(test_acc:torch.Tensor):
    dist.broadcast(test_acc, src=0)

def init_processes(rank, size, queue, args, all_partition_detail, mapper_manager,log_id,all_feat=None):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    # Initializes the default distributed process group
    # rank: Rank of the current process
    # world_size : Number of processes participating in the job
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    init_logging(args,log_id,rank)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    # load the partition which has been saved on the disk
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, queue, args, all_partition_detail, mapper_manager, all_feat)
