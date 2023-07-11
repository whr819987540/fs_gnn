import torch.nn.functional as F
from module.fs_model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score
from new_layer import feature_importance_gini, loss_func
from single_model import GCN


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, args, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    train_mask, test_mask, val_mask = g.ndata['train_mask'], g.ndata['test_mask'], g.ndata['val_mask']
    # val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    # val_logits, test_logits = logits[val_mask], logits[test_mask]
    # val_labels, test_labels = labels[val_mask], labels[test_mask]
    train_acc = calc_acc(logits[train_mask], labels[train_mask])
    test_acc = calc_acc(logits[test_mask], labels[test_mask])
    # OPT4: 将新计算的test_acc广播给其他进程
    # 此时其它进程处于阻塞状态，直到收到广播的test_acc
    # 因此广播行为需要尽快进行
    print("before broadcast_test_acc")
    broadcast_test_acc(torch.Tensor([test_acc]))
    print("after broadcast_test_acc")
    
    val_acc = calc_acc(logits[val_mask], labels[val_mask])
    buf = "{:s} | Train Accuracy {:.2%} | Test Accuracy {:.2%} | Validation Accuracy {:.2%}".format(name, train_acc, val_acc, test_acc)

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
    buf = buf + " | Train Loss {:.5f} | Test Loss {:.5f} | Validation Loss {:.5f}".format(train_loss, test_loss, val_loss)
    
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
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
    else:
        raise NotImplementedError


def reduce_hook(param, name, args, optimizer:torch.optim.Optimizer):
    def fn(grad):
        # OPT3
        epoch = optimizer.state['step']
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


def run(graph, node_dict, gpb, args):
    #   graph is the subgraph
    #   node_dict:
    #   part_id (int): the partition id of each node, from 0 to n_partitions-1
    #       including the inner nodes and boundary nodes
    #   inner_node: whether the node is inner node
    #   torch.unique((node_dict['part_id']==0)==(node_dict['inner_node']))
    #   tensor([True], device='cuda:0')
    #   label, feat, in_degree: only for inner nodes
    #   train_mask, val_mask, test_mask: only for inner nodes
    rank, size = dist.get_rank(), dist.get_world_size()

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # load the whole graph
    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g

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
    print(f"num_in: {num_in}") # equal to part.num_nodes()
    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges ,{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict)
    print("move_to_cuda")

    # 获取边界节点
    boundary = get_boundary(node_dict, gpb)

    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    # [500, 64, 16, 3]
    # [n_feat, n_hidden ...(n_layers-1), n_class]
    layer_size = [args.n_feat, 64, 16, args.n_class]
    if args.fs:
        # [500, 500, 64, 16, 3]
        # [n_feat, fs(n_feat), n_hidden ...(n_layers-1), n_class]
        layer_size.insert(0,layer_size[0])
    print(f"layer_size: {layer_size}")

    pos = get_pos(node_dict, gpb)
    graph = order_graph(part, graph, gpb, node_dict, pos)
    # store the in_deg for inner nodes
    in_deg = node_dict['in_degree']
    print("order_graph")

    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)
    print("move_train_first")

    # recv_shape有size个分量
    # 对于第i个分量，如果i==rank，那么该分量为None
    # 否则，该分量为第i个进程与当前进程的边界节点数，也就是待传输节点数
    recv_shape = get_recv_shape(node_dict)
    print(f"get_recv_shape {recv_shape}")

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
    print("init_buffer")

    if args.use_pp:
        print(node_dict['feat'].shape)
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)
        print(node_dict['feat'].shape)

    del boundary
    del part
    del pos

    torch.manual_seed(args.seed)

    feat = node_dict['feat']
    labels = node_dict['label']
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()

    # model
    train_x = feat[train_mask]
    train_y = labels[train_mask]
    mu = feature_importance_gini(train_x, train_y)
    model = create_model(layer_size,mu, args)
    # model = GCN(
    #     layer_size=layer_size,
    #     dropout=args.dropout,
    #     sigma=args.sigma,
    #     mu=mu,
    #     fs=args.fs,
    # )
    model.cuda()
    print("model")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # create the same type tensor with model parameters for parameter reducing on cpu
    ctx.reducer.init(model)
    print("reducer init")

    # OPT3
    # reduce_hook在每次训练使用backward计算出梯度后都会被调用
    # 但是目前希望在间隔几个epoch才进行梯度聚合
    # 所以需要将epoch作为参数传进去
    # 由于reduce_hook的参数需要在注册时绑定，所以需要将epoch放到一个类中
    # 可以放到optimizer中
    # tell optimizer the epoch
    optimizer.state['step'] = 0
    # register the hook for gradient reducing on cpu
    # the hook will be called after the local gradient is computed
    for i, (name, param) in enumerate(model.named_parameters()):
        # OPT3: fs参数不需要聚合
        if name.endswith("mu"):
            continue
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

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)

    if not args.eval:
        node_dict.pop('val_mask')
        node_dict.pop('test_mask')

    # for epoch in range(args.n_epochs):
    # 将终止条件更改为>=args.target_acc
    epoch = 0
    exit_flag = False
    while not exit_flag:
        print(f"[{rank}] epoch:{epoch}")

        # OPT3: update the step in optimizer
        optimizer.state['step'] = epoch
        update_flag = get_update_flag(epoch, args)
            
        t0 = time.time()
        model.train()

        # forward
        if args.model == 'graphsage':
            logits = model(graph, feat, in_deg, update_flag)
            # print(type(graph)) # <class 'dgl.heterograph.DGLGraph'>
            # logits = model(graph, feat)
        else:
            raise NotImplementedError

        # loss
        if args.inductive:
            loss = loss_func(
                results=logits,
                labels=labels,
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            )
            # loss = loss_fcn(logits, labels)
        else:
            loss = loss_func(
                results=logits[train_mask],
                labels=labels[train_mask],
                lamda=args.lamda,
                sigma=args.sigma,
                model=model,
                fs=args.fs,
                args=args,
            )
            # loss = loss_fcn(logits[train_mask], labels[train_mask])

        del logits

        optimizer.zero_grad(set_to_none=True)
        # OPT3: calculate the gradients but aggregate the gradients periodically by reduce_hook and optimizer.state['step']
        # reduce_hook will aggregate the gradients of the same parameter on different processes
        loss.backward()
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
        if rank == 0 and args.eval and get_update_flag(epoch, args):
            if thread is not None:
                model_copy, train_acc, test_acc, val_acc = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
                if test_acc >= args.target_acc:
                    exit_flag = True

            # OP3: 在model里面加入了一个list[None|torch.Tensor]和一个torch.Tensor作为成员后
            # torch.Tensor默认不支持深度拷贝，所以选择state_dict来拷贝
            # model_copy = copy.deepcopy(model)
            # copy只能在主线程里面进行
            model_copy  = create_model(layer_size,mu, args)
            model_copy.load_state_dict(model.state_dict())
            
            # submit the validation task to another thread
            if not args.inductive:
                thread = pool.apply_async(
                    evaluate_trans,
                    args=(
                        'Epoch %05d' % epoch,
                        model_copy,
                        val_g, # full_g
                        args,
                        result_file_name
                    )
                )
            else:
                thread = pool.apply_async(
                    evaluate_induc,
                    args=(
                        'Epoch %05d' % epoch,
                        model_copy,
                        val_g, # full_g
                        'val',
                        result_file_name
                    )
                )
        # OP4: 非rank 0 process 接收rank 0传来的test_acc以优雅地退出while循环
        if rank != 0 and args.eval and get_update_flag(epoch, args):
            test_acc = torch.empty(1)
            print(f"[{rank}] wait for test_acc")
            broadcast_test_acc(test_acc)
            test_acc = test_acc[0].item()
            print(f"[{rank}] recv test_acc {test_acc}")
            if test_acc >= args.target_acc:
                exit_flag = True

        epoch += 1

    # rank 0 process save the best model
    if args.eval and rank == 0:
        if thread is not None:
            model_copy, train_acc, test_acc, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
        os.makedirs('model/', exist_ok=True)
        torch.save(best_model.state_dict(), 'model/' + args.graph_name + '_final.pth.tar')
        print('model saved')
        with open(result_file_name, 'a+') as f:
            buf = str(args)+ "\n" + "Validation accuracy {:.2%}".format(best_acc)
            f.write(buf + '\n')
            print(buf)
        _, acc = evaluate_induc('Test Result', best_model, test_g, 'test')
        print(f"model param grad communication volume {ctx.reducer.communication_volume}")

    print(f"[{rank}] feature and embedding communication volume {ctx.buffer.communication_volume}")


def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def broadcast_test_acc(test_acc:torch.Tensor):
    dist.broadcast(test_acc, src=0)

def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    # Initializes the default distributed process group
    # rank: Rank of the current process
    # world_size : Number of processes participating in the job
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    # load the partition which has been saved on the disk
    g, node_dict, gpb = load_partition(args, rank)
    print(f"[{rank}] start running")
    run(g, node_dict, gpb, args)
