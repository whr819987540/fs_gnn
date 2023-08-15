import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from os.path import join
from threading import Thread
from typing import List

import dgl
import numpy as np
import scipy
import torch
import torch.distributed as dist
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset
from dgl.distributed import partition_graph
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from helper import sampler
from module.gcn_module.first_method_model import FeatureSeclectOut


def load_ogb_dataset(name):
    # dataset = DglNodePropPredDataset(name=name, root='./dataset/')
    dataset = DglNodePropPredDataset(name=name, root='/root/autodl-fs/')
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g


def load_yelp():
    prefix = './dataset/yelp/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    label = list(class_map.values())
    node_data['label'] = torch.tensor(label)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(
        torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g


def load_data(dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir='./dataset/')
        g = data[0]
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    elif dataset == 'yelp':
        g = load_yelp()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset(raw_dir="./dataset/")
        g = data[0]
    elif dataset == "cora":
        data = CoraGraphDataset(raw_dir='./dataset/')
        g = data[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class


def load_partition(args, rank):
    # node_dict
    # _ID: node id of inner nodes and boundary nodes in the whole graph
    #       all subgraphs use the same namespace
    # part_id: the partition that an inner node or a boundary node belongs to, range from 0 to n_partitions-1
    # inner_node: whether a node is an inner node, True for inner node, False for boundary node

    # Process 1 has 7763 nodes, 36969 edges ,6591 inner nodes, and 31833 inner edges.
    # '_ID':
    # tensor([ 6746,  6747,  6748,  ..., 16110,  1499, 13539], device='cuda:0')
    # node_dict['_ID'].shape
    # torch.Size([7763])
    
    # 'part_id':
    # tensor([1, 1, 1,  ..., 2, 0, 2], device='cuda:0')
    # node_dict['part_id'].shape
    # torch.Size([7763])
    # set(node_dict['part_id'].tolist())
    # {0, 1, 2}
    
    # 'inner_node':
    # tensor([ True,  True,  True,  ..., False, False, False], device='cuda:0')
    # node_dict['inner_node'].shape
    # torch.Size([7763])
    
    # 'label':
    # tensor([1, 1, 1,  ..., 1, 0, 1], device='cuda:0')
    # node_dict['label'].shape
    # torch.Size([6591])
    
    # 'feat':
    # tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #         ...,
    #         [0.0000, 0.0164, 0.0067,  ..., 0.0000, 0.0000, 0.0000],
    #         [0.1651, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.0361, 0.0000,  ..., 0.0000, 0.0000, 0.1022]],
    #        device='cuda:0')
    # node_dict['feat'].shape
    # torch.Size([6591, 500])
    
    # 'in_degree':
    # tensor([ 2, 22,  2,  ..., 31,  2,  2], device='cuda:0')
    # node_dict['in_degree'].shape
    # torch.Size([6591])

    # 'train_mask':
    # tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    # 'val_mask':
    # tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    # 'test_mask':
    # tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    # node_dict['train_mask'].shape
    # torch.Size([6591])

    logger = logging.getLogger(f'[{rank}]')
    logger.info(f'loading partition')

    # graph_dir = 'partitions/' + args.graph_name + '/'
    # part_config = graph_dir + args.graph_name + '.json'
    graph_dir = get_graph_save_path(args)
    part_config = get_graph_config_path(args, graph_dir)

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_degree'] = node_feat[node_type + '/in_degree']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_degree')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb


def get_graph_save_path(args):
    graph_dir = join("partitions", args.dataset, args.graph_name)
    return graph_dir


def get_graph_config_path(args, graph_dir):
    return join(graph_dir, args.graph_name+".json")


def graph_partition(g, args):
    graph_dir = get_graph_save_path(args)
    part_config = get_graph_config_path(args, graph_dir)

    # TODO: after being saved, a bool tensor becomes a uint8 tensor (including 'inner_node')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_degree'] = g.in_degrees()
            partition_graph(g, args.graph_name, args.n_partitions, graph_dir,
                            part_method=args.partition_method, balance_edges=False, objtype=args.partition_obj)


def get_layer_size(args):
    layer_size = [args.n_feat]
    layer_size.extend([args.n_hidden] * (args.n_layers - 1))
    layer_size.append(args.n_class)
    # [n_feat, n_hidden ...(n_layers-1), n_class]
    # [500, 256, 256, 256, 3] n_layers=4

    # 当model为gcn_first、采样方式为layer_importance_sampling时，只有在pretrain为true，fs为true时，才会有fs层真正出现在model中，即layer_size中
    if args.model == "gcn_first" and args.sampling_method=="layer_importance_sampling" and args.pretrain==True and args.fs==True:
        layer_size.insert(0, layer_size[0])
    # if model == "gcn_first":
    #     if pretrain==True and fs==True:
    #         layer_size.insert(0, layer_size[0])
    # else:
    #     if fs==True:
    #         # [n_feat, n_feat, n_hidden ...(n_layers-1), n_class]
    #         # [500, 500, 256, 256, 256, 3] n_layers=4
    #         layer_size.insert(0, layer_size[0])

    return layer_size


def get_boundary(node_dict, gpb):
    # 获取边界节点
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary


def data_transfer(data, recv_shape, backend, dtype=torch.float, tag=0):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        if backend == 'gloo':
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype)
        else:
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype, device='cuda')

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        if backend == 'gloo':
            req = dist.isend(data[right].cpu(), dst=right, tag=tag)
        else:
            req = dist.isend(data[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda()
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    torch.cuda.synchronize()
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))


@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))


def get_writer(*tags,root="./"):
    path = root
    tags = list(tags)
    tags.insert(0,"logs")
    for tag in tags:
        path = join(path, tag)
    writer = SummaryWriter(path)
    return writer


def now_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def matrix_to_sparse_matrix(matrix:torch.Tensor)->torch.Tensor:
    return matrix.to_sparse()


def sparse_matrix_transfer_bytes(matrix:torch.Tensor)->int:
    return matrix.indices().numel() * matrix.indices().element_size() + matrix.values().numel() * matrix.values().element_size()


def get_sampled_batch_from_inner_nodes(inner_nodes, train_mask, args):
    # batch首先属于inner node，然后还属于训练集    
    batch_loader = sampler.DataLoader(
        dataset = sampler.MyDataset(inner_nodes[train_mask]),
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    return next(iter(batch_loader))

def get_adj_matrix_from_graph(g):
    """
        get adjecent matrix(sparse) from graph, including inner nodes and boundary nodes
    """
    # <class 'dgl.sparse.sparse_matrix.SparseMatrix'>
    adj = g.adjacency_matrix() 
    # adj_tensor.shape
    # torch.Size([7763, 7763])
    adj_tensor = torch.sparse_coo_tensor(
            indices=adj.indices(),
            values=adj.val,
            size=adj.shape
    )
    # 将sparse matrix转化为最紧凑的形式
    return adj_tensor.coalesce()


def get_all_partition_detail_and_globalid_index_mapper_manager(num_nodes:int, size:int, args):
    """
        1) get partition id or rank by global_id as the list's index
        2) get globalid by index of the matrix on certain rank, or get index of the matrix on certain rank by globalid 
    """
    # 建立node globalid到partition id的映射
    all_partition_detail = torch.zeros(num_nodes,dtype=torch.int64) # same type as node_dict['_ID'] and node_dict['part_id']
    # 建立globalid到index的映射
    mapper_manager = GlobalidIndexMapperManager()

    for rank in range(size):
        g, node_dict, gpb = load_partition(args, rank)
        mapper = GlobalidIndexMapper(node_dict['_ID'])
        mapper_manager.add_mapper(mapper, rank)
        # node_dict['_ID']
        # tensor([    0,     1,     2,  ..., 12856, 18956, 11570])
        # mapper.globalid_to_index(torch.tensor([12856,0]))
        # tensor([10931,     0])
        # node_dict['_ID'].shape
        # torch.Size([10934])
    
        # for i,j in zip(node_dict['_ID'], node_dict['part_id']):
        #     all_partition_detail[i.item()] = j.item()
        all_partition_detail[node_dict['_ID']] = node_dict['part_id']

    return all_partition_detail, mapper_manager


class GlobalidIndexMapper:
    """
        node_dict['_ID'] is the global id of nodes in the whole graph
        index in the adjecent matrix is only the index
        this class is used to map both of them
    """
    def __init__(self,globalid:torch.Tensor):
        self.globalid = globalid
        
    def globalid_to_index(self,globalid:torch.Tensor)->torch.Tensor:
        # torch.tensor([9534,1275])[:,None]
        # torch([[9534],
            # [1275]])    
        # torch.where(torch.tensor([9534,1275])[:,None]==mapper.globalid)
        # (tensor([0, 1]), tensor([10560, 10561]))

        # TODO: 大数据量下无法运行
        # 首先将globalid转化为n*1的向量，self.globalid是1*n的向量
        # torch.eq的结果是n*n的矩阵，每一行有且仅有一个True
        # torch.where就是获取非零值的行索引与列索引
        # 但是torch.where在INT_MAX<n*n时无法运行
        # return torch.where(torch.eq(globalid[:,None],self.globalid))[1]
        
        # DONE: 排序后二分查找
        sa,ia=torch.sort(globalid)
        sb,ib=torch.sort(self.globalid)
        indices = torch.searchsorted(sb, sa)
        # >>> A = torch.tensor([2, 4, 5])
        # >>> B = torch.tensor([5, 2, 4, 1])
        # >>> torch.sort(A)
        # torch.return_types.sort(
        # values=tensor([2, 4, 5]),
        # indices=tensor([0, 1, 2]))
        # >>> torch.sort(B)
        # torch.return_types.sort(
        # values=tensor([1, 2, 4, 5]),
        # indices=tensor([3, 1, 2, 0]))
        # >>> sa,ia=torch.sort(A)
        # >>> sb,ib=torch.sort(B)
        # >>> indices = torch.searchsorted(sb, sa)
        # >>> indices
        # tensor([1, 2, 3])
        # >>> ib[indices]
        # tensor([1, 2, 0])
        # >>> B[ib[indices]]==A
        # tensor([True, True, True])
        return ib[indices]


    def index_to_globalid(self,index:torch.Tensor)->torch.Tensor:
        return self.globalid[index]

class GlobalidIndexMapperManager:
    """
        管理不同worker上globalid与邻接矩阵 index的映射
    """
    def __init__(self) -> None:
        self.manager = dict()

    def add_mapper(self,mapper:GlobalidIndexMapper,rank:int):
        self.manager[rank] = mapper

    def __getitem__(self,rank:int)->GlobalidIndexMapper:
        return self.manager[rank]

class Swapper:
    # 需要传输的内容：
    # 1）初始化的模型参数
    # 2）每轮训练的梯度
    # 3）feature、adj_line
    RequesetTag = 1
    ResponseTag = 2
    AdjLineTag = 1
    FeatureTag = 2
    
    def __init__(self, features, adj_matrix, sample_num,mapper_manager:GlobalidIndexMapperManager,globalid_index_mapper_in_feature:GlobalidIndexMapper,index_type:torch.dtype,matrix_value_type:torch.dtype,feature_value_type:torch.dtype,all_feat:torch.Tensor=None) -> None:
        self.rank = dist.get_rank()
        self.sample_num = sample_num # 某层采样节点数的最大值

        self.logger = logging.getLogger(f'[{self.rank}]')

        self.adj_matrix = adj_matrix
        self.features = features
        self.all_feat = all_feat # 弃用
        
        self.mapper_manager = mapper_manager
        self.globalid_index_mapper = mapper_manager[self.rank]
        # 用于将globalid转换为feature矩阵中的index
        self.globalid_index_mapper_in_feature = globalid_index_mapper_in_feature

        self.index_type = index_type
        self.matrix_value_type = matrix_value_type
        self.feature_value_type = feature_value_type

        self.feature_communication_volume = 0
        
    def start_listening(self):
        # TODO: 让listener线程优雅地退出
        # 若在主线程中创建了子线程，当主线程结束时根据子线程daemon（设置thread.setDaemon(True)）属性值的不同可能会发生下面的两种情况之一：
        # 如果某个子线程的daemon属性为False，主线程结束时会检测该子线程是否结束，如果该子线程还在运行，则主线程会等待它完成后再退出；
        # 如果某个子线程的daemon属性为True，主线程运行结束时不对这个子线程进行检查而直接退出，同时所有daemon值为True的子线程将随主线程一起结束，而不论是否运行完成。
        # 属性daemon的值默认为False，如果需要修改，必须在调用start()方法启动线程之前进行设置。
        self.adjline_listener_thread = Thread(target=self.adjline_listener)
        self.adjline_listener_thread.setDaemon(True)
        self.adjline_listener_thread.start()

        self.feature_listener_thread = Thread(target=self.feature_listener)
        self.feature_listener_thread.setDaemon(True)
        self.feature_listener_thread.start()

        # t = Thread(target=self.listener)
        # t.start()

    def listener(self):
        while True:
            # global id(no -1 in it)
            nodes = torch.tensor([-1]*(self.sample_num+1),dtype=self.index_type)
            # recv from all ranks
            try:
                work = dist.recv(nodes)
            except Exception as e:
                self.logger.error(f"listener recv exception {e}",stack_info=True)
                
            tag = nodes[0].item()
            nodes = nodes[1:]
            nodes = nodes[nodes!=-1]

            self.logger.debug(f"tag {tag}")
            # adj line
            if tag == self.AdjLineTag:
                self.logger.debug(f"recv adj line nodes, {nodes} {nodes.shape}")
                try:
                    # global id
                    # index
                    nodes = self.globalid_index_mapper.globalid_to_index(nodes)
                    # send adj line to certain rank
                    dist.send(self.adj_matrix.index_select(0,nodes).to_dense(),dst=work,tag=self.AdjLineTag)
                    self.logger.debug(f"send adj lines, {nodes} {nodes.shape}")
                except Exception as e:
                    self.logger.error(f"listener send adjline exception {e}",stack_info=True)
            # feature
            else:
                self.logger.debug(f"recv feature nodes, {nodes} {nodes.shape}")
                try:
                    # index
                    nodes_index = self.globalid_index_mapper_in_feature.globalid_to_index(nodes.cuda())
                    # send features to certain rank
                    dist.send(self.features[nodes_index,:].cpu(),dst=work,tag=self.FeatureTag)
                    self.logger.debug(f"feature send, nodes_index {nodes_index} {nodes_index.shape}")
                except Exception as e:
                    self.logger.error(f"listener send feature exception {e}",stack_info=True)
                
    def adjline_listener(self):
        # 负责接收请求并放入请求队列中
        while True:
            # global id(no -1 in it)
            # cpu
            nodes = torch.tensor(
                data=[-1]*(self.sample_num+1),
                dtype=self.index_type
            )
            
            # recv from all ranks
            try:
                work = dist.recv(
                    nodes,
                    tag=int(f"{self.RequesetTag}{self.AdjLineTag}")
                )
            except Exception as e:
                self.logger.error(f"adjline_listener recv request exception {e}",stack_info=True)
            else:
                tag = nodes[0].item()
                # global id
                nodes = nodes[1:]
                nodes = nodes[nodes!=-1]

                self.logger.debug(f"tag {tag}")
                self.logger.debug(f"adjline_listener recv request {nodes} {nodes.shape}")

            try:
                # index
                nodes = self.globalid_index_mapper.globalid_to_index(nodes.cuda())
                # send adj line to certain rank
                start = time.time()
                data = self.adj_matrix.index_select(0,nodes.cuda()).to_dense()
                end = time.time()
                self.logger.debug(f"sparse to dense time: {end-start}")
                dist.send(
                    data.cpu(),
                    dst=work,
                    tag=int(f"{self.ResponseTag}{self.AdjLineTag}")
                )
            except Exception as e:
                self.logger.error(f"adjline_listener send response exception {e}",stack_info=True)
            else:
                self.logger.debug(f"adjline_listener send response {self.adj_matrix.index_select(0,nodes).shape} ")
            

    def get_adj_line_from_worker(self, rank, nodes:torch.Tensor):
        """
            向指定的worker寻找邻接矩阵中nodes所在行.

            由于邻接矩阵取决于子图, 因此邻接矩阵是固定不变的.
            由于通信是将CPU的数据转移到内核, 因此需要将邻接矩阵从GPU转移到CPU, 避免频繁的GPU-CPU通信.         
            采用点对点通信

            这个函数只是交换固定的数据, 访问数据时数据一定是ready的, 因此没必要做异步.
            用了多线程也为了提高网络IO效率.
            
            返回:
            adj_line
            rank: worker id
            node: List[int] global id
        """
        # 接收方需要知道nodes的数量
        tmp = torch.cat([
            torch.tensor(
                data=[int(f"{self.RequesetTag}{self.AdjLineTag}")],
                dtype=nodes.dtype,
                device=nodes.device
            ),
            nodes,
        ])
        
        try:
            dist.send(
                tmp.cpu(),
                dst=rank, 
                tag=int(f"{self.RequesetTag}{self.AdjLineTag}")
            )
        except Exception as e:
            self.logger.exception(f"get_adj_line_from_worker send request exception {e}, {nodes} {nodes.shape}",stack_info=True)
        else:
            self.logger.debug(f"get_adj_line_from_worker send request, {nodes} {nodes.shape}")

        # 接收方需要知道adj_line的形状，行是nodes的个数，列是rank中inner_nodes和boundary_nodes的个数，即globalid的个数
        try:
            adj_lines = torch.zeros(
                size=(len(nodes), self.mapper_manager[rank].globalid.shape[0]), 
                dtype=self.matrix_value_type
            )
            dist.recv(
                adj_lines, 
                src=rank, 
                tag=int(f"{self.ResponseTag}{self.AdjLineTag}")
            )
        except Exception as e:
            self.logger.error(f"get_adj_line_from_worker recv response exception {e}",stack_info=True)
        else:
            self.logger.debug(f"get_adj_line_from_worker recv response {adj_lines.shape} ")

        # tmp = []
        # for i in range(len(nodes)):
        #     tmp.append({
        #         "node":nodes[i],
        #         # dense matrix
        #         # "adj_line":adj_lines[i,:].unsqueeze(0),
        #         # dense matrix to sparse matrix
        #         "adj_line":dense_matrix_to_sparse_matrix(adj_lines[i,:].unsqueeze(0)),
        #         # dense matrix to sparse matrix
        #         # "adj_line":adj_lines.index_select(0,torch.LongTensor([i])).coalesce(),
        #         "rank":rank,
        #     })
        tmp = {
            "nodes":nodes,
            "adj_lines":dense_matrix_to_sparse_matrix(adj_lines),
            "rank":rank,
        }
        return tmp

    def feature_listener(self):
        # 负责接收请求并放入请求队列中
        while True:
            # global id(no -1 in it)
            # cpu
            nodes = torch.tensor(
                [-1]*(self.sample_num+1),
                dtype=self.index_type
            )
            
            # recv from all ranks
            try:
                work = dist.recv(
                    nodes,
                    tag=int(f"{self.RequesetTag}{self.FeatureTag}")
                )
            except Exception as e:
                self.logger.error(f"feature_listener recv request exception {e}",stack_info=True)
            else:
                tag = nodes[0].item()
                # global id
                nodes = nodes[1:]
                nodes = nodes[nodes!=-1]
                
                self.logger.debug(f"tag {tag}")
                self.logger.debug(f"feature_listener recv request {nodes} {nodes.shape}")

                self.feature_communication_volume += get_tensor_bytes_size(nodes)
                
            try:
                # index
                nodes = self.globalid_index_mapper_in_feature.globalid_to_index(nodes.cuda())
                # send features to certain rank
                data = self.features[nodes,:]
                dist.send(
                    data.cpu(),
                    dst=work,
                    tag=int(f"{self.ResponseTag}{self.FeatureTag}")
                )
            except Exception as e:
                self.logger.error(f"feature_listener send response exception {e}",stack_info=True)
            else:
                self.logger.debug(f"feature_listener send response {data.shape} ")

                self.feature_communication_volume += get_tensor_bytes_size(data)

    def get_feature_from_worker(self, rank, idx:torch.Tensor, nodes:torch.Tensor):
        """
            向指定的worker获取nodes的feature

            由于邻接矩阵取决于子图, 因此邻接矩阵是固定不变的.
            由于通信是将CPU的数据转移到内核, 因此需要将邻接矩阵从GPU转移到CPU, 避免频繁的GPU-CPU通信.         
            采用点对点通信

            这个函数只是交换固定的数据, 访问数据时数据一定是ready的, 因此没必要做异步.
            用了多线程也为了提高网络IO效率.
            
            返回:
            adj_line
            rank: worker id
            node: List[int] global id
        """
        # FIXME: 用本地获取的全图的all_feature来替代rank之间的通信
        # DONE: feature通信bug已修复
        if self.all_feat is not None:
            return {
                "feat":self.all_feat[nodes,:],
                "idx":idx,
            }

        # 接收方需要知道nodes的数量
        tmp = torch.cat([
            torch.tensor(
                data=[int(f"{self.RequesetTag}{self.FeatureTag}")],
                dtype=nodes.dtype,
                device=nodes.device
            ),
            nodes,
        ])

        try:
            # 接收方需要知道nodes的数量
            # self.logger.debug(f"before {self.rank} get feature from {rank}, {tmp}")
            # dist.send(nodes.cpu(), dst=rank,tag=self.FeatureTag)
            dist.send(
                tmp.cpu(), 
                dst=rank, 
                tag=int(f"{self.RequesetTag}{self.FeatureTag}")
            )
        except Exception as e:
            self.logger.exception(f"get_feature_from_worker send request exception {e}",stack_info=True)
        else:
            self.logger.debug(f"get_feature_from_worker send request, {nodes} {nodes.shape}")

            self.feature_communication_volume += get_tensor_bytes_size(tmp)
            
        try:
            # 接收方需要知道adj_line的形状，行是nodes的个数，列是rank中inner_nodes和boundary_nodes的个数，即globalid的个数
            features = torch.zeros(
                size=(len(nodes), self.features.shape[1]),
                dtype=self.feature_value_type
            )
            dist.recv(features, src=rank,tag=int(f"{self.ResponseTag}{self.FeatureTag}"))
        except Exception as e:
            self.logger.error(f"get_feature_from_worker recv response exception {e}",stack_info=True)
        else:
            self.logger.debug(f"get_feature_from_worker recv response {features} {features.shape} {features.dtype}")

            self.feature_communication_volume += get_tensor_bytes_size(features)

        return {
            "idx":idx,
            "feat":features.cuda(),
        }


def init_logging(args,log_id:str,rank=-1):
    # 2023-08-02 19:23:44 [0] "/home/whr/fs_gnn/dist_gcn_train.py", line 844, DEBUG: torch.Size([512, 500])
    # asctime: 时间戳，可指定格式
    # name: logger的名字
    # pathname: 调用日志输出函数的模块的完整路径名，可能没有
    # lineno: 调用日志输出函数的语句所在的代码行(所在文件内的行号)
    # levelname: 日志的最终等级(以字符串的形式显示)
    # message: 用户输出的消息(传入logger的参数)
    log_format = f'%(asctime)s %(name)s "%(pathname)s", line %(lineno)d, %(levelname)s: %(message)s\n'
    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    os.makedirs('./results',exist_ok=True)
    logging.basicConfig(
        level=(choices.index(args.log_level)+1)*10, format=log_format, 
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f'results/{log_id}_{rank}.log'
    )


def select_feature(args, feat, random_selection_mask=None):
    # model为gcn_first
    # pretrain为true，fs必为true（因为pretrain就是在训练fs），表示在pretrain阶段训练fs, 只有在这种情况下model中才有fs, 并且需要导出fs的参数
    # pretrain为flase，fs为true，表示在offline阶段，需要加载已经训练好的fs, 然后处理feature, 但model中没有fs层
    # pretrain为false，fs为false，表示使用gcn_first，但model中没有fs层true
    logger = logging.getLogger(f"[{dist.get_rank()}]")
    if args.model=="gcn_first":
        if args.sampling_method=="layer_importance_sampling":
            if args.pretrain==False and args.fs==True:
                if args.fs_init_method=="seed":
                    if random_selection_mask is None:
                        raise ValueError
                    shape=feat.shape
                    feat = feat[:,random_selection_mask]
                    logger.info(f"使用random selection, feature shape由{shape}变为{feat.shape}")
                else:
                    fs_weights = torch.load(
                        join('model/', args.graph_name + '_fs_layer_final.pth.tar')
                    )['fs_layer.weights'].cuda()
                    shape = feat.shape
                    feat = FeatureSeclectOut(args.fsratio, fs_weights, feat)
                    logger.info(f"offline阶段, feature shape由{shape}变为{feat.shape}")
        elif args.sampling_method=="layer_wise_sampling":
            if args.pretrain==False and args.fs==True:
                if args.fs_init_method=="seed":
                    if random_selection_mask is None:
                        raise ValueError
                    shape=feat.shape
                    feat = feat[:,random_selection_mask]
                    logger.info(f"使用random selection, feature shape由{shape}变为{feat.shape}")

    return feat


def get_tensor_bytes_size(tensor:torch.Tensor)->int:
    return tensor.numel() * tensor.element_size()

def get_gnn_layer_num(layer_size:List[int],args)->int:
    # 最后一个线性层不属于GNN
    gnn_layer_num = len(layer_size) - 1 - 1

    # model为gcn_first,采样方式为layer_importance_sampling
    # pretrain为true，fs必为true（因为pretrain就是在训练fs），表示在pretrain阶段训练fs, 只有在这种情况下model中才有fs, 并且需要导出fs的参数
    # pretrain为flase，fs为true，表示在offline阶段，需要加载已经训练好的fs, 然后处理feature, 但model中没有fs层
    # pretrain为false，fs为false，表示使用gcn_first，但model中没有fs层true

    # 只计算GNN的层数，不包括FS层
    # 当model为gcn_first、采样方式为layer_importance_sampling时，只有在pretrain为true，fs为true时，才会有fs层真正出现在model中，即layer_size中
    if args.model == "gcn_first" and args.sampling_method=="layer_importance_sampling" and args.pretrain==True and args.fs==True:
        gnn_layer_num -= 1

    return gnn_layer_num

def dense_matrix_to_sparse_matrix(dense:torch.Tensor):
    non_zero_idx = dense.nonzero()
    rows = non_zero_idx[:, 0].tolist()
    cols = non_zero_idx[:, 1].tolist()
    values = dense[rows, cols]

    sparse = torch.sparse.IntTensor(
        indices=torch.LongTensor([rows, cols]),
        values=values,
    )

    return sparse.coalesce()