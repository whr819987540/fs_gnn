import os

import scipy
import torch
import dgl
from dgl.data import RedditDataset, PubmedGraphDataset, CoraGraphDataset
from dgl.distributed import partition_graph
import torch.distributed as dist
import time
from contextlib import contextmanager
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from os.path import join
from datetime import datetime


def load_ogb_dataset(name):
    dataset = DglNodePropPredDataset(name=name, root='./dataset/')
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

    print(f'[{rank}] loading partition')

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


def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
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


def get_writer(*tags):
    path = 'logs'
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
