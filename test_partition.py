from helper.parser import create_parser
from helper.utils import load_data,graph_partition
from helper.utils import get_graph_save_path,get_graph_config_path
import dgl
from os.path import join

# parse args
# python test_partition.py --dataset pubmed --graph_name pubmedgraph --n_partitions 3 --eval 
args = create_parser()
print(args)


# load data
g, n_feat, n_class = load_data(args.dataset)

# graph partition and save partition
if args.node_rank == 0:
    if args.inductive:
        graph_partition(g.subgraph(g.ndata['train_mask']), args)
    else:
        graph_partition(g, args)
args.n_class = n_class
args.n_feat = n_feat
args.n_train = g.ndata['train_mask'].int().sum().item()

graph_dir = get_graph_save_path(args)
part_config = get_graph_config_path(args,graph_dir)


# load graph partition for certain process(identified by rank)
rank = 0
# DGLGraph – The graph partition structure.
# Dict[str, Tensor] – Node features.
# Dict[str, Tensor] – Edge features.
# GraphPartitionBook – The graph partition information.
# str – The graph name
# List[str] – The node types
# List[str] – The edge types
subg, node_feat, edge_feature, gpb, graph_name, node_type, edge_type = dgl.distributed.load_partition(part_config, rank)
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

