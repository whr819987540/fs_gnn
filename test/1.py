# 载入OGB的Graph Property Prediction数据集
import dgl
import torch
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.dataloading import GraphDataLoader

def _collate_fn(batch):
    # 小批次是一个元组(graph, label)列表
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels

# 载入数据集
dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
split_idx = dataset.get_idx_split()
# dataloader
train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)