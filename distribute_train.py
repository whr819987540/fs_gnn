# 单机上使用cora数据集训练fs-gnn

from helper.utils import get_layer_size
from PipeGCN.distribute_model import GraphSAGE
import torch.nn.functional as F
import torch.nn as nn
import torch
import dgl.data
import dgl
import os
from new_layer import feature_importance_gini, loss_func
os.environ["DGLBACKEND"] = "pytorch"


# load cora dataset
dataset = dgl.data.CoraGraphDataset()
print(f"Number of categories: {dataset.num_classes}")

g = dataset[0]

# define the model
layer_size = get_layer_size(g.ndata['feat'].shape[1], 64, dataset.num_classes, 3)

train_mask = g.ndata["train_mask"]
train_x = g.ndata['feat'][train_mask]
train_y = g.ndata['label'][train_mask]
mu = feature_importance_gini(train_x, train_y)
model = GraphSAGE(
    layer_size=layer_size,
    activation=F.relu,
    use_pp=False,
    sigma=1,
    mu=mu,
)

# train on cpu


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(100):
        # forward
        logits = model(g, features) # in distributed gnn, in_deg can't be none
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss = loss_func(
            results=logits[train_mask],
            labels=labels[train_mask],
            lamda=0.1,
            sigma=1,
            model=model,
            args={"dataset": "cora"},
        )

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )


train(g, model)
