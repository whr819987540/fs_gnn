# 单机上使用cora数据集训练fs-gnn
from sklearn.metrics import f1_score
from helper.utils import get_layer_size, get_writer, now_str
import torch
import dgl.data
import dgl
from new_layer import feature_importance_gini, loss_func
from single_model import GCN
from d2l import torch as d2l
from matplotlib import pyplot as plt
import os
os.environ["DGLBACKEND"] = "pytorch"


fs = True
gpu = True
lr = 1e-4  # 0.001都很难收敛，尤其是没有fs
epoch = 10000  # 训练终止条件1
target_test_acc = 0.6  # 训练终止条件2

# device
if gpu:
    device = d2l.try_gpu()
else:
    device = torch.device('cpu')

# load cora dataset
# dataset = dgl.data.CoraGraphDataset()  # 特征很稀疏
# dataset = dgl.data.CitationGraphDataset("citeseer") # 特征很稀疏
# dataset = dgl.data.PPIDataset() # label是一个向量而不简单只是一个标量
dataset = dgl.data.PubmedGraphDataset()
print(dataset.num_labels)
# print(f"Number of categories: {dataset.num_labels}")

g = dataset[0]
g = g.to(device)


# define the model
# layer_size = get_layer_size(g.ndata['feat'].shape[1], 64, dataset.num_labels, 2)
layer_size = [g.ndata['feat'].shape[1], 64, 16, dataset.num_labels]
print(layer_size, lr)  # 500 64 3


train_mask = g.ndata["train_mask"]
train_x = g.ndata['feat'][train_mask]
train_y = g.ndata['label'][train_mask]
mu = feature_importance_gini(train_x, train_y)

model = GCN(
    layer_size=layer_size,
    dropout=0.1,
    sigma=1,
    mu=mu,
    fs=fs,
)
model.to(device)

# general acc calculate function


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')

# train on cpu


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    val_mask = g.ndata["val_mask"]

    # animator = d2l.Animator("epoch","loss/acc")
    writer = get_writer("logs", "single", "cpu", "fs" if fs else "no fs",
                        f"dataset={dataset.name}", f"layer={layer_size}" f"lr={lr}", now_str())
    for e in range(epoch):
        # Forward
        # logits = model(features, g.adjacency_matrix().to_dense())
        # print(type(g)) # <class 'dgl.heterograph.DGLGraph'>
        logits = model(g, features)

        # # Compute prediction
        # pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        train_loss = loss_func(
            results=logits[train_mask],
            labels=labels[train_mask],
            lamda=0.1,
            sigma=1,
            model=model,
            fs=fs,
            args=None,
        )
        test_loss = torch.nn.CrossEntropyLoss(reduction='sum')(logits[test_mask], labels[test_mask])

        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Compute accuracy on training/validation/test
        # train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        train_acc = calc_acc(logits[train_mask], labels[train_mask])
        test_acc = calc_acc(logits[test_mask], labels[test_mask])
        val_acc = calc_acc(logits[val_mask], labels[val_mask])
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # print
        if e % 5 == 0:
            print(
                f"In epoch {e}, train_loss: {train_loss:.3f},\
                test loss: {test_loss:.3f},\
                train acc: {train_acc:.3f},\
                test acc: {test_acc:.3f} (best {best_test_acc:.3f}),\
                val acc: {val_acc:.3f} (best {best_val_acc:.3f})"
            )

        # log
        writer.add_scalar("train loss", train_loss, e)
        writer.add_scalar("train acc", train_acc, e)
        writer.add_scalar("test loss", test_loss, e)
        writer.add_scalar("test acc", test_acc, e)
        writer.add_scalar("val acc", val_acc, e)

        # draw
        # epoch - (loss, train_acc, test_acc, val_acc)
        # with torch.no_grad():
        #     animator.add(e,[loss, train_acc, test_acc, val_acc])

        # exit
        if test_acc >= target_test_acc:
            break


train(g, model)

print("transfer_volume", model.transfer_volume)
if fs:
    print("model.feature_num, model.feature_zero, model.feature_fs_zero",
          model.feature_num, model.feature_zero, model.feature_fs_zero)
    print((model.feature_fs_zero-model.feature_zero)*1.0/model.feature_num)

print("normal_layer_output_zero_num", model.normal_layer_output_zero_num)
