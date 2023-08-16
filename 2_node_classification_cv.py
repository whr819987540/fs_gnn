import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from helper.utils import *
from helper.parser import create_parser
import warnings
import random
from module.gcn_module.first_method_model import DGLModel
from helper.gini import continous_feature_importance_gini
from sklearn.metrics import f1_score
from copy import deepcopy
import psutil

def get_network_usage(interface):
    net_io = psutil.net_io_counters(pernic=True)
    if interface in net_io:
        io = net_io[interface]
        recv_bytes = io.bytes_recv
        sent_bytes = io.bytes_sent
        return recv_bytes, sent_bytes
    else:
        return None

def run(proc_id, devices, args, log_id, graph, train_nids, valid_nids, test_nids):
    init_logging(args, log_id, proc_id)
    logger = logging.getLogger(f"[{proc_id}]")
    logger.info("start")
    torch.manual_seed(args.seed)

    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=args.master_addr, master_port=args.port
    )
    device = torch.device("cpu")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=proc_id,
    )

    # 生成random_selection_mask
    random_selection_mask = None
    # 需要处理feature
    if args.model == 'DGLModel' and args.sampling_method=="neighbor_sampling":
        if args.pretrain == False and args.fs == True:
            new_feat_num = int(args.n_feat*args.fsratio)
            if args.fs_init_method=="seed":
                # 传输中的类型不能是bool
                random_selection_mask = torch.zeros(args.n_feat,dtype=torch.int64)
                # if proc_id==0:
                #     index = torch.randperm(args.n_feat,dtype=torch.int64,device=device)[:new_feat_num]
                #     random_selection_mask[index]=1
                #     dist.broadcast(random_selection_mask,src=0)
                # else:
                #     dist.broadcast(random_selection_mask,src=0)
                index = torch.randperm(args.n_feat,dtype=torch.int64)[:new_feat_num]
                random_selection_mask[index]=1
                shape=graph.ndata["feat"].shape
                graph.ndata["feat"] = graph.ndata["feat"][:,random_selection_mask.bool()]
                logger.info(f"使用random selection, feature shape由{shape}变为{graph.ndata['feat'].shape}")

            elif args.fs_init_method=="random" or args.fs_init_method=="gini":
                fs_weights = torch.load(
                        "model/" + args.graph_name + f"_{args.fs_init_method}_fs_layer_final.pth.tar",
                    )['fs_layer.weights'].cpu()
                shape = graph.ndata["feat"].shape
                print(fs_weights.device,graph.ndata["feat"].device)
                graph.ndata["feat"] = FeatureSeclectOut(args.fsratio, fs_weights, graph.ndata["feat"])
                logger.info(f"offline阶段, feature shape由{shape}变为{graph.ndata['feat'].shape}")
            args.n_feat = new_feat_num
    else:
        raise ValueError
    # 修改graph中的所有feature
    # node_features = select_feature(args,node_features,)

    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    if args.dataset == "ogbn-products":
        fanouts = [5,10,15] # fixed
    elif args.dataset == "ogbn-arxiv":
        fanouts = [4,4,4] 
    elif args.dataset == "reddit":
        fanouts = [15,15,15] # fixed
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    logger.debug(f"sampler {fanouts}")

    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    test_dataloader = dgl.dataloading.DataLoader(
        graph,
        test_nids,
        sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph,
        valid_nids,
        sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    layer_size = [args.n_feat]
    for i in range(args.n_layers - 1):
        layer_size.append(args.n_hidden)
    layer_size.append(args.n_class)
    if args.pretrain == True and args.fs == True:
        layer_size.insert(0, args.n_feat)
    logger.debug(f"layer_size: {layer_size}")

    # model = Model(num_features, 128, num_classes).to(device)
    model = create_model(args, deepcopy(layer_size), device)
    
    # Wrap the model with distributed data parallel module.
    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.model == "DGLModel" and args.sampling_method == "neighbor_sampling":
        if args.fs:
            if args.fs_init_method == "seed":
                tag = 12
            elif args.fs_init_method == "random" or args.fs_init_method == "gini":
                tag = 11
            else:
                raise ValueError
        else:
            tag = 13
    else:
        raise ValueError

    best_accuracy = 0
    best_model_path = "model/" + args.graph_name + "_final.pth.tar"
    best_model = None

    # Copied from previous tutorial with changes highlighted.
    start_recv_bytes, start_sent_bytes = get_network_usage('lo')
    for epoch in range(args.n_epochs):
        model.train()

        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )

                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )
        end_recv_bytes, end_sent_bytes = get_network_usage('lo')
        print(f"recv_bytes: {end_recv_bytes-start_recv_bytes} sent_bytes: {end_sent_bytes-start_sent_bytes}")
        break
        


    if proc_id == 0:
        # save model
        torch.save(best_model.state_dict(), best_model_path)
        logger.info(f"save model to {best_model_path}")
        # save fs laye weights
        if args.pretrain == True and args.fs == True:
            path = "model/" + args.graph_name + f"_{args.fs_init_method}_fs_layer_final.pth.tar"
            torch.save(
                {
                    "fs_layer.weights": best_model.state_dict()["fs_layer.weights"],
                },
                path,
            )
            logger.info(f"save fs layer weights to {path}")

def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


def get_acc_loss(dataloader, model):
    model.eval()
    logits = []
    labels = []
    with tqdm.tqdm(dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata["feat"]
            labels.append(mfgs[-1].dstdata["label"])
            logits.append(model(mfgs, inputs))

        logits = torch.concatenate(logits)
        labels = torch.concatenate(labels)
        loss = F.cross_entropy(logits, labels)
        accuracy = calc_acc(logits, labels)

    return accuracy, loss

    # predictions = []
    # labels = []
    # with tqdm.tqdm(dataloader) as tq, torch.no_grad():
    #     for input_nodes, output_nodes, mfgs in tq:
    #         inputs = mfgs[0].srcdata["feat"]
    #         labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
    #         predictions.append(
    #             model(mfgs, inputs).argmax(1).cpu().numpy()
    #         )
    #     predictions = np.concatenate(predictions)
    #     labels = np.concatenate(labels)
    #     accuracy = sklearn.metrics.accuracy_score(labels, predictions)

    # return accuracy, 0


def create_model(args, layer_size, device):
    if args.model == "DGLModel":
        if args.pretrain == True and args.fs == True and args.fs_init_method == "gini":
            path = join("gini", f"{args.dataset}_gini_impurity.pth")
            weights = torch.load(path)
            print("use gini to init fs layer weights")
        else:
            weights = None
        model = DGLModel(layer_size, args.dropout, weights, args.fs, args.pretrain).to(device)
        return model
    else:
        raise ValueError


def select_feature(args, feat, random_selection_mask=None):
    logger = logging.getLogger(f"[{dist.get_rank()}]")
    if args.model=="DGLModel" and args.sampling=="neighbor_sampling":
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
    else:
        raise ValueError

    return feat


if __name__ == "__main__":
    # parse args
    args = create_parser()

    # init logging
    log_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    init_logging(args, log_id)
    logger = logging.getLogger("main")

    # load dataset
    if args.dataset == "reddit":
        dataset = RedditDataset(raw_dir='./dataset/')
        graph = dataset[0]
        node_labels = graph.ndata['label']
        node_features = graph.ndata["feat"]
        num_features = node_features.shape[1]
        num_classes = (node_labels.max() + 1).item()

        train_nids = graph.ndata['train_mask'].nonzero(as_tuple=True)[0]
        valid_nids = graph.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_nids = graph.ndata['test_mask'].nonzero(as_tuple=True)[0]
    else:
        if args.dataset == "ogbn-products":
            dataset = DglNodePropPredDataset("ogbn-products", root="/root/autodl-fs/")
        else:
            dataset = DglNodePropPredDataset("ogbn-arxiv",root="./dataset/")
        graph, node_labels = dataset[0]
        
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)
        graph.ndata["label"] = node_labels[:, 0]
        
        node_labels = graph.ndata['label']
        node_features = graph.ndata["feat"]
        num_features = node_features.shape[1]
        num_classes = (node_labels.max() + 1).item()

        idx_split = dataset.get_idx_split()
        train_nids = idx_split["train"]
        valid_nids = idx_split["valid"]
        test_nids = idx_split["test"]

    # 在主进程中提前计算gini impurity，子进程直接加载
    os.makedirs("gini/", exist_ok=True)
    if args.pretrain == True and args.fs == True and args.fs_init_method == "gini":
        path = join("gini", f"{args.dataset}_gini_impurity.pth")
        if not os.path.exists(path):
            gini_impurity = continous_feature_importance_gini(
                node_features.cuda(), node_labels.cuda()
            )
            torch.save(gini_impurity, path)
            logger.info(f"save gini impurity to {path}")
        else:
            logger.info(f"gini impurity from {path} already exists")


    # fix the seed
    if args.fix_seed is False:
        if args.parts_per_node < args.n_partitions:
            warnings.warn("Please enable `--fix-seed` for multi-node training.")
        args.seed = random.randint(0, 1 << 31)

    args.n_feat = num_features
    args.n_class = num_classes
    args.n_train = len(train_nids)

    # set the graph name
    if args.graph_name == "":
        if args.inductive:
            args.graph_name = "%s-%d-induc" % (args.dataset, args.n_partitions)
        else:
            args.graph_name = "%s-%d-trans" % (args.dataset, args.n_partitions)
    logger.info(args)

    graph.create_formats_()

    mp.spawn(
        run,
        args=(
            list(range(args.n_partitions)),
            args,
            log_id,
            graph,
            train_nids,
            valid_nids,
            test_nids,
        ),
        nprocs=args.n_partitions,
    )
