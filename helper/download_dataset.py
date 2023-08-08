from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from helper.utils import load_yelp

dst_path = "./dataset/"

data = CoraGraphDataset(raw_dir="./dataset/")
data = PubmedGraphDataset(raw_dir="./dataset/")
data = RedditDataset(raw_dir="./dataset/")
data = DglNodePropPredDataset(name="ogbn-products", root="./dataset/")
# data = DglNodePropPredDataset(name="ogbn-papers100M", root="./dataset/")
data = load_yelp()
