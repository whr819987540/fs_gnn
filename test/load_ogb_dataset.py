from helper import utils
import os

os.environ["DGLBACKEND"] = "pytorch"
g, n_feat, n_class = utils.load_data("ogbn-products")