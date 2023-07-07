use pytorch as the backend



# conda

note:cached packages may be corrupted.
conda clean --all

conda create --name gnn python==3.8.16
conda activate gnn

## cpu
conda install -c dglteam dgl pytorch

## gpu
conda install -c dglteam/label/cu117 dgl
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

note: the version of pytorch and dgl should be compatible with cuda.

conda安装不好用就改成pip

pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

