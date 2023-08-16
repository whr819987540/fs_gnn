import subprocess

fsratio = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

for i in fsratio:
    result = subprocess.run(f"python 2_node_classification.py --model DGLModel --dataset reddit --n_partitions 4 --lr 1e-3 --dropout 0.5 --sigma 1.0 --fsratio {i} --n-layers 3 --n-hidden 256 --port 18119 --log-every 1 --n-epochs 60 --eval --fix_seed --sampling-method neighbor_sampling --fs --fs-init-method seed --batch-size 1024", shell=True, capture_output=True, text=True)
    print(result.stdout)

