python dist_gcn_main.py \
--model gcn_first \
--dataset pubmed \
--n_partitions 2 \
--lr 1e-3 \
--dropout 0.1 \
--sigma 1.0 \
--fsratio 0.5 \
--n-layers 4 \
--n-hidden 256 \
--port 18119 \
--log-every 1 \
--n-epochs 100 \
--eval \
--fix_seed \
--fs \
--pretrain





