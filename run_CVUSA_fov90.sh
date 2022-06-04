#!/usr/bin/env bash
python -u train.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --save_path ./result_fov90 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --fov 90
python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --resume ./result_fov90/checkpoint.pth.tar --save_path ./result_fov90 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop --fov 90