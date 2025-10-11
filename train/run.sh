#!/bin/bash


python train.py --train_dir ./train_set.csv --valid_dir ./valid_set.csv --result_dir ./result-training.csv --checkpoint_path ./best_model.pt --batch-size 6 --lr 1e-5 --weight_decay 5e-3 --epochs 100 --patience 5

python test.py --checkpoint_path ./best_model.pt --test_dir ./test_set.csv --result_dir ./result-test.csv 

