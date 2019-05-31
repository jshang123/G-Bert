#! /bin/bash
python run_pretraining.py --model_name GBert-pretraining-q --num_train_epochs 5 --do_train --graph
python run_gbert.py --model_name GBert-predict-q --use_pretrain --pretrain_dir ../saved/GBert-pretraining-q --num_train_epochs 5 --do_train --graph

for i in {1..15}
do
python run_pretraining.py --model_name GBert-pretraining-q --use_pretrain --pretrain_dir ../saved/GBert-predict-q --num_train_epochs 5 --do_train --graph
python run_gbert.py --model_name GBert-predict-q --use_pretrain --pretrain_dir ../saved/GBert-pretraining-q --num_train_epochs 5 --do_train --graph
done


