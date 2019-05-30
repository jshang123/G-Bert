python run_pretraining.py --model_name GBert-pretraining --num_train_epochs 10 --do_train --graph
python run_gbert.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-pretraining --num_train_epochs 10 --do_train --graph
python run_pretraining.py --model_name GBert-pretraining --use_pretrain --pretrain_dir ../saved/GBert-predict --num_train_epochs 10 --do_train --graph
python run_gbert.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-pretraining --num_train_epochs 20 --do_train --graph
python run_pretraining.py --model_name GBert-pretraining --use_pretrain --pretrain_dir ../saved/GBert-predict --num_train_epochs 10 --do_train --graph
python run_gbert.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-pretraining --num_train_epochs 30 --do_train --graph
