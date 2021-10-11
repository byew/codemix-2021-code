#export CUDA_VISIBLE_DEVICES=0

##data_data  197
##    dmis-lab/biobert-v1.1
#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--model_type xlmroberta \
#--do_train \
#--do_eval \
#--do_eval_during_train \
#--model_name_or_path /media/lab1510/b/tamil/roberta/xlmroberta \
#--data_dir ../../data_2021_2/data_StratifiedKFold_42/data_origin_$i/ \
#--label_directory ../../data_2021_2/label_1.txt \
#--output_dir ../checkpoints/tamil_2021_2/$i \
#--max_seq_length 80 \
#--learning_rate 6e-5 \
#--hidden_dropout_prob 0.3 \
#--adam_epsilon 1e-8 \
#--per_gpu_train_batch_size 4 \
#--gradient_accumulation_steps 4 \
#--per_gpu_eval_batch_size 32 \
#--num_train_epochs 15 \
#--weight_decay 0 \
#--adv_type fgm
#done

##attention
#6e-5 0.4824


















#
#
#for ((i=0;i<5;i++));
#do
#
#python train.py \
#--model_type xlmroberta \
#--do_train \
#--do_eval \
#--do_eval_during_train \
#--model_name_or_path ../tamil_model/ \
#--data_dir ../../data_2021/tamil/data_StratifiedKFold_42/data_origin_$i/ \
#--label_directory ../../data_2020/tamil/label_1.txt \
#--output_dir ../checkpoints/tamil/$i \
#--max_seq_length 60 \
#--learning_rate 6e-5 \
#--hidden_dropout_prob 0.3 \
#--adam_epsilon 1e-8 \
#--per_gpu_train_batch_size 4 \
#--gradient_accumulation_steps 4 \
#--per_gpu_eval_batch_size 32 \
#--num_train_epochs 10 \
#--weight_decay 0 \
#--adv_type fgm
#done

python predict.py \
--model_type xlmroberta \
--vote_model_paths ../checkpoints/tamil_2021_2/ \
--predict_file ../../data_2021_2/test.tsv \
--predict_result_file ../prediction_result/tamil_2021.csv


#xlmroberta   3e-5