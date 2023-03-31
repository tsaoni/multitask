NUM_LABELS=2
TASK_SPECIFIC_PARAMS=$( jq -n \
                            --arg num_labels "$NUM_LABELS" \
                        '{
                            mrpc: {
                                num_labels: $num_labels, 
                            }
                        }' )
# --task_specific_params "$TASK_SPECIFIC_PARAMS" \

# get arguments
# victim
python set_args.py \
 --custom_model_name victim \
 --model_mode sequence-classification \
 --model_name_or_path bert-base-cased \
 --num_labels 2 \
 --task mrpc \
 --do_eval \
 --gpus 1 \
 --logger_name default \
 --val_metric accuracy \
 --save_top_k 1 \
 --data_dir glue/mrpc \
 --device cuda:0 \

# struct2text
python set_args.py \
 --custom_model_name struct2text \
 --model_mode summarization \
 --model_name_or_path facebook/bart-large \
 --task wiki80 \
 --do_train \
 --do_eval \
 --fp16 \
 --gpus 1 \
 --logger_name default \
 --val_metric rouge2 \
 --save_top_k 1 \
 --data_dir wiki80 \
 --eval_beams 3 \
 --max_source_length 10 \
 --max_target_length 10 \
 --val_max_target_length 10 \
 --test_max_target_length 10 \
 --device cuda:0 \

# multi-task
python set_args.py \
 --custom_model_name multitask \
 --model_mode summarization \
 --model_name_or_path facebook/bart-large \
 --task wiki80 \
 --do_train \
 --do_eval \
 --gpus 1 \
 --logger_name default \
 --val_metric rouge2 \
 --save_top_k 1 \
 --data_dir wiki80 \
 --eval_beams 3 \
 --max_source_length 10 \
 --max_target_length 10 \
 --val_max_target_length 10 \
 --test_max_target_length 10 \
 --eval_max_gen_length 10 \
 --model_mode1 sequence-classification \
 --model_mode2 summarization \
 --model_name_or_path1 bert-base-cased \
 --model_name_or_path2 facebook/bart-large \
 --tokenizer_task summarization \
 --tokenizer_name facebook/bart-large \
 --task1 mrpc \
 --task2 wiki80 \
 --val_metric1 accuracy \
 --val_metric2 rouge2 \
 --max_source_length1 10 \
 --max_source_length2 10 \
 --data_dir1 glue/mrpc \
 --data_dir2 wiki80 \
 --embed_size 768 \
 --device cuda:0 \
 --num_labels 2 \

# main program
python train.py 