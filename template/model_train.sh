NUM_LABELS=2
TASK_SPECIFIC_PARAMS=$( jq -n \
                            --arg num_labels "$NUM_LABELS" \
                        '{
                            mrpc: {
                                num_labels: $num_labels, 
                            }
                        }' )
# --task_specific_params "$TASK_SPECIFIC_PARAMS" \

# seq2seq task
python model_train.py \
 --model_mode summarization \
 --model_name_or_path facebook/bart-large \
 --task news-summary \
 --do_train \
 --do_eval \
 --fp16 \
 --gpus 1 \
 --logger_name default \
 --val_metric rouge2 \
 --save_top_k 1 \
 --data_dir news-summary \
 --eval_beams 3 \
 --max_source_length 10 \
 --max_target_length 10 \
 --val_max_target_length 10 \
 --test_max_target_length 10 \

# sentiment task

#python model_train.py \
# --model_mode sequence-classification \
# --model_name_or_path bert-base-cased \
# --num_labels 2 \
# --task mrpc \
# --do_eval \
# --fp16 \
# --gpus 1 \
# --logger_name default \
# --val_metric accuracy \
# --save_top_k 1 \
# --data_dir glue/mrpc \
