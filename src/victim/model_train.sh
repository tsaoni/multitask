NUM_LABELS=2
TASK_SPECIFIC_PARAMS=$( jq -n \
                            --arg num_labels "$NUM_LABELS" \
                        '{
                            mrpc: {
                                num_labels: $num_labels, 
                            }
                        }' )

python model_train.py \
 --model_mode sequence-classification \
 --model_name_or_path bert-base-cased \
 --num_labels 2 \
 --task mrpc \
 --do_train \
 --do_eval \
 --fp16 \
 --gpus 1 \
 --logger_name default \
 --val_metric accuracy \
 --save_top_k 1 \
 --task_specific_params "$TASK_SPECIFIC_PARAMS" \
 --data_dir glue/mrpc \