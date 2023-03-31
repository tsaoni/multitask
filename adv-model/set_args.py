import os, sys
import json
import argparse
from typing import List
from argparse import Namespace
from pathlib import Path

from model_cls import (
    MultiTaskWithSharedEmbedding,
)

from utils import (
    get_scheduler_info, 
    pickle_save, 
)

def parse_args() -> List[Namespace]:
    # todo: delete redundant args 

    parser = argparse.ArgumentParser(description=f'get custom arguments... ')
    parser.add_argument('--custom_model_name', type=str, default=None, required=True, help='model name')
    scheduler_dict = get_scheduler_info()

    # model parameters
    parser.add_argument("--model_mode", type=str, default=None, required=False, help="")
    parser.add_argument('--model_name_or_path', type=str, default=None, help='')
    parser.add_argument(
        "--config_name", default=None, type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument('--num_labels', type=int, default=1, help='')

    # path parameters
    parser.add_argument('--task', type=str, default=None, help='task name')
    parser.add_argument('--cache_dir', type=str, default=None, help='')
    parser.add_argument('--output_dir', type=str, default=None, help='')

    # other settings
    parser.add_argument('--device', type=str, default='cpu', help='')
    parser.add_argument('--do_train', action='store_true', help='')
    parser.add_argument('--do_eval', action='store_true', help='')
    parser.add_argument('--fp16', action='store_true', help='')
    parser.add_argument('--gpus', type=int, default=1, help='')
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
    parser.add_argument("--val_metric", type=str, default=None, choices=["bleu", "rouge2", "loss", "accuracy", None])
    parser.add_argument("--save_top_k", type=int, default=1, help="How many checkpoints to save")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    # unknown
    parser.add_argument('--lowdata_token', type=str, default='summarize', help='')
    parser.add_argument('--use_lowdata_token', type=str, default='yes', help='')
    parser.add_argument('--parametrize_emb', type=str, default='MLP', help='')

    
    parser.add_argument('--length_pen', type=float, default=1.0, help='')
    parser.add_argument('--use_deep', type=str, default='no', help='')
    parser.add_argument('--mid_dim', type=int, default=512, help='')

    parser.add_argument('--use_task_specific_params', action='store_true', help='') 
    parser.add_argument(
        '--task_specific_params', 
        type=str, 
        default=None, 
        help='a string of a json object that stores task specific params. '
    )

    # data parameters
    # parser.add_argument('--max_source_length', type=int, default=1024, help='')
    parser.add_argument('--max_source_length', type=int, default=1024, help='')
    parser.add_argument('--max_target_length', type=int, default=1024, help='')
    parser.add_argument('--val_max_target_length', type=int, default=1024, help='')
    parser.add_argument('--test_max_target_length', type=int, default=1024, help='')
    parser.add_argument('--eval_max_gen_length', type=int, default=1024, help='')
    parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
    parser.add_argument("--sortish_sampler", action="store_true", default=False)
    parser.add_argument("--pad_to_max_len", action="store_true", default=True)
    parser.add_argument('--train_batch_size', type=int, default=10, help='')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='')
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="kwarg passed to DataLoader")
    parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")


    # model parameters
    parser.add_argument('--use_big', action='store_true', help='whether to use large tokenizer. ')

    # custom model parameters
    # todo: add task_model_list, tokenizer_list, option: config_list

    # training parameters.
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--use_dropout', type=str, default='no', help='')
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--num_train_epochs', type=int, default=5, help='')
    parser.add_argument('--max_steps', type=int, default=400, help='')
    parser.add_argument('--eval_steps', type=int, default=50, help='')
    parser.add_argument('--warmup_steps', type=int, default=100, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='') 
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        choices=scheduler_dict['arg_to_scheduler_choices'],
        metavar=scheduler_dict['arg_to_scheduler_metavar'],
        type=str,
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        required=False,
        help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
    )

    # path parameters
    # parser.add_argument('--data_dir', type=str, default=None, help='')
    parser.add_argument('--data_dir', type=str, default=None, help='')

    # task specific arguments
    ## sequence-classification
        
    ## seq2seq
    parser.add_argument("--eval_beams", type=int, default=None, required=False)
    
    ## summarization

    ## translation
    parser.add_argument("--src_lang", type=str, default="", required=False)
    parser.add_argument("--tgt_lang", type=str, default="", required=False)

    # todo: condition on only multi-task input
    MultiTaskWithSharedEmbedding.add_model_specific_args(parser)

    args = parser.parse_args()
    if args.use_task_specific_params:
        args.task_specific_params = json.loads(args.task_specific_params)
    else:
        delattr(args, 'task_specific_params')
    args = split_args(args)

    return args

def add_overlap_params(src_dict, tgt_dict, params_list=None):
    # the default list is used for passing to model template.
    overlap_params = ['model_mode', 'output_dir', 'gpus', 'task', 'num_train_epochs', 'cache_dir', 
        'gradient_accumulation_steps', 'seed', 'do_train', 'num_labels', 'custom_model_name', 'device', 
        ] if params_list is None else params_list
    for param in overlap_params:
        if param in src_dict.keys():
            tgt_dict[param] = src_dict[param]

def split_args(args):
    main_args = [ 'model_mode', 'task', 'output_dir', 'do_train', 'do_eval', 'fp16', 'gpus', 'n_tpu_cores', 
        'logger_name', 'save_top_k', 'fp16_opt_level', 'lowdata_token', 'use_lowdata_token', 'seed',
        'parametrize_emb', 'length_pen', 'use_deep', 'mid_dim', 'early_stopping_patience', 'cache_dir', 
        'gradient_accumulation_steps', 'num_train_epochs', 'custom_model_name', 'num_labels', 
        'gradient_clip_val', 'device', ] + \
        [ 'embed_from_pretrained', 'embed_load_from_path', 'embed_path', 'embed_num', 'embed_size', ] + \
        [ 'model_mode1', 'model_mode2', 'model_name_or_path1', 'model_name_or_path2', 
        'config_name1', 'config_name2', 'task1', 'task2', 'val_metric1', 'val_metric2', 
        'max_source_length1', 'max_source_length2', 'data_dir1', 'data_dir2', 'tokenizer_task', ]
    model_args = [
        'task_specific_params', 
    ]
    module_args = [
        'model_name_or_path', 'config_name', 'tokenizer_name', 'max_source_length', 
        'sortish_sampler', 'train_batch_size', 'eval_batch_size', 'max_tokens_per_batch', 'val_metric',
        'num_workers', 'n_train', 'n_val', 'n_test', 'use_big', 'adafactor', 'adam_epsilon', 'use_dropout', 
        'max_steps', 'eval_steps', 'warmup_steps', 'learning_rate', 'weight_decay', 
        'dropout', 'label_smoothing', 'lr_scheduler', 'pad_to_max_len', 
        'eval_beams', 'src_lang', 'tgt_lang', 'length_penalty', 'data_dir', 
        'max_target_length', 'val_max_target_length', 'test_max_target_length', 'eval_max_gen_length', 
    ]
    args_dict = vars(args)
    new_args = argparse.Namespace()
    main_dict, model_dict, module_dict = dict(), dict(), dict()
    for key in args_dict.keys():
        if key in main_args:
            main_dict[key] = args_dict[key]
        elif key in model_args:
            model_dict[key] = args_dict[key]
        elif key in module_args:
            module_dict[key] = args_dict[key]
        else:
            print('{} parameter is not in the lists of arguments, thus unstored. '.format(key))

    add_overlap_params(main_dict, module_dict)
    new_args.main, new_args.model, new_args.module = argparse.Namespace(**main_dict), \
                            argparse.Namespace(**model_dict), argparse.Namespace(**module_dict)

    return new_args

if __name__ == '__main__':
    args = parse_args()
    hparams_path = './hparams'
    Path(hparams_path).mkdir(exist_ok=True)
    hparams_save_path = os.path.join(hparams_path, f"{args.main.custom_model_name}.pkl")
    pickle_save(args, hparams_save_path)