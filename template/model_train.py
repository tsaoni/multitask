import os, sys
import json
import argparse
import pytorch_lightning as pl
from typing import List
from argparse import Namespace
from pathlib import Path
# from transformers import (
# )

sys.path.append('..')

from model_template_class import (
    ModelTrainTemplate,
)

from utils import (
    get_early_stopping_callback,
    check_argument_setting,
    get_scheduler_info,
    VictimTrainDataset, 
    Seq2SeqLoggingCallback, 
    get_checkpoint_callback, 
)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def parse_args() -> List[Namespace]:
    # todo: delete redundant args 

    parser = argparse.ArgumentParser(description='main program. ')
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
    parser.add_argument('--num_labels', type=int, default=None, help='')

    # path parameters
    parser.add_argument('--task', type=str, default=None, help='task name')
    parser.add_argument('--cache_dir', type=str, default=None, help='')
    parser.add_argument('--output_dir', type=str, default=None, help='')

    # other settings
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
    parser.add_argument('--max_source_length', type=int, default=1024, help='')
    parser.add_argument('--max_target_length', type=int, default=1024, help='')
    parser.add_argument('--val_max_target_length', type=int, default=1024, help='')
    parser.add_argument('--test_max_target_length', type=int, default=1024, help='')
    parser.add_argument('--eval_max_gen_length', type=int, default=1024, help='')
    parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
    parser.add_argument("--sortish_sampler", action="store_true", default=False)
    parser.add_argument('--train_batch_size', type=int, default=10, help='')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='')
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="kwarg passed to DataLoader")
    parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")

    # model parameters
    parser.add_argument('--use_big', action='store_true', help='whether to use large tokenizer. ')

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
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
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
    parser.add_argument('--data_dir', type=str, default=None, help='')

    # task specific arguments
    ## sequence-classification
        
    ## seq2seq
    parser.add_argument("--eval_beams", type=int, default=None, required=False)
    
    ## summarization

    ## translation
    parser.add_argument("--src_lang", type=str, default="", required=False)
    parser.add_argument("--tgt_lang", type=str, default="", required=False)


    args = parser.parse_args()
    if args.use_task_specific_params:
        args.task_specific_params = json.loads(args.task_specific_params)
    else:
        delattr(args, 'task_specific_params')
    args = split_args(args)

    return args

def add_overlap_params(src_dict, tgt_dict, params_list=None):
    # the default list is used for passing to model template.
    overlap_params = ['model_mode', 'output_dir', 'gpus', 'task', 'num_train_epochs', \
        'gradient_accumulation_steps', 'seed', 'do_train'] if params_list is None else params_list
    for param in overlap_params:
        if param in src_dict.keys():
            tgt_dict[param] = src_dict[param]

def split_args(args):
    main_args = [
        'model_mode', 'task', 'output_dir', 'do_train', 'do_eval', 'fp16', 'gpus', 'n_tpu_cores', 
        'logger_name', 'save_top_k', 'fp16_opt_level', 'lowdata_token', 'use_lowdata_token', 'seed',
        'parametrize_emb', 'length_pen', 'use_deep', 'mid_dim', 'early_stopping_patience', 
        'gradient_accumulation_steps', 'num_train_epochs', 
    ]
    model_args = [
        'task_specific_params', 
    ]
    module_args = [
        'model_name_or_path', 'cache_dir', 'config_name', 'tokenizer_name', 'num_labels', 'max_source_length', 
        'sortish_sampler', 'train_batch_size', 'eval_batch_size', 'max_tokens_per_batch', 'val_metric',
        'num_workers', 'n_train', 'n_val', 'n_test', 'use_big', 'adafactor', 'adam_epsilon', 'use_dropout', 
        'max_steps', 'eval_steps', 'warmup_steps', 'learning_rate', 'weight_decay', 
        'dropout', 'label_smoothing', 'max_grad_norm', 'lr_scheduler', 
        'data_dir', 'eval_beams', 'src_lang', 'tgt_lang', 'length_penalty', 
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

def get_logger(args):
    if args.logger_name == "default":
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    return logger


def main(args):
    check_argument_setting(args.main, 'task')
    for data_dir in [Path('../data'), Path('../models')]:
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

    check_argument_setting(args.main, 'model_mode')
    dataset_cls = ( VictimTrainDataset )
    model_tmp = ModelTrainTemplate(args.module, dataset_cls=dataset_cls, **vars(args.model))
    
    # a = model_tmp.validation_step([ x for i, x in enumerate(model_tmp._get_dataloader('train', 10)) if i == 0][0], batch_idx=0)
    
    logger = get_logger(args.main)
    output_dir = model_tmp.output_dir
    val_metric = model_tmp.val_metric_name
    lower_is_better = True

    if args.main.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.main.early_stopping_patience)
    else:
        es_callback = False

    # set callbacks
    # logging_callback=Seq2SeqLoggingCallback()
    # checkpoint_callback=get_checkpoint_callback(output_dir, val_metric, args.main.save_top_k, lower_is_better)
    early_stopping_callback=es_callback

    checkpoint_callback = None
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            output_dir, monitor="val_{}".format(args.module.val_metric), mode="min", save_top_k=1
        )
    
    # checkpoint_callback = OurModelCheckPoint(filepath=args.main.output_dir, prefix="checkpoint", monitor="rouge2", mode="max", save_top_k=-1)

    #if logging_callback is None:
    #    logging_callback = LoggingCallback()

    # train
    train_params = {}
    pl.seed_everything(args.main.seed)
    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.main.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.main.fp16_opt_level
        train_params['amp_backend'] = 'apex'

    #if args.main.gpus > 1:
    #    train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.main.gradient_accumulation_steps
    # train_params['progress_bar_refresh_rate'] = 0

    print('the max number of epochs is {}'.format(args.main.num_train_epochs))
    print('early stopping', early_stopping_callback)
    print('checkpoint_callback', checkpoint_callback)
    # print('logging', logging_callback)

    trainer = pl.Trainer.from_argparse_args(
        args.main,
        max_epochs=args.main.num_train_epochs,
        # weights_summary=None,
        callbacks=[checkpoint_callback],
        logger=logger,
        # checkpoint_callback=checkpoint_callback,
        #early_stop_callback=early_stopping_callback,
        **train_params,
    )

    print('args.do_train:', args.main.do_train)

    if args.main.do_train:
        trainer.fit(model_tmp)

    result = trainer.test(model_tmp)

    import pdb
    pdb.set_trace()

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.main.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    ######## evaluate ############

    # test() without a model tests using the best checkpoint automatically
    trainer.test()

    if args.main.do_eval:
        Path(args.main.output_dir).mkdir(exist_ok=True)
        if len(os.listdir(args.main.output_dir)) > 3 and args.main.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.main.output_dir))

        victim_model = ModelTrainTemplate(model, args.main.model_mode)

        # print(model)
        dataset = Path(args.main.data_dir).name

        with torch.no_grad():
            model.eval()
            print(dataset)
            model = model.cuda()
            print(model.device)
            data_loader = model.test_dataloader()
            out_lst = []
            for batch_idx, batch in enumerate(data_loader):
                # print(batch)
                batch = model.transfer_batch_to_device(batch, model.device)
                # if batch_idx>10:
                #     continue
                # print(batch['input_ids'].device, model.device)
                out = model.test_step(batch, batch_idx)
                out_lst.append(out)
                print(out['preds'])
                # batch = model.transfer_batch_to_device(batch, 'cpu')
            result = model.test_epoch_end(out_lst)

        for k, v in result.items():
            if k != 'preds':
                print(k, v)

        out_1 = args.main.model_name_or_path
        out_path = os.path.join(out_1, 'test_beam_{}'.format(args.main.length_penalty))
        print('writing the test results to ', out_path)
        with open(out_path, 'w') as f:
            for preds in result['preds']:
                print(preds, file=f)

        # print(result)
        for k, v in result.items():
            if k != 'preds':
                print(k, v)
        


if __name__ == '__main__':
    args = parse_args()
    main(args)
