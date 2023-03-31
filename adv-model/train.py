import os, sys
import threading
import json
import argparse
import torch
import pytorch_lightning as pl
from typing import List, Dict, Union
from argparse import Namespace
from pathlib import Path
# from transformers import (
# )

from module import (
    ModelTrainTemplate,
)

from model_cls import (
    MultiTaskWithSharedEmbedding,
)

from utils import (
    get_early_stopping_callback,
    check_argument_setting,
    # get_scheduler_info,
    Seq2SeqLoggingCallback, 
    get_checkpoint_callback, 
    get_file_names, 
    pickle_load, 
)

from dataset import (
    GeneralDataset, 
)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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

def get_callback(module, args, checkpoint_callback_kwargs, is_dict=False):
    output_dir = module.output_dir
    val_metric = module.val_metric_name
    lower_is_better = True

    # early stopping cb
    if args.main.early_stopping_patience >= 0:
        early_stopping_callbacks = [get_early_stopping_callback(val_metric, args.main.early_stopping_patience)]
    else:
        early_stopping_callbacks = []

    # logging_callback=Seq2SeqLoggingCallback()
    # checkpoint_callback=get_checkpoint_callback(output_dir, val_metric, args.main.save_top_k, lower_is_better)

    # checkpoint cb
    checkpoint_callbacks = []
    if len(checkpoint_callbacks) == 0:
        checkpoint_callbacks = [ pl.callbacks.ModelCheckpoint(
            output_dir, monitor="val_{}".format(val_metric), **checkpoint_callback_kwargs,
        ) ]
    # logging cb
    if module.is_seq2seq:
        logging_callbacks = [Seq2SeqLoggingCallback()]
    else:
        logging_callbacks = []

    try:
        print('early stopping', early_stopping_callbacks[0])
        print('checkpoint_callback', checkpoint_callbacks[0])
        print('logging', logging_callbacks[0])
    except (IndexError, ):
        print('warning: there is an empty list in early stopping, checkpoint, logging callback. ')

    if is_dict:
        return {
            'early_stopping': early_stopping_callbacks,
            'checkpoint': checkpoint_callbacks, 
            'logging': logging_callbacks, 
        }
    else:
        return [*early_stopping_callbacks, *checkpoint_callbacks, *logging_callbacks]

    # checkpoint_callback = OurModelCheckPoint(filepath=args.main.output_dir, prefix="checkpoint", monitor="rouge2", mode="max", save_top_k=-1)

    #if logging_callback is None:
    #    logging_callback = LoggingCallback()

def reset_multitask_argument(src_args: argparse.Namespace, tgt_args: argparse.Namespace, task_id: str):
    # src: args.multitask.main, tgt: args.multitask.module/main
    src_dict, tgt_dict = vars(src_args), vars(tgt_args)
    for param_name, param_value in src_dict.items():
        if param_name[-1] == task_id:
            update_param_name = param_name[:-1]
            if update_param_name in tgt_dict.keys():
                print(f'the origin value of parameter {update_param_name} is {tgt_dict[update_param_name]}, '
                                                                                f'update to {param_value}')
                tgt_dict[update_param_name] = param_value

    return argparse.Namespace(**tgt_dict)

def train_settings(args, from_argparse=False):
    train_params = {}
    pl.seed_everything(args.main.seed)
    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.main.fp16:
        train_params["precision"] = 16
        if from_argparse:
            train_params["amp_level"] = args.main.fp16_opt_level
            train_params['amp_backend'] = 'apex'

    #if args.main.gpus > 1:
    #    train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.main.gradient_accumulation_steps
    # train_params['progress_bar_refresh_rate'] = 0

    print('the max number of epochs is {}'.format(args.main.num_train_epochs))

    return train_params

def trainer_kwargs(args: Namespace, logger=None, callbacks=None) -> Dict:
    if args.main.custom_model_name == 'victim':
        kwargs = dict(
            accelerator="gpu",
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            max_steps=100,
            min_steps=100,
            check_val_every_n_epoch=1,
            log_every_n_steps=20,
            gradient_clip_val=args.main.gradient_clip_val,
            limit_train_batches=0.2,
            limit_val_batches=0.1,
        )
    elif args.main.custom_model_name == 'multitask':
        kwargs = dict(
            accelerator="gpu",
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            max_steps=100,
            min_steps=100,
            check_val_every_n_epoch=1,
            log_every_n_steps=20,
            gradient_clip_val=args.main.gradient_clip_val,
            limit_train_batches=0.2,
            limit_val_batches=0.8,
        )
    elif args.main.custom_model_name == 'struct2text':
        kwargs = dict(
            accelerator="gpu",
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            max_epochs=50,
            #max_steps=100,
            #min_steps=100,
            check_val_every_n_epoch=1,
            log_every_n_steps=20,
            gradient_clip_val=args.main.gradient_clip_val,
            limit_train_batches=0.2,
            limit_val_batches=0.8,
        )
    return kwargs

def get_trainer(args, module):
    module_name = module.hparams.custom_model_name
    multitask_id = module.multitask_id
    new_args = vars(args)[module_name]
    logger = get_logger(new_args.main)
    callbacks = get_callback(module, new_args, dict(mode="max", save_top_k=new_args.main.save_top_k))
    module_name = module_name if multitask_id is None else module_name + str(multitask_id)
    return pl.Trainer(
        **trainer_kwargs(
            new_args, 
            logger=logger, 
            callbacks=callbacks, 
        ), 
        **train_settings(new_args),
    )
   
# in the multitask case, pass list of trainers and list of modules, 0: victim mode, 1: text2struct mode
def model_train(
    args: argparse.Namespace, 
    module: Union[List[ModelTrainTemplate], ModelTrainTemplate], 
    multitask_epoch_num=1, 
):
    with torch.cuda.amp.autocast():
        if isinstance(module, ModelTrainTemplate) and module.multitask_id is None: # victim and struct2text
            print(f'start training {module.hparams.custom_model_name} model... ')
            trainer = get_trainer(args, module)
            trainer.fit(module)
        elif isinstance(module, list): # multitask
            for epoch in range(multitask_epoch_num):
                trainer = [get_trainer(args, x) for x in module]
                print(f'Epoch: {epoch}, start training multitask model in the victim mode... ')
                trainer[0].fit(module[0])
                print(f'Epoch: {epoch}, start training multitask model in the text2struct mode... ')
                trainer[1].fit(module[1])
        else:
            raise ValueError('error raises when pass into model_train. ')

    print('finish training! ')

def main(args):
    data_dir_list = ['../data']
    for model_key, params in vars(args).items():
        for param in ['task', 'model_mode']:
            check_argument_setting(params.main, param)
        data_dir_list.append(f'../models/{model_key}')
    for data_dir in data_dir_list:
        Path(data_dir).mkdir(exist_ok=True)

    dataset_cls = ( GeneralDataset )
    # 1: victim, 2: text2struct
    config_list = [] if args.multitask.main.config_name1 is None \
                        else [args.multitask.main.config_name1, args.multitask.main.config_name2]
    task_model_list = [
        [args.multitask.main.model_mode1, args.multitask.main.model_name_or_path1], 
        [args.multitask.main.model_mode2, args.multitask.main.model_name_or_path2] 
    ]
    tokenizer_list = [args.multitask.main.tokenizer_task, args.multitask.module.tokenizer_name]
    multitask_model = MultiTaskWithSharedEmbedding(task_model_list, args.multitask.main, tokenizer_list)

    # set module arguments
    victim_module_args = args.victim.module
    struct2text_module_args = args.struct2text.module
    multitask_module_args_1 = reset_multitask_argument(args.multitask.main, args.multitask.module, '1')
    multitask_module_args_2 = reset_multitask_argument(args.multitask.main, args.multitask.module, '2')
    
    # create modules
    model_dict = multitask_model.get_config_tokenizer_model_dict

    victim_module = ModelTrainTemplate(
        victim_module_args, 
        dataset_cls=dataset_cls, 
        **vars(args.victim.model)
    )
    multitask_module_1 = ModelTrainTemplate(
        multitask_module_args_1, 
        dataset_cls=dataset_cls, 
        multitask_id=1, 
        **model_dict, 
        **vars(args.multitask.model)
    )
    multitask_module_2 = ModelTrainTemplate(
        multitask_module_args_2, 
        dataset_cls=dataset_cls, 
        multitask_id=2, 
        **model_dict, 
        **vars(args.multitask.model)
    )
    struct2text_module = ModelTrainTemplate(
        struct2text_module_args, 
        dataset_cls=dataset_cls, 
        **vars(args.struct2text.model)
    )

    # a = multitask_module_1.validation_step([ x for i, x in enumerate(multitask_module_1._get_dataloader('train', 2)) if i == 0][0], batch_idx=0)
    # b = multitask_module_2.validation_step([ x for i, x in enumerate(multitask_module_2._get_dataloader('train', 2)) if i == 0][0], batch_idx=0)

    # a = model_tmp.validation_step([ x for i, x in enumerate(model_tmp._get_dataloader('train', 10)) if i == 0][0], batch_idx=0)

    # start training
    print('args.victim.do_train:', args.victim.main.do_train)
    print('args.multitask.do_train:', args.multitask.main.do_train)
    print('args.struct2text.do_train:', args.struct2text.main.do_train)

    """
    thread_list = [
        threading.Thread(
            target=model_train, 
            args=(args, victim_module, )
        ), 
        threading.Thread(
            target=model_train, 
            args=(
                args, 
                [ multitask_module_1, multitask_module_2 ], 
            ), 
            kwargs={'multitask_epoch_num': args.multitask.main.num_train_epochs}, 
        ), 
        threading.Thread(
            target=model_train, 
            args=(args, struct2text_module, )
        ), 
    ]
    """
    thread_list = [
        threading.Thread(
            target=model_train, 
            args=(
                args, 
                [ multitask_module_1, multitask_module_2 ], 
            ), 
            kwargs={'multitask_epoch_num': args.multitask.main.num_train_epochs}, 
        ), 
    ]

    for thread in thread_list:
        thread.start()
   
    for thread in thread_list:
        thread.join()


    import pdb
    pdb.set_trace()

    if args.main.do_train:
        trainer.fit(model_tmp)

    result = trainer.test(model_tmp)


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
    args_dict = dict()
    custom_hparams_filename = get_file_names('./hparams')
    for filename in custom_hparams_filename:
        hparams = pickle_load(filename)
        args_dict[hparams.main.custom_model_name] = hparams
    
    args = argparse.Namespace(**args_dict)
    main(args)
