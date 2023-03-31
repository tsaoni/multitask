import argparse
import torch
import numpy as np
import torch.nn as nn
from typing import Callable, Dict, Iterable, List, Union
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertTokenizer,
    BertModel,
    GPT2Tokenizer, 
    GPT2Model, 
    BartTokenizer, 
    BartModel, 
    AutoConfig, 
    BertForSequenceClassification, 
    BartForConditionalGeneration, 
)


from utils import (
    check_variable_status,
)
from module import (
    MODEL_MODES,
    BaseModel,
)

MODEL_DICT = {
    'base': {
        'auto': AutoModel,
    },
    'sequence-classification': {
        'bert-base-uncased': BertForSequenceClassification,
    },
    'summarization': {
        'gpt2': GPT2Model, 
        'facebook/bart-large': BartForConditionalGeneration, 
    },
}

TOKENIZER_DICT = {
    'base': {
        'auto': AutoTokenizer,
    },
    'sequence-classification': {
        'bert-base-uncased': BertTokenizer,
    },
    'summarization': {
        'gpt2': GPT2Tokenizer,
        'facebook/bart-large': BartTokenizer, 
    },
}
"""
EMBED_FUNC_DICT = {
    'base': {
        'auto': None,
    },
    'sequence-classification': {
        'bert-base-uncased': lambda model: model.embeddings,
    },
    'summarization': {
        'gpt2': lambda model: model.wte,
        'facebook/bart-large': lambda model: model.get_input_embeddings(),  
    },
}
"""
class MultiTaskWithSharedEmbedding(BaseModel):
    def __init__(
        self, 
        task_model_list: List[List[str]], 
        src_args: argparse.Namespace,
        tokenizer_list: List[str],
        config_list=[], 
    ):
        super().__init__()
        self._get_config_args(src_args)
        
        check_variable_status(task_model_list, name="task_model_list")
        check_variable_status(tokenizer_list, name="tokenizer_list")
        assert len(task_model_list) == len(config_list) or len(config_list) == 0

        self.task_num = len(task_model_list)
        self.configs, self.models = [], []
        config_list = task_model_list if len(config_list) == 0 else config_list
        tokenizer_task, tokenizer_name = tokenizer_list

        tokenizer_type = AutoTokenizer
        if tokenizer_name in TOKENIZER_DICT[tokenizer_task].keys():
            tokenizer_type = TOKENIZER_DICT[tokenizer_task][tokenizer_name]
            
        self.tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            cache_dir=self.config_args.cache_dir,
        )

        # set embedding
        if self.config_args.embed_from_pretrained:
            # assume that the model type in tokenizer is surely in model dict ...
            model_type = MODEL_DICT[tokenizer_task][tokenizer_name]
            tmp = model_type.from_pretrained(tokenizer_name)
            embeddings = tmp.get_input_embeddings()
            hidden_size = tmp.config.hidden_size
            del tmp
        else:
            if self.config_args.embed_load_from_path:
                # todo: load pretrained embed
                embeddings = None
            else: # random init
                embeddings = nn.Embedding(self.tokenizer.vocab_size, self.config_args.embed_size, \
                            device='cuda:0' if torch.cuda.is_available() else 'cpu')

                hidden_size = self.config_args.embed_size

        # default assume that tasks in task_model_list are matched to those in config_list ...
        config_list = np.array(config_list)[:, 1]
        for [task, model_name_or_path], config_name in zip(task_model_list, config_list):                                                    
            config = AutoConfig.from_pretrained(
                config_name,
                num_labels=self.config_args.num_labels,
                cache_dir=self.config_args.cache_dir,
                # **config_kwargs,
            )
            config.hidden_size = hidden_size

            model_type = MODEL_DICT[task][model_name_or_path] \
                    if model_name_or_path in MODEL_DICT[task].keys() else MODEL_MODES[task]
            model = model_type.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                cache_dir=self.config_args.cache_dir,
                ignore_mismatched_sizes=True, 
            )
            model.to(self.config_args.device)
            model.set_input_embeddings(embeddings)

            self.configs.append(config)
            self.models.append(model)
        self.models = nn.ModuleList(self.models)

        # set config
        self.config = dict()
        for i, (config_name, config) in enumerate(zip(config_list, self.configs)):
            config.model_id = i + 1
            self.config[config_name] = config

    def forward(self, input_ids, task_id, **kwargs): 
        # input_embeds = self.embeddings(input_ids)
        # self._freeze_parameters(task_id)
        return self.models[task_id - 1](input_ids=input_ids, **kwargs)

    def freeze_parameters(self, optimizer, task_id):
        # setting requires_grad
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        for param in self.models[task_id - 1].parameters():
            param.requires_grad = True
        # update lr and weight decay
        try:
            for name, param in zip(optimizer.param_groups_name, optimizer.param_groups):
                for attr in ['lr', 'weight_decay']:
                    optimizer.params_state[name][attr] = param[attr]
        except AttributeError:
            print('the states of parameters are not stored. '
            'may raise assertion error if the parameters in optimizer are not fully pass to forward path. ')
        # set optimizer param group
        update_param_list = filter(lambda tp: tp[-1].requires_grad, self.named_parameters())
        optimizer.param_groups, optimizer.param_groups_name = [], []
        for (n, p) in update_param_list:
            optimizer.param_groups_name.append(n)
            optimizer.param_groups.append({
                'params': p, 
                'lr': optimizer.params_state[n]['lr'], 
                'weight_decay': optimizer.params_state[n]['weight_decay'],
            })

    @property
    def get_config_tokenizer_model_dict(self):
        return dict(
            config=self.config,
            tokenizer=self.tokenizer,
            model=self, 
        )    

    @staticmethod
    def add_model_specific_args(parser):
        # embedding
        parser.add_argument('--embed_from_pretrained', action="store_true", default=False)
        parser.add_argument('--embed_load_from_path', action="store_true", default=False)
        parser.add_argument('--embed_path', type=str, default='./embed', help='the default embedding path is ./embed. ')
        parser.add_argument('--embed_num', type=int, default=50257, help='the default size in gpt2. ')
        parser.add_argument('--embed_size', type=int, default=768, help='the default size in gpt2. ')

        # multi-task settings, 1: sentiment (same as victim), 2: text to struct
        parser.add_argument("--model_mode1", type=str, default=None, required=False, help="")
        parser.add_argument("--model_mode2", type=str, default=None, required=False, help="")
        parser.add_argument('--model_name_or_path1', type=str, default=None, help='')
        parser.add_argument('--model_name_or_path2', type=str, default=None, help='')
        parser.add_argument(
            "--config_name1", default=None, type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--config_name2", default=None, type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument("--tokenizer_task", type=str, default=None, required=False, help="")
        parser.add_argument('--task1', type=str, default=None, help='task name')
        parser.add_argument('--task2', type=str, default=None, help='task name')
        parser.add_argument("--val_metric1", type=str, default=None, choices=["bleu", "rouge2", "loss", "accuracy", None])
        parser.add_argument("--val_metric2", type=str, default=None, choices=["bleu", "rouge2", "loss", "accuracy", None])
        parser.add_argument('--max_source_length1', type=int, default=1024, help='')
        parser.add_argument('--max_source_length2', type=int, default=1024, help='')
        parser.add_argument('--data_dir1', type=str, default=None, help='')
        parser.add_argument('--data_dir2', type=str, default=None, help='')

    def _get_config_args(self, src_args: argparse.Namespace) -> argparse.Namespace:
        params_list = ['num_labels', 'cache_dir', 'embed_from_pretrained', 'embed_load_from_path', 
                        'embed_path', 'embed_num', 'embed_size', 'device', ]
        src_dict, tgt_dict = vars(src_args), {}
        for param in params_list:
            if param in src_dict.keys():
                tgt_dict[param] = src_dict[param]
            else:
                raise ValueError(f'the parameter {param} is required in the multi-task model but can not '
                                    'be retrieved by the src_args. ')

        self.config_args = argparse.Namespace(**tgt_dict)

"""
# Define your input sequence
input_text = "Hello, how are you today?"

# Tokenize and embed the input sequence
input_ids = tokenizer.encode(input_text, add_special_tokens=False)
input_embedding = model.transformer.wte(torch.tensor(input_ids).unsqueeze(0))

# Create an InputFeatures object
input_features = InputFeatures(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, input_embeddings=input_embedding)

# Pass the input features to the model
output = model(inputs_embeds=input_features.input_embeddings)
"""

class MultiTaskWithSharedEncoder(nn.Module):
    def __init__(self, ):
        pass

    def forward(self, ):
        pass



class AdapterForTask():
    pass # todo: how to use adapter to share the same parameters ?
