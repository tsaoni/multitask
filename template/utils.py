import argparse
import pickle
import itertools
import linecache
import torch
import logging
import os
import numpy as np
import pytorch_lightning as pl
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Union
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from rouge_score import rouge_scorer, scoring
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from transformers import (
    BartTokenizer,
)

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


try:
    from fairseq.data.data_utils import batch_by_size
    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False

class BaseDataset(Dataset):

    DEFAULT_MAX_SOURCE_LENGTH = 150
    DEFAULT_MAX_TARGET_LENGTH = 150

    def __init__(
        self,
        tokenizer,
        data_dir=None,
        max_source_length=DEFAULT_MAX_SOURCE_LENGTH,
        max_target_length=None,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        task_mode='sequence-classification',
        task_name=None,
    ):
        super().__init__()
        # read file
        check_variable_status(data_dir, name='data_dir', status='None')
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.task_mode = task_mode
        self.task_name = task_name

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")

class VictimTrainDataset(BaseDataset):
    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        if self.task_mode == 'summarization':
            batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                [x["src_texts"] for x in batch],
                # src_lang=self.src_lang,
                tgt_texts=[x["tgt_texts"] for x in batch],
                # tgt_lang=self.tgt_lang,
                max_length=self.max_source_length,
                max_target_length=self.max_target_length,
                return_tensors="pt",
            ).data
        elif self.task_mode == 'sequence-classification':
            source_inputs = [encode_line(self.tokenizer, x["src_texts"], \
                            self.max_source_length, dataset_name=self.task_name) for x in batch]
            target_inputs = [label_to_tensor(x['tgt_texts']) for x in batch]

            source_ids = [x["input_ids"].squeeze() for x in source_inputs]
            src_mask = [x["attention_mask"].squeeze() for x in source_inputs]
            src_token_type_ids = [x["token_type_ids"].squeeze() for x in source_inputs]
            target_ids = target_inputs
                
            input_ids = torch.stack(source_ids)
            masks = torch.stack(src_mask)
            token_type_ids = torch.stack(src_token_type_ids)
            target_ids = torch.stack(target_ids).squeeze().to(torch.long)
            pad_token_id = self.pad_token_id
            source_ids, source_mask, source_token_type_ids = trim_batch(input_ids, token_type_ids, pad_token_id, attention_mask=masks)
            batch_encoding = {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "token_type_ids": source_token_type_ids,
                "labels": target_ids,
            }

        # batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding

class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class Seq2SeqLoggingCallback(pl.Callback):

    logger = logging.getLogger(__name__)

    def on_train_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        self.logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        # Uncommenting this will save val generations
        # return self._write_logs(trainer, pl_module, "valid")

def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        # period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )

def get_scheduler_info():
    # update this and the import above to support new schedulers from transformers.optimization
    arg_to_scheduler = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
        "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
        "polynomial": get_polynomial_decay_schedule_with_warmup,
        # '': get_constant_schedule,             # not supported for now
        # '': get_constant_schedule_with_warmup, # not supported for now
    }
    arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
    arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

    scheduler_dict = dict(
        arg_to_scheduler=arg_to_scheduler,
        arg_to_scheduler_choices=arg_to_scheduler_choices,
        arg_to_scheduler_metavar=arg_to_scheduler_metavar,
    )

    return scheduler_dict

def label_to_tensor(label: str):
    return torch.Tensor([int(label)])

def encode_line(tokenizer, line, max_length, dataset_name=None, pad_to_max_length=True, return_tensors="pt"):
    check_variable_status(dataset_name, name='dataset_name')
    is_nli = check_nli_dataset(dataset_name)
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    separater = '@separater@'
    texts = (sent.strip() for sent in line.split(separater)) if is_nli else (line, )
    return tokenizer(
        [*texts],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def trim_batch(
    input_ids,
    token_type_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return (input_ids[:, keep_column_mask], token_type_ids[:, keep_column_mask])
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], token_type_ids[:, keep_column_mask])

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

def calculate_rouge(output_lns: List[str], reference_lns: List[str], rouge_keys: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def use_task_specific_params(model, task=None):
    """Update config with summarization specific params."""
    if task is not None:
        task_specific_params = model.config.task_specific_params

        if task_specific_params is not None:
            pars = task_specific_params.get(task, {})
            check_task_specific_params_type(pars)
            print(f"using task specific params for {task}: {pars}")
            model.config.update(pars)

def check_task_specific_params_type(pars):
    int_params = ['num_labels']
    float_params = []
    for param in int_params:
        if param in pars.keys():
            pars[param] = int(pars[param])
    for param in float_params:
        if param in pars.keys():
            pars[param] = float(pars[param])

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)

def load_json(path):
    with open(path) as f:
        return json.load(f)

''' check functions '''

def check_argument_setting(args, arg_name):
    if arg_name == 'task':
        assert args.task in ['agnews', 'mrpc', 'news-summary']
    elif arg_name == 'model_mode':
        assert args.model_mode in ['base', 'sequence-classification', 'question-answering', \
            'pretraining', 'token-classification', 'language-modeling', \
            'summarization', 'translation']

def check_variable_status(variable, name="", status='None'):
    if status == 'None':
        if variable is None:
            raise ValueError('{} parameter should not be none. '.format(name))

def check_parameter_value(hparams, param_name_list, check_all=False):
    if isinstance(hparams, argparse.Namespace):
        hparams = vars(hparams)
    # not_none_param = ['model_mode', 'model_name_or_path', 'config_name', 'tokenizer_name']
    check_exist = False
    for param_name in param_name_list:
        if hparams[param_name] is None:
            if check_all:
                raise ValueError('{} parameter should not be none. '.format(param_name))
        else:
            check_exist = True
    if not check_exist:
        raise ValueError('paramters in list should have at least one not none value. ')

def check_nli_dataset(dataset_name):
    nli = []
    if dataset_name in nli:
        return True
    return False