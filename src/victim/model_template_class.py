import abc
import os, sys
import argparse
import torch
import evaluate
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import (
    BartForConditionalGeneration,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.models.bart.modeling_bart import shift_tokens_right # for seq2seq model

sys.path.append('..')

from utils import (
    use_task_specific_params, 
    pickle_save, 
    lmap,
    label_smoothed_nll_loss,
    flatten_list,
    calculate_rouge,
    check_parameter_value,
    VictimTrainDataset,
    get_scheduler_info,
)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


class ModelTrainTemplate(pl.LightningModule):

    DEFAULT_MODEL_MODE = "sequence-classification"
    DEFAULT_VAL_METRIC = "accuracy"
    ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]

    def __init__(
        self,
        hparams: argparse.Namespace, 
        dataset_cls=None,
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        super().__init__()

        # setting path and save parameters
        self.save_hyperparameters(hparams)
        
        if self.hparams.output_dir is None:
            output_dir_name = self.hparams.task + '_' + 'tb={}_'.format(self.hparams.train_batch_size) + \
                        'e={}_'.format(self.hparams.num_train_epochs) + 'd={}_'.format(self.hparams.dropout) + \
                        'l={}_'.format(self.hparams.label_smoothing) + 'lr={}_'.format(self.hparams.learning_rate) \
                        + 'w={}_'.format(self.hparams.weight_decay) + 's={}'.format(self.hparams.seed)
            self.output_dir = os.path.join('../models', output_dir_name)
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = os.path.join('../models', self.hparams.output_dir)

        if len(os.listdir(self.output_dir)) > 3 and self.hparams.do_train:
            print('Output directory ({}) already exists and is not empty, overwrite to it...'.format(self.output_dir))

        self.hparams_save_path = os.path.join(self.output_dir, "hparams.pkl")
        pickle_save(self.hparams, self.hparams_save_path)
        self.logging_dir = os.path.join(self.output_dir, 'log')

        # set dataset parameters
        self._check_sampler_usage()
        self.data_dir = os.path.join('../data', self.hparams.data_dir)
        self.dataset_kwargs: dict = dict(
            data_dir=self.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.target_lens = None
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        self.num_workers = self.hparams.num_workers
        self.train_batch_size = self.hparams.gradient_accumulation_steps * self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.gradient_accumulation_steps * self.hparams.eval_batch_size
        self.dataset_class = ( dataset_cls )
        
        # set training parameters
        self.step_count = 0
        self.config, self.tokenizer, self.model = self._get_model_from_argparse_args(**config_kwargs)
        self.vocab_size = self.config.vocab_size
        self._initialize_metric()
        self._task_specific_parameter_setting()
    

    ''' initialize functions '''

    def _check_sampler_usage(self):
        if self.hparams.sortish_sampler and self.hparams.gpus > 1:
            pass
            # self.hparams.replace_sampler_ddp = False
        elif self.hparams.max_tokens_per_batch is not None:
            if self.hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if self.hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

    def _initialize_metric(self):
        self.loss_names = ["loss"]
        self.metrics_save_path = os.path.join(self.output_dir, "metrics.json")
        if self.hparams.model_mode == 'summarization':
            self.metric_names = ModelTrainTemplate.ROUGE_KEYS
        elif self.hparams.model_mode == 'sequence-classification':
            self.metric_names = ['accuracy', ]
            if self.hparams.task is not None:
                self.metric = evaluate.load("glue", self.hparams.task)
            else:
                self.metric = evaluate.load("accuracy")
        self.metrics = defaultdict(list)
        self.val_metric_name = ModelTrainTemplate.DEFAULT_VAL_METRIC if self.hparams.val_metric is None else self.hparams.val_metric
        self.log_val_metric = 'accuracy'
        self.training_loss_across_batches_at_curr_epoch = []

    def _get_model_from_argparse_args(
        self,
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        check_parameter_value(self.hparams, ['model_mode'])
        if config is None:
            check_parameter_value(self.hparams, ['model_name_or_path', 'config_name'])
            config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": self.hparams.num_labels} if self.hparams.num_labels is not None else {}),
                cache_dir=self.hparams.cache_dir,
                **config_kwargs,
            )

        if tokenizer is None:
            check_parameter_value(self.hparams, ['model_name_or_path', 'tokenizer_name'])
            tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=self.hparams.cache_dir,
            )

        model_type = MODEL_MODES[self.hparams.model_mode]
        if model is None:
            check_parameter_value(self.hparams, ['model_name_or_path'])
            model = model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=config,
                cache_dir=self.hparams.cache_dir,
            )

        use_task_specific_params(model, self.hparams.task)

        return config, tokenizer, model

    def _task_specific_parameter_setting(self):
        seq2seq_models = ["summarization", "translation"]

        if self.hparams.model_mode in seq2seq_models:
            self.decoder_start_token_id = None  # default to config
            self.eval_max_length = 62
            self.eval_min_length = 11
            self.eval_beams = 6

            print('for deocoding, eval_max_length={}, '
                'eval_min_length={}, eval_beams={}'.format(self.eval_max_length, self.eval_min_length, self.eval_beams))
            self.target_lens = {
                "train": self.hparams.max_target_length,
                "val": self.hparams.val_max_target_length,
                "test": self.hparams.test_max_target_length,
            }
            assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
            assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
            
            extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
            for p in extra_model_params:
                if getattr(self.hparams, p, None):
                    assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                    setattr(self.config, p, getattr(self.hparams, p))

            if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
                self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
                self.model.config.decoder_start_token_id = self.decoder_start_token_id

            self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
            assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}. Need an integer > 1"
            if self.hparams.eval_max_gen_length is not None:
                self.eval_max_length = self.hparams.eval_max_gen_length
            else:
                self.eval_max_length = self.model.config.max_length

    ''' setup and properties '''

    def setup(self, mode="fit", stage=None):
        if mode == "fit":
            self.train_loader = self._get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps * num_devices
        dataset_size = len(self.train_loader.dataset)
        return int(dataset_size / effective_batch_size) * self.hparams.num_train_epochs

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def is_seq2seq(self) -> bool:
        seq2seq = ["summarization", "translation"]
        return self.hparams.model_mode in seq2seq

    def _get_lr_scheduler(self):
        arg_to_scheduler = get_scheduler_info()['arg_to_scheduler']
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self._get_lr_scheduler()

        return [optimizer], [scheduler]

    """ dataset and dataloader """

    def _get_dataset(self, type_path) -> VictimTrainDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = None if self.target_lens is None else self.target_lens[type_path] 
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            task_mode=self.hparams.model_mode,
            task_name=self.hparams.task,
            **self.dataset_kwargs,
        )
        return dataset

    def _get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self._get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self._get_dataloader("val", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    """ for seq2seq model """
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.pad
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        if self.hparams.model_mode == 'seq2seq':
            if isinstance(self.model, T5ForConditionalGeneration):
                decoder_input_ids = self.model._shift_right(tgt_ids)
            else:
                decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)

            # outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
            #                use_prefix=True, return_dict=True, labels=tgt_ids)
            #
            # return (outputs.loss,)

            outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                        use_prefix=True)

            lm_logits = outputs[0]
            if self.hparams.label_smoothing == 0:
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

                assert lm_logits.shape[-1] == self.vocab_size
                # print(lm_logits.shape, tgt_ids.shape, lm_logits.shape[-1] )
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            else:
                lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(
                    lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
                )

        elif self.hparams.model_mode == 'sequence-classification':
            outputs = self(**batch)
            # todo: use default loss, which can be changed to customized one
            if self.hparams.label_smoothing == 0:
                loss = outputs['loss']
            else:
                # todo: implement label smoothing
                loss = outputs['loss']
            
        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        if self.hparams.model_mode == 'seq2seq':
            logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
            # tokens per batch
            logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
            logs["bs"] = batch["input_ids"].shape[0]
            logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
            logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()

            # print('hi', loss_tensors[0].item())
            self.training_loss_across_batches_at_curr_epoch.append(loss_tensors[0].item())
            # TODO(SS): make a wandb summary metric for this
            return {"loss": loss_tensors[0], "log": logs}

        elif self.hparams.model_mode == 'sequence-classification':
            logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
            # tokens per batch
            logs["tpb"] = batch["input_ids"].ne(self.pad).sum()
            logs["bs"] = batch["input_ids"].shape[0]
            logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
            logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()

            # print('hi', loss_tensors[0].item())
            self.training_loss_across_batches_at_curr_epoch.append(loss_tensors[0].item())
            return {"loss": loss_tensors[0], "log": logs}

        

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self._inference_step(batch)

    def on_train_epoch_end(self):
        train_loss_mean = np.mean(self.training_loss_across_batches_at_curr_epoch)
        print('train_loss = {}'.format(train_loss_mean))
        # print('train_PPL = {}'.format(train_acc_mean.exp()))
        self.training_loss_across_batches_at_curr_epoch = []  # reset for next epoch

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in self.loss_names}
        loss = losses["loss"]
        print('loss: ', loss)
        if self.is_seq2seq:
            metrics = {
                k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
            }
        else:
            metrics = {
                k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names
            }
        metrics.update(losses)
        # metrics.update({k: v.item() for k, v in losses.items()})
        metric_val = (
            metrics[self.val_metric_name] if self.val_metric_name in metrics else losses[self.val_metric_name]
        )
        # metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        losses.update(metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        
        """ log after epoch end. """
        eval_metric = self.metric.compute()
        log_val_metric = 'loss' if self.log_val_metric is None else self.log_val_metric
        self.log('val_{}'.format(self.val_metric_name), float(eval_metric[log_val_metric]))

        return all_metrics


    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    """ for seq2seq model """
    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)
        # return calculate_bleu(preds, target)

    def _inference_step(self, batch: dict) -> dict:
        bsz = batch["input_ids"].size(0)

        if self.hparams.model_mode == 'seq2seq':
            t0 = time.time()
            generated_ids = self.model.generate(
                batch["input_ids"],
                past_key_values=None,
                attention_mask=batch["attention_mask"],
                use_cache=True,
                length_penalty=self.hparams.length_penalty,
                use_prefix=True,
                decoder_start_token_id=self.decoder_start_token_id,
                num_beams=self.eval_beams,
                min_length=self.eval_min_length,
                max_length=self.eval_max_length,
            )
            gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
            preds: List[str] = self.ids_to_clean_text(generated_ids)
            target: List[str] = self.ids_to_clean_text(batch["labels"])
            loss_tensors = self._step(batch)
            base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
            # print('INPUT:', self.ids_to_clean_text(batch["input_ids"]))
            # print(preds, target)
            rouge: Dict = self.calc_generative_metrics(preds, target)
            self.log('val_{}'.format(self.val_metric_name), float(rouge['rouge2']))
            summ_len = np.mean(lmap(len, generated_ids))
            base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        
        elif self.hparams.model_mode == 'sequence-classification':
            outputs = self.model(**batch)
            preds: torch.Tensor(List[int]) = outputs.logits.argmax(dim=-1)
            target: torch.Tensor(List[int]) = batch["labels"]
            loss_tensors = self._step(batch)
            base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
            batch_acc = sum([1 if p == t else 0 for p, t in zip(preds, target)]) / bsz
            self.metric.add_batch(predictions=preds, references=target)
            # self.log('val_{}'.format(self.val_metric), float())
            base_metrics.update(accuracy=batch_acc, preds=preds.tolist(), target=target.tolist())

        return base_metrics

    """ save and load """

    @pl.utilities.rank_zero_only
    def save_checkpoint(self, checkpoint) -> None:
        print('Saving the the checkpoint.')
        return

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any], filepath='checkpoint') -> None:
        # if filepath is not None:
        #     save_path = filepath[:-5]
        # else:
        #     save_path = self.output_dir.joinpath("checkpoint-hello")
        save_path = Path(self.output_dir).joinpath("checkpoint-curr_best")
        print('the suggested save_path is {}, saving to {}'.format(filepath, save_path))

        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print('SAVING TO checkpoint {}'.format(save_path))
 

