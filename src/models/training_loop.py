import argparse
import json
import logging
import math
import os
import numpy as np
import torch
import time
from functools import wraps
from pathlib import Path
from typing import Set, Optional, Union
from typing_extensions import Literal
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

# Transformers
import transformers
import datasets
import evaluate
import accelerate.utils
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler
)
from evaluation import Evaluation
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed  # reproducability across devices
from accelerate.utils import DistributedType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig


logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function train Took {total_time:.4f} seconds')

        return result
    return timeit_wrapper


class Trainer:
    def __init__(self,
                 model_name_or_path: str,
                 output_dir: str,
                 dataloaders: Set[DataLoader],
                 max_target_length: Optional[int] = 40,
                 ignore_pad_token_for_loss: bool = True,
                 num_beams: Optional[int] = 4,
                 config_name: Optional[str] = None,
                 tokenizer_name: Optional[str] = None,
                 use_slow_tokenizer: bool = False,
                 per_device_batch_size: Optional[int] = 8,
                 learning_rate: Optional[float] = 5e-5,
                 weight_decay: Optional[float] = 0.0,
                 num_train_epochs: Optional[int] = 3,
                 max_train_steps: Optional[int] = None,
                 gradient_accumulation_steps: Optional[int] = 1,
                 lr_scheduler_type: Literal = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                               "constant_with_warmup"],
                 num_warmup_steps: Optional[int] = 0,
                 seed: Optional[int] = None,
                 model_type: Optional[str] = None,
                 checkpointing_steps: Optional[Union[str,int]] = None,
                 resume_from_checkpoint: Optional[Union[str,bool]] = False,
                 with_tracking: bool = False,
                 report_to: Optional[str] = None,
                 do_eval_per_epoch: Optional[bool] = False):

        # Save the input parameters
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.output_dir = output_dir
        self.dataloaders = dataloaders

        self.max_target_length = max_target_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.num_beams = num_beams
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.use_slow_tokenizer = use_slow_tokenizer

        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps

        self.seed = seed

        self.checkpointing_steps = checkpointing_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.with_tracking = with_tracking
        self.report_to = report_to
        self.do_eval_per_epoch = do_eval_per_epoch

    @timeit
    def train(self):

        accelerator_log_kwargs = {}

        if self.seed is not None:
            set_seed(self.seed)

        if self.with_tracking:
            accelerator_log_kwargs["log_with"] = self.report_to
            accelerator_log_kwargs["logging_dir"] = self.output_dir

        accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps,
                                  **accelerator_log_kwargs)

        # Get gradient accumulation steps from deepspeed config if available
        if accelerator.state.deepspeed_plugin is not None:
            self.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        accelerator.gradient_accumulation_steps = self.gradient_accumulation_steps

        self.dataloaders = self.dataloaders.__call__()

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        if self.config_name:
            config = AutoConfig.from_pretrained(self.config_name, return_dict=True)
        elif self.model_name_or_path:
            config = AutoConfig.from_pretrained(self.model_name_or_path, return_dict=True)
        else:
            config = CONFIG_MAPPING[self.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if self.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=not self.use_slow_tokenizer)
        elif self.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                           use_fast=not self.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        device = accelerator.device
        with accelerator.main_process_first():
            if self.model_name_or_path:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name_or_path,
                    from_tf=bool(".ckpt" in self.model_name_or_path),
                    config=config
                ).to(device)
            else:
                logger.info("Training new model from scratch")
                self.model = AutoModelForSeq2SeqLM.from_config(config).to(device)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(tokenizer))
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.dataloaders['train']) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        # Prepare for training
        model, optimizer, dataloaders, lr_scheduler = self.prepare_any(accelerator.distributed_type, accelerator)

        accelerator.register_for_checkpointing(lr_scheduler)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(dataloaders['train']) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.with_tracking:
            experiment_config = {
                "model_name_or_path": self.model_name_or_path,
                "max_target_length": self.max_target_length,
                "num_beams": self.num_beams,
                "per_device_batch_size": self.per_device_batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_train_epochs": self.num_train_epochs,
                "max_train_steps": self.max_train_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "lr_scheduler_type": self.lr_scheduler_type,
                "num_warmup_steps": self.num_warmup_steps,
                "seed": self.seed,
                "model_type": self.model_type
            }
            # TensorBoard cannot log Enums, need the raw value
            if isinstance(experiment_config["lr_scheduler_type"], str):
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
            else:
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            i=0
            name = "state_prediction"
            while True:
                try:
                    accelerator.init_trackers(name,experiment_config)
                    break
                except Exception:
                    name = name + str(i)
                    i+=1
        # Metric
        metric_rouge = evaluate.load("rouge",num_process=accelerator.num_processes,process_id=accelerator.process_index)
        metric_bleu = evaluate.load("bleu",num_process=accelerator.num_processes,process_id=accelerator.process_index)
        metrics = [metric_rouge,metric_bleu]
        evaluator = Evaluation(eval_dataloaders = dataloaders['eval'],
                               ignore_pad_token_for_loss = self.ignore_pad_token_for_loss,
                               metrics_name = metrics,
                               with_tracking = self.with_tracking,
                               num_beams = self.num_beams,
                               max_target_length = self.max_target_length)

        total_batch_size = self.per_device_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataloaders['train'])}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.per_device_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.max_train_steps), disable=not accelerator.is_local_main_process, colour="green")
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            if self.resume_from_checkpoint is not None or self.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {self.resume_from_checkpoint}")
                accelerator.load_state(self.resume_from_checkpoint)
                path = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.gradient_accumulation_steps
                starting_epoch = resume_step // len(dataloaders['train'])
                resume_step -= starting_epoch * len(dataloaders['train'])

        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * self.num_update_steps_per_epoch)
        completed_steps = starting_epoch * self.num_update_steps_per_epoch
        rougeLSum_scores = []
        for epoch in range(starting_epoch, self.num_train_epochs):
            model.train()
            if self.with_tracking:
                total_loss = 0
            for step, batch in enumerate(dataloaders['train']):
                # We need to skip steps until we reach the resumed step
                if self.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % self.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if self.with_tracking:
                        loss_detached = loss.detach().float()
                        total_loss += loss_detached
                        accelerator.log({"training_loss_batch": float(loss_detached)}, step=completed_steps)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                if isinstance(self.checkpointing_steps, int):
                    self.save_cpkt(accelerator, checkpointing_steps=self.checkpointing_steps,
                                   completed_steps=completed_steps)

                if completed_steps >= self.max_train_steps:
                    break

            # Eval per epoch
            if self.do_eval_per_epoch:
                if self.with_tracking:
                    result, total_loss_eval = evaluator.eval(accelerator=accelerator,
                                                             tokenizer=tokenizer, model=model)
                else:
                    result = evaluator.eval(accelerator=accelerator,
                                            tokenizer=tokenizer, model=model)

                if accelerator.is_main_process:
                    logger.info(result)
                    if self.with_tracking:
                        result["train_loss"] = total_loss.item() / len(self.dataloaders['train'])
                        result["epoch"] = epoch
                        result["eval_loss"] = total_loss_eval.item() / len(self.dataloaders['eval'])
                        rougeLSum_scores.append(result['rougeLsum'])
                        accelerator.log(result, step=completed_steps)
                        logger.info(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
                        logger.info(result["train_loss"])
                        logger.info(f"*** EVAL LOSS AT EPOCH {epoch} ***")
                        logger.info(result["eval_loss"])

                # Saving best rougeLsum score
                if self.output_dir is not None and self.with_tracking:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    if accelerator.distributed_type == DistributedType.FSDP:
                        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT,
                                                  full_state_dict_config):
                            state = accelerator.get_state_dict(unwrapped_model)
                    else:
                        state = accelerator.get_state_dict(model)

                    if accelerator.is_main_process:
                        if result['rougeLsum'] == max(rougeLSum_scores):
                            logger.info(f"***** Saving best rougeLsum score epoch *****")
                            logger.info(f"Saving epoch: {epoch}")
                            self.save(accelerator, unwrapped_model, tokenizer, result, state)
                        else:
                            logger.info(f"***** Discarding epoch {epoch} *****")

            else:
                result = {}
                if self.with_tracking:
                    result["epoch"] = epoch
                    result["train_loss"] = total_loss.item() / len(self.dataloaders['train'])
                    accelerator.log(result, step=completed_steps)
                    logger.info(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
                    logger.info(result["train_loss"])

            accelerator.wait_for_everyone()
            if self.checkpointing_steps == "epoch":
                self.save_cpkt(accelerator,checkpointing_steps=self.checkpointing_steps,epoch=epoch)
            print(f"ENDING EPOCH: {epoch} on process "+str(accelerator.process_index))

        if self.with_tracking:
            accelerator.end_training()

        # If eval per epoch == False, only save when the training is done
        if self.output_dir is not None and not self.do_eval_per_epoch:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.distributed_type == DistributedType.FSDP:
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT,
                                          full_state_dict_config):
                    state = accelerator.get_state_dict(unwrapped_model)
            else:
                state = accelerator.get_state_dict(model)
            self.save(accelerator, unwrapped_model, tokenizer, result, state)

    def save(self, accelerator, unwrapped_model, tokenizer, result, state):
        unwrapped_model.save_pretrained(
            self.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(self.output_dir)
            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

    def save_cpkt(self, accelerator, checkpointing_steps, epoch=None, completed_steps=None):
        if checkpointing_steps == "epoch":
            logger.info(f"***** Saving checkpoint at epoch {epoch} *****")
            output_dir = f"epoch_{epoch}"
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, output_dir)
            accelerator.save_state(output_dir)

        elif isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                logger.info(f"***** Saving checkpoint at steps {completed_steps} *****")
                output_dir = f"step_{completed_steps}"
                if self.output_dir is not None:
                    output_dir = os.path.join(self.output_dir, output_dir)
                accelerator.save_state(output_dir)

    def prepare_any(self, distributed_type, accelerator):

        def get_grouped_parameters(model):
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            return optimizer_grouped_parameters

        dataloaders = {}
        if distributed_type != DistributedType.DEEPSPEED:
            model = accelerator.prepare(self.model)
            optimizer_grouped_parameters = get_grouped_parameters(model)
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            lr_scheduler = get_scheduler(
                name=self.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps * self.gradient_accumulation_steps,
                num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
            )
            optimizer, dataloaders['train'], dataloaders['eval'], lr_scheduler = accelerator.prepare(
                optimizer, self.dataloaders['train'], self.dataloaders['eval'], lr_scheduler
            )
        else:
            # Creates Dummy Optimizer if `optimizer` was specified in the config
            # file else creates Adam Optimizer
            optimizer_cls = (
                torch.optim.AdamW
                if accelerator.state.deepspeed_plugin is None
                   or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
                else DummyOptim
            )
            optimizer_grouped_parameters = get_grouped_parameters(self.model)
            optimizer = optimizer_cls(optimizer_grouped_parameters, lr=self.learning_rate)
            # Creates Dummy Scheduler if `scheduler` was specified in the config
            # file else creates `self.lr_scheduler_type` Scheduler
            if (
                    accelerator.state.deepspeed_plugin is None
                    or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
            ):
                lr_scheduler = get_scheduler(
                    name=self.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.num_warmup_steps * self.gradient_accumulation_steps,
                    num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
                )
            else:
                lr_scheduler = DummyScheduler(
                    optimizer,
                    total_num_steps=self.max_train_steps * self.gradient_accumulation_steps,
                    warmup_num_steps=self.num_warmup_steps * self.gradient_accumulation_steps
                )
            model, optimizer, dataloaders['train'], dataloaders['eval'], lr_scheduler = accelerator.prepare(
                self.model, optimizer, self.dataloaders['train'], self.dataloaders['eval'], lr_scheduler
            )

        return model, optimizer, dataloaders, lr_scheduler












