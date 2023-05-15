import json
import logging
import math
import os
import numpy as np
import torch
import time
from functools import wraps
from pathlib import Path
from typing import Set, Optional, Union, List, Dict, Tuple
from typing_extensions import Literal
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

# Transformers
import transformers
import datasets
import accelerate.utils
from transformers import (
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
                 training_args,
                 model,
                 tokenizer,
                 config,
                 dataloaders: Set[DataLoader],
                 ):

        self.args = training_args
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataloaders = dataloaders

    @timeit
    def train(self):
        # log in wandb
        if self.args.with_tracking and (self.args.report_to == 'all' or self.args.report_to == 'wandb'):
            if os.getenv("WANDB_API_KEY") is None:
                logger.info('Can\'t log to WanDB')
            else:
                wandb_key = os.getenv("WANDB_API_KEY")
                try:
                    wandb.login(key=wandb_key)
                except:
                    logger.warning('WanDB is not available')

        if self.args.seed is not None:
            set_seed(self.args.seed)
            logger.info('Training seed: {}'.format(self.args.seed))

        if self.args.with_tracking:
            accelerator_log_kwargs = {}
            accelerator_log_kwargs["log_with"] = self.args.report_to
            accelerator_log_kwargs["project_dir"] = self.args.output_dir

        accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                                  **accelerator_log_kwargs)

        # Get gradient accumulation steps from deepspeed config if available
        if accelerator.state.deepspeed_plugin is not None:
            self.args.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        accelerator.gradient_accumulation_steps = self.args.gradient_accumulation_steps

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

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.dataloaders['train']) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        # Prepare for training
        model, optimizer, dataloaders, lr_scheduler = self.prepare_any(accelerator.distributed_type, accelerator)

        accelerator.register_for_checkpointing(lr_scheduler)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(dataloaders['train']) / self.args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # we need to initialize the trackers we use, and also store our configuration.
        # the trackers initializes automatically on the main process.
        if self.args.with_tracking:
            experiment_config = vars(self.args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers(
                                        project_name="resgen",
                                        init_kwargs={"wandb": {"name": self.args.exp_name}, 'mlflow': {'run_name': self.args.exp_name}},
                                        config=experiment_config)
        # Init Evaluation
        metrics_name = ['rouge', 'bertscore']
        gen_kwargs = {"num_beams": self.args.num_beams,
                      "max_length": self.args.max_target_length,
                      "min_length": self.args.min_length,
                      "top_p": self.args.top_p,
                      "top_k": self.args.top_k,
                      "repetition_penalty": self.args.repetition_penalty,
                      "temperature": self.args.temperature,
                      "no_repeat_ngram_size": self.args.no_repeat_ngram_size,
                      "num_return_sequences": self.args.num_return_sequences}
        evaluator = Evaluation(dataloader=dataloaders['eval'],
                               ignore_pad_token_for_loss=self.args.ignore_pad_token_for_loss,
                               metrics_name=metrics_name,
                               with_tracking=self.args.with_tracking,
                               gen_kwargs=gen_kwargs)

        total_batch_size = self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataloaders['train'])}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        print("***** Running training *****")
        print(f"  Num examples = {len(dataloaders['train'])}")
        print(f"  Num Epochs = {self.args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self.args.max_train_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process, colour="green")
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint is not None or self.args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {self.args.resume_from_checkpoint}")
                accelerator.load_state(self.args.resume_from_checkpoint)
                path = os.path.basename(self.args.resume_from_checkpoint)
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
                resume_step = int(training_difference.replace("step_", "")) * self.args.gradient_accumulation_steps
                starting_epoch = resume_step // len(dataloaders['train'])
                resume_step -= starting_epoch * len(dataloaders['train'])

        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch
        rougeLSum_scores = []
        for epoch in range(starting_epoch, self.args.num_train_epochs):
            model.train()
            if self.args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(dataloaders['train']):
                # We need to skip steps until we reach the resumed step
                if self.args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % self.args.gradient_accumulation_steps == 0:
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
                    if self.args.with_tracking:
                        loss_detached = loss.detach().float()
                        total_loss += loss_detached
                        accelerator.log({"training_loss_batch": float(loss_detached)}, step=completed_steps)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                if isinstance(self.args.checkpointing_steps, int):
                    with accelerator.main_process_first():
                        self.save_cpkt(accelerator, checkpointing_steps=self.args.checkpointing_steps,
                                   completed_steps=completed_steps)

                if completed_steps >= self.args.max_train_steps:
                    break
            # Eval per epoch
            if self.args.do_eval_per_epoch:
                if self.args.with_tracking:
                    result, total_loss_eval = evaluator.eval(accelerator=accelerator,
                                                             tokenizer=self.tokenizer, model=model, mode='val')
                else:
                    result = evaluator.eval(accelerator=accelerator,
                                            tokenizer=self.tokenizer, model=model, mode='val')

                if accelerator.is_main_process:
                    logger.info(result)
                    if self.args.with_tracking:
                        result["train_loss"] = total_loss.item() / len(self.dataloaders['train'])
                        result["epoch"] = epoch
                        result["eval_loss"] = total_loss_eval.item() / len(self.dataloaders['eval'])
                        rougeLSum_scores.append(result['rougeLsum'])
                        accelerator.log(result, step=completed_steps)
                        logger.info(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
                        logger.info(result["train_loss"])
                        logger.info(f"*** EVAL LOSS AT EPOCH {epoch} ***")
                        logger.info(result["eval_loss"])

                        print(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
                        print("train_loss: ", total_loss.item() / len(self.dataloaders['train']))
                        print(f"*** EVAL LOSS AT EPOCH {epoch} ***")
                        print("eval_loss: ", total_loss_eval.item() / len(self.dataloaders['eval']))

                # Saving best rougeLsum score
                if self.args.output_dir is not None and self.args.with_tracking:
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

                            print(f"***** Saving best rougeLsum score epoch *****")
                            print(f"Saving epoch: {epoch}")
                            self.save(accelerator, unwrapped_model, self.tokenizer, result, state, 'Best')
                        else:
                            logger.info(f"***** Discarding epoch {epoch} *****")
                            print(f"***** Discarding epoch {epoch} *****")

            else:
                result = {}
                if self.args.with_tracking:
                    result["epoch"] = epoch
                    result["train_loss"] = total_loss.item() / len(self.dataloaders['train'])
                    accelerator.log(result, step=completed_steps)
                    logger.info(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
                    logger.info(result["train_loss"])

                    print(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
                    print("train_loss: ",total_loss.item() / len(self.dataloaders['train']))
            accelerator.wait_for_everyone()
            if self.args.checkpointing_steps == "epoch":
                with accelerator.main_process_first():
                    self.save_cpkt(accelerator,checkpointing_steps=self.args.checkpointing_steps,epoch=epoch)

            # print(f"*** TRAINING LOSS AT EPOCH {epoch} ***")
            # print("train_loss: ", total_loss.item() / len(self.dataloaders['train']))
            # print(f"*** EVAL LOSS AT EPOCH {epoch} ***")
            # print("eval_loss: ", total_loss_eval.item() / len(self.dataloaders['eval']))
            print("RougeL: ", result["rougeL"])
            print(f"ENDING EPOCH: {epoch} on process "+str(accelerator.process_index))

        if self.args.with_tracking:
            accelerator.end_training()

        # Final save when the training is done
        if self.args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.distributed_type == DistributedType.FSDP:
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT,
                                          full_state_dict_config):
                    state = accelerator.get_state_dict(unwrapped_model)
            else:
                state = accelerator.get_state_dict(model)
            self.save(accelerator, unwrapped_model, self.tokenizer, result, state, save_type='Final')

    def save(self, accelerator, unwrapped_model, tokenizer, result, state, save_type):
        if self.args.output_dir is not None:
            model_dir = Path(os.path.join(self.args.output_dir, save_type))
            if not model_dir.is_dir():
                model_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model.save_pretrained(
            model_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(model_dir)
            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(model_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

    def save_cpkt(self, accelerator, checkpointing_steps, epoch=None, completed_steps=None):
        if self.args.output_dir is not None:
            checkpoints_dir = Path(os.path.join(self.args.output_dir, 'checkpoints'))
            if not checkpoints_dir.is_dir():
                checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if checkpointing_steps == "epoch":
            logger.info(f"***** Saving checkpoint at epoch {epoch} *****")
            print(f"***** Saving checkpoint at epoch {epoch} *****")
            output_dir = f"epoch_{epoch}"
            output_dir = os.path.join(checkpoints_dir, output_dir)
            accelerator.save_state(output_dir)

        elif isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                logger.info(f"***** Saving checkpoint at steps {completed_steps} *****")
                print(f"***** Saving checkpoint at steps {completed_steps} *****")
                output_dir = f"step_{completed_steps}"
                output_dir = os.path.join(checkpoints_dir, output_dir)
                accelerator.save_state(output_dir)

    def prepare_any(self, distributed_type, accelerator):

        def get_grouped_parameters(model):
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
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
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.num_warmup_steps * self.args.gradient_accumulation_steps,
                num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
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
            optimizer = optimizer_cls(optimizer_grouped_parameters, lr=self.args.learning_rate)
            # Creates Dummy Scheduler if `scheduler` was specified in the config
            # file else creates `self.lr_scheduler_type` Scheduler
            if (
                    accelerator.state.deepspeed_plugin is None
                    or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
            ):
                lr_scheduler = get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.num_warmup_steps * self.args.gradient_accumulation_steps,
                    num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
                )
            else:
                lr_scheduler = DummyScheduler(
                    optimizer,
                    total_num_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
                    warmup_num_steps=self.args.num_warmup_steps * self.args.gradient_accumulation_steps
                )
            model, optimizer, dataloaders['train'], dataloaders['eval'], lr_scheduler = accelerator.prepare(
                self.model, optimizer, self.dataloaders['train'], self.dataloaders['eval'], lr_scheduler
            )

        return model, optimizer, dataloaders, lr_scheduler


def get_model(
        model_name_or_path: str,
        config: Dict,
) -> AutoModelForSeq2SeqLM:
    """
    Load model from checkpoint or huggingface hub

    Args:
        model_name_or_path: path to checkpoint folder
                            or name of model on huggingface hub
        config: model configuration including dropout rate, hidden dim,
                ...

    Returns:
        model
    """
    if model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    return model


def get_model_config(
        config_name: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        model_type: str = "t5-base") -> AutoConfig:
    """Get model's configuration to initialize it

    Args:
        config_name: pretrained config name or path if not the same
                    as model name
        model_name_or_path: path to pretrained model or model identifier
                    from huggingface.co/models
    Returns
        config: model's configuration"""
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    return config


def get_tokenizer(
        tokenizer_name: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        use_slow: bool = False
) -> AutoTokenizer:
    """Create model tokenizer

    Args:
        tokenizer_name: name or path to tokenizer checkpoint
        model_name_or_path: path to pretrained model or model identifier
                            from huggingface hub
        use_slow: whether to use fast tokenizer or not

    Returns:
        tokenizer
    """
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=not use_slow)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=not use_slow)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def setup_training(
                config_name: str,
                model_name_or_path: str,
                tokenizer_name: str,
                model_type: str,
                use_slow: bool = False
        ) -> Tuple[AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer]:
    config = get_model_config(
        config_name=config_name,
        model_name_or_path=model_name_or_path,
        model_type=model_type
    )
    model = get_model(
        model_name_or_path=model_name_or_path,
        config=config
    )
    tokenizer = get_tokenizer(
        tokenizer_name=tokenizer_name,
        model_name_or_path=model_name_or_path,
        use_slow=use_slow
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return config, model, tokenizer