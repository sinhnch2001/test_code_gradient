import numpy as np
import torch
import time

import nltk
nltk.download('punkt', quiet=True)

from tqdm.auto import tqdm
from functools import wraps
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from accelerate.utils import DistributedType

from metric import Metric


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function evaluation Took {total_time:.4f} seconds')

        return result
    return timeit_wrapper

class Evaluation:
    def __init__(self,
                 metrics_name,
                 eval_dataloaders: DataLoader,
                 ignore_pad_token_for_loss: bool = True,
                 with_tracking: bool = False,
                 num_beams: Optional[int] = 4,
                 max_target_length: Optional[int] = 40
                 ):

        self.eval_dataloaders = eval_dataloaders
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.metrics_name = metrics_name
        self.with_tracking = with_tracking
        self.num_beams = num_beams
        self.max_target_length = max_target_length

    @timeit
    def eval(self, accelerator, tokenizer, model):
        accelerator.wait_for_everyone()
        accelerator.print("\n\n**** Starting evaluation ****\n")
        model.eval()

        gen_kwargs = {
            "max_length": self.max_target_length,
            "num_beams": self.num_beams
        }
        metrics_list = {}
        for metric_name in self.metrics_name:
            metrics_list[metric_name]=(Metric(metric_name))

        total_loss_eval = 0
        for step, batch in enumerate(tqdm(self.eval_dataloaders,
                                          desc="Eval on process: " + str(accelerator.process_index),
                                          colour="blue", position=accelerator.process_index)):
            # Pass dummy batch to avoid caffe error
            if step == 0 and accelerator.distributed_type == DistributedType.FSDP:
                model(**batch)
            with torch.no_grad():
                # synced_gpus was necessary else resulted into indefinite hang
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    synced_gpus=True if accelerator.distributed_type != DistributedType.NO else False,
                    **gen_kwargs
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                          pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if self.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                for metric_name in metrics_list.keys():
                    metrics_list[metric_name].add_batch(decoded_preds=decoded_preds, decoded_labels=decoded_labels)

                del decoded_preds
                del decoded_labels

                if self.with_tracking:
                    # Compute and log the loss
                    unwrap_model = accelerator.unwrap_model(model)
                    outputs = unwrap_model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
                    loss = outputs.loss
                    total_loss_eval += loss.detach().float()

        result_list = {}
        for metric_name in metrics_list.keys():
            result = metrics_list[metric_name].compute()
            result_list[metric_name] = result

        print(f"** Evaluation of process {accelerator.process_index} completed **")
        if self.with_tracking:
            return result_list, total_loss_eval
        return result_list

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
