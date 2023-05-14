import numpy as np
import torch
import evaluate
import nltk
import time
from tqdm.auto import tqdm
from functools import wraps
from typing import (
        Tuple,
        List,
        Dict,
        Optional,
    )
from torch.utils.data.dataloader import DataLoader
from accelerate.utils import DistributedType

from src.utils.text_normalization import normalize_answer
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
                 gen_kwargs,
                 dataloader: DataLoader,
                 ignore_pad_token_for_loss: bool = True,
                 with_tracking: bool = False
                 ):

        self.dataloader = dataloader
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.metrics_name = metrics_name
        self.with_tracking = with_tracking
        self.gen_kwargs = gen_kwargs

    @timeit
    def eval(self, accelerator, tokenizer, model, mode):
        accelerator.wait_for_everyone()
        accelerator.print("\n\n**** Starting evaluation ****\n")

        model.eval()

        metrics_list = {}
        for metric_name in self.metrics_name:
            metrics_list[metric_name]=(Metric(metric_name, accelerator))

        total_loss_eval = 0
        results = {'examples':{}}
        for step, batch in enumerate(tqdm(self.dataloader,
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
                    **self.gen_kwargs
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

                if mode == 'test':
                    if accelerator.is_main_process:
                        idx = 0
                        prompts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                        for prompt, decoded_pred, decoded_label in zip(prompts, decoded_preds, decoded_labels):
                            results['examples'][step+idx] = \
                                                    {
                                                       'prompt': prompt,
                                                       'prediction': decoded_pred,
                                                       'label': decoded_label
                                                    }
                            idx += 1

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

        for metric_name in metrics_list.keys():
            result = metrics_list[metric_name].compute()
            if result is not None:
                results.update(result)

        print(f"** Evaluation of process {accelerator.process_index} completed **")
        if self.with_tracking:
            return results, total_loss_eval
        return results

    def postprocess_text(self,
                    decoded_preds: List[str],
                    decoded_labels: List[str]) -> Tuple[List[str]]:
        """Normalize text before calculating metric
        """
        preds = [normalize_answer(pred.strip()) for pred in decoded_preds]
        labels = [normalize_answer(label.strip()) for label in decoded_labels]

        return preds, labels