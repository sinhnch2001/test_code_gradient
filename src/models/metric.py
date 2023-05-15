import numpy as np
import os
import tensorflow as tf
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import evaluate
import datasets
# https://github.com/huggingface/evaluate/issues/428
from datasets import DownloadConfig

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 256MB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class Metric:
    def __init__(self, metric_name,accelerator):
        self.metric_name = metric_name
        self.dict_preds_labels = {}

        if self.metric_name == "bleurt":
            self.metric = evaluate.load(self.metric_name,
                                        "bleurt-base-128",
                                        download_config=DownloadConfig(use_etag=False),
                                        num_process=accelerator.num_processes,
                                        process_id=accelerator.process_index)
        elif self.metric_name == "rouge" or self.metric_name == "bleu" or self.metric_name == "bertscore":
            self.metric = evaluate.load(self.metric_name,
                                        num_process=accelerator.num_processes,
                                        process_id=accelerator.process_index)
        else:
            self.metric = None

    def add_batch(self, decoded_preds, decoded_labels):
        if self.metric != None:
            self.metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels)
        else:
            for i in range(len(decoded_preds)):
                self.dict_preds_labels.setdefault(decoded_labels[i], decoded_preds[i])

    def parse_string(label):
        # Tạo một dictionary trống
        print(label)
        elements = label.split(' <domain> ')
        slots = []
        for element in elements:
            parts = element.split(' <slot> ')
            domain = parts[0]
            domain = domain.replace("<domain> ", "")
            for i in range(1, len(parts)):
                slots.append(domain.lower() + "-" + parts[i])
        return slots

    def compute(self):
        if self.metric_name == "rouge":
            result_rouge = self.metric.compute(use_stemmer=True)
            print(type(result_rouge))
            for k, v in result_rouge.items():
                result_rouge[k] = round(v * 100, 4)
            result = result_rouge

        elif self.metric_name == "bleu":
            result_bleu = self.metric.compute()
            for k, v in result_bleu.items():
                if k == 'precisions':
                    for i in range(len(v)):
                        result_bleu['precisions'][i] = round(v[i] * 100, 4)
                else:
                    result_bleu[k] = round(v * 100, 4)
            result = result_bleu

        elif self.metric_name == "bertscore":
            result_bert = self.metric.compute(model_type="distilbert-base-uncased")
            result_bert["precision"] = round(np.mean(result_bert["precision"]) * 100, 4)
            result_bert["recall"] = round(np.mean(result_bert["recall"]) * 100, 4)
            result_bert["f1"] = round(np.mean(result_bert["f1"]) * 100, 4)
            result = result_bert

        elif self.metric_name == "bleurt":
            result_bleurt = self.metric.compute()
            result_bleurt["scores"] = round(np.mean(result_bleurt["scores"])*100, 4)
            result = result_bleurt
        elif self.metric_name == "relative-slot-acc":
            a=1
        return result


if __name__ == '__main__':
    metrics_name = ["bertscore", "bleu", "rouge", "bleurt"]
    metrics_list = {}
    for metric_name in metrics_name:
        metrics_list[metric_name] = (Metric(metric_name))
    print(metrics_list)
    metric = Metric("rouge") # or bertscore, bleu, rouge, bleurt
    metric.add_batch(["hello there general kenobi", "foo bar foobar"], ["hi there general kenobi","foo bar foobar"])
    result = metric.compute()
    print(result)