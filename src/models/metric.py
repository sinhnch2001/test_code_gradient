import numpy as np
import os
import re
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import evaluate
import datasets
# https://github.com/huggingface/evaluate/issues/428
from datasets import DownloadConfig


class Metric:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.dict_preds_labels = {}

        if self.metric_name == "bleurt":
            self.metric = evaluate.load(self.metric_name, "bleurt-base-128", download_config=DownloadConfig(use_etag=False))
        elif self.metric_name == "rouge" or self.metric_name == "bleu" or self.metric_name == "bertscore":
            self.metric = evaluate.load(self.metric_name)
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
            result_rouge = {k: round(v * 100, 4) for k, v in result_rouge.items()}
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
            result = result_bleurt["scores"]
        elif self.metric_name == "relative-slot-acc":
            a=1
        return result


if __name__ == '__main__':
    metrics_name = ["bertscore", "bleu", "rouge", "bleurt"]
    metrics_list = {}
    for metric_name in metrics_name:
        metrics_list[metric_name] = (Metric(metric_name))
    print(metrics_list)
    # metric = Metric("bleurt") # or bertscore, bleu, rouge, bleurt
    # metric.add_batch(["hello there general kenobi", "foo bar foobar"], ["hi there general kenobi","foo bar foobar"])
    # result = metric.compute()
    # print(result)