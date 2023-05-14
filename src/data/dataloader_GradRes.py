import torch
import re

from typing import Optional, List, Union, Set
from datasets import DatasetDict, load_dataset, concatenate_datasets
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer


class GradResDataLoader:
    def __init__(self,
                 model_name: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]],
                 test_file: Optional[Union[str, List[str]]],
                 batch_size: int = 8,
                 seed: int = 42,
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None
                 ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples

        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, *args, **kwargs) -> Union[Set[DataLoader], Set]:
        dataloaders = {}

        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data('train', self.train_file)
            if self.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.max_train_samples))
            dataloaders['train'] = self.get_dataloader(train_dataset, shuffle_flag=True)

        if self.val_file is not None:
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data('val', self.val_file)
            if self.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.max_eval_samples))
            dataloaders['eval'] = self.get_dataloader(eval_dataset)

        if self.test_file is not None:
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data('test', self.test_file)
            if self.max_predict_samples is not None:
                test_dataset = test_dataset.select(range(self.max_predict_samples))
            dataloaders['test'] = self.get_dataloader(test_dataset)

        return dataloaders

    def load_data(self, key: str, data_file: List[str]) -> DatasetDict:
        """
        Loads a dataset from a file on disk and returns it as a dictionary of Dataset objects.

        Args:
            key (str): The key to assign to the loaded dataset in the returned dictionary of Dataset objects.
            data_file (Union[str, List[str]]): The path or paths to the data file(s) to load. If multiple is True,
                        data_file should be a list of file paths. Otherwise, it should be a single file path.
            mutiple (bool): A flag that indicates whether the data_file argument is a list of multiple file paths.

        Returns:
            A dictionary of Dataset objects that represents the loaded dataset. If mutiple is True, the function
            concatenates the datasets from the multiple files before returning them. Otherwise, it returns a single
            dataset loaded from the data_file path.
        """

        dataset_list = []
        for file in data_file:
            data_files = {key: file}
            extension = file.split(".")[-1]
            dataset_list.append(load_dataset(extension, data_files=data_files, split=key))
        dataset = concatenate_datasets(dataset_list)
        dataset.shuffle(self.seed)
        return dataset

    def dynamic_collate(self, batch_samples):
        """
        A collate function that tokenizes the inputs and targets, and applies dynamic padding and truncation
        based on the maximum length in the batch.

        Args:
            batch (list): A list of examples, where each example is a dictionary with a text column and a target column.

        Returns:
            dict: A dictionary with the input IDs, attention masks, and target IDs with attention masks
            where tokens are padded, and the target IDs are masked to exclude padded values.
        """

        def mapping_sample(samples):
            inputs, targets = [], []
            for sample in samples:
                item = sample['instruction'] \
                    .replace('{context}', sample['context'].strip()) \
                    .replace('{ontology}', sample['ontology'].strip()) \
                    .replace('{system_action}', sample['system_action'].strip()) \
                    .replace('{documents}', sample['documents'].strip()) \
                    .replace('{style}', sample['style'].strip())

                inputs.append(re.sub('\s+', ' ', item))
                targets.append(re.sub('\s+', ' ', sample['response'].strip()))
            return inputs, targets

        inputs, targets = mapping_sample(batch_samples)

        inp_tokens = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        tgt_tokens = self.tokenizer.batch_encode_plus(
            targets,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        target_ids = tgt_tokens["input_ids"]
        target_mask = tgt_tokens["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        return {"input_ids": inp_tokens["input_ids"],
                "attention_mask": inp_tokens["attention_mask"],
                "labels": target_ids}

    def get_dataloader(self, dataset, shuffle_flag: bool = False) -> DataLoader:
        """
        :param dataset: (Dataset): dataset from which to load the data.
        :param shuffle_flag: set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
        :return: a dataset
        """
        if shuffle_flag:
            sampler = RandomSampler(data_source=dataset, generator=self.generator)
        else:
            sampler =  SequentialSampler(dataset)

        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                collate_fn=self.dynamic_collate,
                                batch_size=self.batch_size,
                                drop_last=True,)

        return dataloader