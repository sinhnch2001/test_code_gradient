import os
import json
import random
import logging
import argparse
import numpy as np

from pathlib import Path
from typing import (
    Tuple,
    List,
    Dict,
    Union,
    Optional
)
from argparse import ArgumentParser
from src.features.converter import DialConverter

logger = logging.getLogger(__name__)


class WoW_Converter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 instruct_file_path: str,
                 tag_speaker: str = 'USER',
                 tag_agent: str = 'AGENT',
                 style_odd: List[str] = ["politely", "empathically", "safety", "friendly"],
                 window_context: int = 0,
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path,
                         save_path,
                         tag_speaker,
                         tag_agent,
                         window_context,
                         )
        self.instruct_file_path = instruct_file_path
        self.style_odd = style_odd

    def define_ontology(
            self,
            ontolopy_path: Optional[str] = None
    ) -> Union[Dict, List[Dict]]:
        """Define and load an ontology use in Task oriented Domain Dialogue
        for inference or evaluate

        Args:
            ontolopy_path: where on disk the source ontology template is located.

        Returns:
            The list of dictionaries or the dictionary of samples in the dataset
        """
        logger.warning("No ontology is specified")
        return None

    def define_instruct(self,
                        instruct_file_path: str) -> List[str]:
        with open(instruct_file_path, "r") as f:
            instructions = f.readlines()
        return instructions

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction[:-1]

    def get_context(self,
                    first_diag: str,
                    list_turns: List[str]):
        context = ""
        """
            Concat text with length for context
            Ex: context = <tag_speaker> .... <tag_agent> .... <tag_speaker>
        """
        if first_diag is not None:
            context = context + "USER" + ": " + first_diag + " "
        for turn in list_turns:
            if turn["speaker"] == "1_Apprentice":
                context = context + self.tag_speaker + ": " + turn['text'] + " "
                continue
            context = context + self.tag_agent + ": " + turn['text'] + " "
        return context.strip()

    def get_knowledge(self, turn):
        if turn['checked_sentence'] == {}:
            return "no_passages_used"
        knowledge = list(turn['checked_sentence'].values())[0]
        return knowledge

    def _process_diag(self,
                      sample: dict) -> List[dict]:
        processed_sample = []
        for i, turn in enumerate(sample['dialog']):
            info = {}
            # pass first iter
            if turn["speaker"] != "0_Wizard" or i == 0: continue
            if turn["speaker"] == "0_Wizard" and i <= self.window_context:
                info["instruction"] = self.get_instruction(self.list_instructions)
                info['context'] = self.get_context(first_diag=sample['persona'],
                                                   list_turns=sample['dialog'][0:i])
                info['documents'] = self.get_knowledge(turn)
                info["style"] = self.style_odd[np.random.randint(low=0, high=4)]
                info['response'] = turn['text']
                info['ontology'] = ''
                info['system_action'] = ''
                processed_sample.append(info)
                continue
            info["instruction"] = self.get_instruction(self.list_instructions)
            info['context'] = self.get_context(first_diag=None,
                                               list_turns=sample['dialog'][(i - self.window_context):i])
            info['documents'] = self.get_knowledge(turn)
            info["style"] = self.style_odd[np.random.randint(low=0, high=4)]
            info['response'] = turn['text']

            info['ontology'] = ''
            info['system_action'] = ''
            processed_sample.append(info)
        return processed_sample

    def process(self, ratio_val: int = None, ratio_test: int = None) -> None:
        if ratio_test is not None and ratio_val is not None:
            self.process_with_ratio(ratio_val, ratio_test)
        else:
            self.process_without_ratio()

    def process_without_ratio(self) -> None:
        """
        Convert raw train, val, test datasets to specified format
        """

        logger.info("Loading ungrounded train, val, test datasets...")
        train, val, test = self.load_datapath()

        logger.info("Loading instructions...")
        self.list_instructions = self.define_instruct(self.instruct_file_path)

        logger.info("Processing ungrounded train, val, test datasets...")
        self.train = []
        for each_sample in train:
            processed_sample = self._process_diag(each_sample)
            self.train.extend(processed_sample)

        self.val = []
        for each_sample in val:
            processed_sample = self._process_diag(each_sample)
            self.val.extend(processed_sample)

        self.test = []
        for each_sample in test:
            processed_sample = self._process_diag(each_sample)
            self.test.extend(processed_sample)

        logger.info("Saving grounded train, val, test datasets...")
        self.save_datapath()
        return None

    def process_with_ratio(self,
                           valid_ratio: float,
                           test_ratio: float
                           ) -> None:

        def distribute_datsets(
                dataset: List[Dict],
                valid_ratio: float,
                test_ratio: float
        ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
            """Distribute data to train, val, test sets by type and ratio
            Args:
                dataset (List[Dict]): dataset
                valid_ratio (float): ratio of splitting valid set
                test_ratio (float): ratio of splitting test set
            Returns:
                train (List[Dict]): train dataset
                valid (List[Dict]): valid dataset
                test (List[Dict]): test dataset
            """

            train_ratio = 1.0 - valid_ratio - test_ratio
            train_size = round(len(dataset) * train_ratio)
            valid_size = round(len(dataset) * (train_ratio + valid_ratio))

            train = dataset[:train_size]
            valid = dataset[train_size:valid_size]
            test = dataset[valid_size:]

            return train, valid, test

        logger.info("Loading file ungrounded dataset...")
        with open(os.path.join(self.file_path, "data.json"), mode="r", encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info("Loading instructions...")
        self.list_instructions = self.define_instruct(self.instruct_file_path)

        processed_data = []
        logger.info("Processing file ungrounded dataset...")
        for each_sample in self.data:
            samples = self._process_diag(each_sample)
            processed_data.extend(samples)

        self.train, self.val, self.test = distribute_datsets(dataset=processed_data,
                                                             valid_ratio=valid_ratio,
                                                             test_ratio=test_ratio)

        logger.info("Saving grounded train, val, test datasets...")
        self.save_datapath()
        return None

    def save_datapath(self) -> None:
        """
        Save file by save_path path
        """
        with open(os.path.join(self.save_path, "train.json"), "w+") as f:
            json.dump(self.train, f, indent=4)

        with open(os.path.join(self.save_path, "val.json"), "w+") as f:
            json.dump(self.val, f, indent=4)

        with open(os.path.join(self.save_path, "test.json"), "w+") as f:
            json.dump(self.test, f, indent=4)
        return None

    def load_datapath(self) -> Tuple[List[dict], List[dict]]:
        """
        Load dataset from two path files.
        Returns:
            tuple[list[dict], list[dict]]: dialogues and facts of each dialogues
        """

        with open(os.path.join(self.file_path, "train.json"), mode="r", encoding="utf-8") as f:
            train = json.load(f)

        with open(os.path.join(self.file_path, "valid_random_split.json"), mode="r", encoding="utf-8") as f:
            val_random_dts = json.load(f)

        with open(os.path.join(self.file_path, "valid_topic_split.json"), mode="r", encoding="utf-8") as f:
            val_topic_dts = json.load(f)

        with open(os.path.join(self.file_path, "test_random_split.json"), mode="r", encoding="utf-8") as f:
            test_random_dts = json.load(f)

        with open(os.path.join(self.file_path, "test_topic_split.json"), mode="r", encoding="utf-8") as f:
            test_topic_dts = json.load(f)

        return train, val_random_dts + val_topic_dts, test_random_dts + test_topic_dts


def parse_args() -> ArgumentParser:
    """ Get argument object and do sanity checks
    Returns:
          args (object): arugment object
    """
    args = ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve"
    )
    args.add_argument(
        "--dataset_file_path", type=str, default=None, help="Path to convs.txt"
    )
    args.add_argument(
        "--instruct_path", type=str, default=None, help="Path to instruction.txt"
    )
    args.add_argument(
        "--ratio_val", type=float, default=None, help="Ratio of validation dataset"
    )
    args.add_argument(
        "--ratio_test", type=float, default=None, help="Ratio of test dataset"
    )
    args.add_argument(
        "--window_context", type=int, default=None, help="Number of max_turn for dialogue history"
    )
    args.add_argument(
        "--save_path", type=str, default=None, help="Path to output folder for saving"
    )
    args.add_argument(
        "--log_file", type=str, default=None, help="path to log_file.txt"
    )

    args = args.parse_args()

    # sanity checks
    if args.save_path is None:
        raise ValueError(
            "You are running without save_path dir. "
            "You can set save_path dir using --save_path.")
    else:
        # check and create save_path directory if it doesn't already exist
        dir = Path(args.save_path)
        if not dir.is_dir():
            dir.mkdir(parents=True, exist_ok=True)

    if args.log_file is not None:
        # check and create log directory if it doesn't already exist
        dir = Path(args.log_file).parents[0]
        if not dir.is_dir():
            dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_file_path is None:
        raise ValueError(
            "You are running script without input data. "
            "Make sure you set all input data to --dataset_file_path")

    if args.ratio_val is not None and args.ratio_test is None:
        raise ValueError(
            "You are running script with specified ratio"
            "Make sure you set both input to --ratio_val and --ratio_test")
    elif args.ratio_test is None and args.ratio_test is not None:
        raise ValueError(
            "You are running script with specified ratio"
            "Make sure you set both input to --ratio_val and --ratio_test")

    if args.ratio_val is not None and args.ratio_test is not None:
        logger.warning("""You are running with specified ratio for validation dataset and test dataset..."
                        Converter will automatically get the total raw data - `data.json` file - which is available in the extracted data folder""")

    return args


def config_logger(log_file) -> None:
    """ Config logger
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(filename=log_file))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers
    )


def make_dataset():
    args = parse_args()
    config_logger(args.log_file)

    dataset_file_path = os.path.join(os.getcwd(), args.dataset_file_path)
    save_file_path = os.path.join(os.getcwd(), args.save_path)

    wow_obj = WoW_Converter(
        file_path=dataset_file_path,
        save_path=save_file_path,
        window_context=args.window_context,
        instruct_file_path=args.instruct_path
    )
    wow_obj.process(ratio_val=args.ratio_val,
                    ratio_test=args.ratio_test)
    logger.info("Done")


if __name__ == "__main__":
    # make_dataset()
    wow_obj = WoW_Converter(
        file_path="C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\\raw\WOW",
        save_path="C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\WOW",
        window_context=5,
        instruct_file_path="C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions\instruct_GradRes.txt"
    )
    wow_obj.process()
