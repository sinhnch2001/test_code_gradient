import os
import json
import random
import argparse
import logging
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from typing import (
    Tuple,
    List,
    Dict,
    Union,
    Optional
)
from src.features.converter import DialConverter

logger = logging.getLogger(__name__)


class ORQuAC_Converter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 instruct_file_path: str,
                 tag_speaker: str = 'USER',
                 tag_agent: str = 'AGENT',
                 style_tod: List[str] = ["politely"],
                 style_odd: List[str] = ["safety", "friendly"],
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
        self.style_tod = style_tod
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
                    history: List[str],
                    question: str) -> str:
        context = ""
        """
            Concat text with length for context
            Ex: context = <tag_speaker> .... <tag_agent> .... <tag_speaker>
        """
        for turn in history:
            context = context + self.tag_speaker + ": " + turn['question'] + " "
            context = context + self.tag_agent + ": " + turn['answer']['text'] + " "
        # add question of user to context
        context = context + self.tag_speaker + ": " + question + " "
        return context.strip()

    def get_knowledge(self, diag: dict) -> str:
        def index_of(val: int, in_list: list):
            try:
                return in_list.index(val)
            except ValueError:
                return -1

        idx_gold_passage = index_of(1, diag['retrieval_labels'])
        if idx_gold_passage < 0:
            return "no_passages_used"
        return diag['evidences'][idx_gold_passage]

    def _process_diag(self,
                      sample: dict) -> dict:
        processed = {}
        # Get instruction
        processed["instruction"] = self.get_instruction(self.list_instructions)
        # Get history
        if self.window_context is None:
            processed['context'] = self.get_context(history=sample['history'],
                                                    question=sample['rewrite'])
        elif len(sample['history']) <= self.window_context:
            processed['context'] = self.get_context(history=sample['history'],
                                                    question=sample['rewrite'])
        else:
            processed['context'] = self.get_context(history=sample['history'],
                                                    question=sample['rewrite'])
        # Get knowledge
        processed['documents'] = '' if self.get_knowledge(sample) == 'no_passages_used' else self.get_knowledge(sample)
        # Get style
        processed["style"] = self.style_odd[np.random.randint(0, len(self.style_odd))]
        # Get answer and style
        if sample['answer']['text'] == "NOTRECOVERED" or sample['answer']['text'] == "CANNOTANSWER":
            processed['response'] = "I don't know."
        else:
            processed['response'] = sample['answer']['text']
        # get sample template format
        processed['ontology'] = ''
        processed['system_action'] = ''

        return processed

    def process(self) -> None:
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
            self.train.append(processed_sample)

        self.val = []
        for each_sample in val:
            processed_sample = self._process_diag(each_sample)
            self.val.append(processed_sample)

        self.test = []
        for each_sample in test:
            processed_sample = self._process_diag(each_sample)
            self.test.append(processed_sample)

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

        with open(os.path.join(self.file_path, "train.txt"), "r", encoding="utf-8") as f:
            train = [json.loads(line.strip()) for line in f.readlines()]

        with open(os.path.join(self.file_path, "val.txt"), mode="r", encoding="utf-8") as f:
            val = [json.loads(line.strip()) for line in f.readlines()]

        with open(os.path.join(self.file_path, "test.txt"), mode="r", encoding="utf-8") as f:
            test = [json.loads(line.strip()) for line in f.readlines()]

        return train, val, test


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
        "--dataset_file_path", type=str, default=None, help="Path to folder contains raw dataset"
    )
    args.add_argument(
        "--instruct_path", type=str, default=None, help="Path to instruction.txt"
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

    if args.instruct_path is None:
        raise ValueError(
            "You are running script without specifying file `instruction.txt` "
            "Make sure you set input to --instruct_path")

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

    orquac_obj = ORQuAC_Converter(
        file_path=dataset_file_path,
        save_path=save_file_path,
        window_context=args.window_context,
        instruct_file_path=args.instruct_path
    )
    orquac_obj.process()
    logger.info("Done")


if __name__ == "__main__":
    # make_dataset()
    # python .\src\features\convert_GradRes\ORQuAC_Converter.py
    #     dataset_file_path 'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\\raw\ORQUAC'
    #     --instruct_path 'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions\instruct_GradRes.txt'
    #     --save_path 'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC'
    #     --window_context 5
    orquac_obj = ORQuAC_Converter(
        file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\\raw\ORQUAC',
        save_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC',
        window_context=5,
        instruct_file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions\instruct_GradRes.txt'
    )
    orquac_obj.process()