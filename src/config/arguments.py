from abc import abstractmethod, ABC
import argparse
import logging

from transformers import (
    MODEL_MAPPING,
    SchedulerType
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

USE_WANDB = True


class Arguments(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve"
        )
        self.init_model_args()
        self.init_data_args()
        self.init_training_args()
        self.init_inference_args()

    @abstractmethod
    def init_model_args(self):
        """Provide model's arguments
        """
        raise NotImplemented

    @abstractmethod
    def init_data_args(self):
        """Provide model's arguments
        """
        raise NotImplemented

    @abstractmethod
    def init_training_args(self):
        """Provide training's arguments
        """
        raise NotImplemented

    @abstractmethod
    def init_inference_args(self):
        """Provide inference's arguments"""

    def parse(self):
        args = self.parser.parse_args()
        return args


class ModuleArguments(Arguments):
    def __init__(self) -> None:
        super().__init__()
        self.init_model_args()
        self.init_data_args()
        self.init_training_args()
        self.init_inference_args()

    def init_model_args(self) -> None:
        """Initialize model's arguments
        """

        self.parser.add_argument(
            "--model_name_or_path",
            type=str,
            default=None,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            required=True,
        )

        self.parser.add_argument(
            "--config_name",
            type=str,
            default=None,
            help="Pretrained config name or path if not the same as model_name",
        )

        self.parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )

        self.parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If passed, will use a slow tokenizer (not backed by the 馃 Tokenizers library).",
        )

        self.parser.add_argument(
            "--model_type",
            type=str,
            default=None,
            help="Model type to use if training from scratch",
            choices=MODEL_TYPES
        )

    def init_data_args(self) -> None:
        """Initialize data's arguments
        """
        self.parser.add_argument(
            "--train_files",
            nargs='+',
            required=True,
            help="Path to csv or a json file containing the training data."
        )

        self.parser.add_argument(
            "--val_files",
            nargs='+',
            required=True,
            help="Path to csv or a json file containing the validation data."
        )

        self.parser.add_argument(
            "--test_files",
            nargs='+',
            help="Path to csv or a json file containing the test data."
        )

        self.parser.add_argument(
            "--source_prefix",
            type=str,
            default="Instruction: ",
            help="A prefix to add before every source text " "(useful for T5 models).",
        )

        self.parser.add_argument(
            '--text_column',
            type=str,
            default='prompt',
            help="The name of the column in the datasets containing the full texts .")

        self.parser.add_argument(
            '--target_column',
            type=str,
            default='output',
            help="The name of the column in the label containing the full texts .")

        self.parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=1,
            help="The number of processes to use for the preprocessing.",
        )

        self.parser.add_argument(
            '--max_train_samples',
            type=int,
            default=None,
            help="Number of training samples")

        self.parser.add_argument(
            '--max_eval_samples',
            type=int,
            default=None,
            help="Number of validation samples")

        self.parser.add_argument(
            "--max_source_length",
            type=int,
            default=512,
            help="The maximum total input sequence length after "
                 "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
        )

        self.parser.add_argument(
            "--max_target_length",
            type=int,
            default=80,
            help="The maximum total sequence length for target text after "
                 "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
                 "during ``evaluate`` and ``predict``.",
        )

        self.parser.add_argument(
            "--val_max_target_length",
            type=int,
            default=None,
            help="The maximum total sequence length for validation "
                 "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
                 "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
                 "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
        )

        self.parser.add_argument(
            "--pad_to_max_length",
            type=bool,
            default=True,
            help="do padding"
        )

    def init_training_args(self) -> None:
        """Initialize training arguments
        """

        self.parser.add_argument(
            "--per_device_train_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the training dataloader.",
        )

        self.parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=4,
            help="Batch size (per device) for the evaluation dataloader.",
        )

        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-5,
            help="Initial learning rate (after the potential warmup period) to use.",
        )

        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay to use.")

        self.parser.add_argument(
            "--num_train_epochs",
            type=int,
            default=3,
            help="Total number of training epochs to perform.")

        self.parser.add_argument('--checkpointing_steps',
                                 type=str,
                                 default=None,
                                 help="Whether the various states should be saved at the end of every n steps,"
                                      " or 'epoch' for each epoch.(can be 'epoch' or int)")

        self.parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
        )

        self.parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )

        self.parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default="linear",
            help="The scheduler type to use.",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )

        self.parser.add_argument(
            "--num_warmup_steps",
            type=int,
            default=0,
            help="Number of steps for the warmup in the lr scheduler."
        )

        self.parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Where to store the final model.")

        self.parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="A seed for reproducible training.")

        self.parser.add_argument(
            "--ignore_pad_token_for_loss",
            type=bool,
            default=True,
            help="do padding"
        )
        self.parser.add_argument(
            '--logging_steps',
            type=int,
            default=5000,
            help='log intermdiate results of training process every <logging_steps>')

        self.parser.add_argument(
            '--log_file',
            type=str,
            default=None,
            help='path to logging file txt which store logged information')

        self.parser.add_argument(
            "--save_steps",
            type=int,
            default=5000,
            help="Whether the various states should be saved at the end of every n steps"
        )

        self.parser.add_argument(
            "--with_tracking",
            action="store_true",
            help="Whether to enable experiment trackers for logging.",
        )

        self.parser.add_argument(
            "--report_to",
            type=str,
            default="all",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
                "Only applicable when `--with_tracking` is passed.")
        )

        self.parser.add_argument(
            "--exp_name",
            type=str,
            help="Description to the experiment",
            default='exp',
        )

        self.parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="Path to a checkpoint folder if the training should be continue from",
        )

        self.parser.add_argument(
            '--do_eval_per_epoch',
            action='store_true',
            help="Whether to run evaluate per epoch.")

    def init_inference_args(self) -> None:
        """Initialize inference arguments
        """
        self.parser.add_argument(
            "--num_beams",
            type=int,
            default=1,
            help="Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``",
        )

        self.parser.add_argument(
            "--max_length",
            type=int,
            default=128,
            help="max length"
        )

        self.parser.add_argument(
            "--min_length",
            type=int,
            default=30,
            help="min length"
        )

        self.parser.add_argument(
            "--top_p",
            type=float,
            default=0.8,
            help="Top-p sampling"
        )

        self.parser.add_argument(
            "--top_k",
            type=int,
            default=128,
            help="Top-k sampling"
        )

        self.parser.add_argument(
            "--repetition_penalty",
            type=float,
            default=1.1,
            help="Repetition penalty not to repeat token when sampling"
        )

        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Temperature to sharpen probabili"
        )

        self.parser.add_argument(
            "--no_repeat_ngram_size",
            type=int,
            default=3,
            help="avoid repeating n-gram when sampling"
        )
        self.parser.add_argument(
            "--num_return_sequences",
            type=int,
            default=1,
            help="How many answers would be generated?"
        )


class ConverterArguments():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.init_general_args()

    def init_general_args(self) -> None:
        """Initialize general converter's arguments
        """
        self.parser.add_argument(
            '--instructions_path',
            type=str,
            default=None,
            help='path to instructions.txt')
        self.parser.add_argument(
            '--user_tag',
            type=str,
            default='USER: ',
            help='special tag for user turn')
        self.parser.add_argument(
            '--system_tag',
            type=str,
            default='SYSTEM: ',
            help='special tag for system turn')
        self.parser.add_argument(
            '--domain_tag',
            type=str,
            default='DOMAIN: ',
            help='special tag for domain in state_of_user section')
        self.parser.add_argument(
            '--action_tag',
            type=str,
            default='',
            help='special tag for system action in state_of_user section')
        self.parser.add_argument(
            '--knowledge_tag',
            type=str,
            default='KNOWLEDGE: ',
            help='special tag for knowledge in state_of_user section')

        self.parser.add_argument(
            '--final_output_dir',
            type=str,
            default=None,
            help='path to final output data folder')
        self.parser.add_argument(
            '--test_ratio',
            type=float,
            default=0.005,
            help='ratio of splitting test set')
        self.parser.add_argument(
            '--valid_ratio',
            type=float,
            default=0.005,
            help='ratio of splitting valid set')
        self.parser.add_argument(
            '--num_of_utterances',
            type=int,
            default=5,
            help='number of utterances in context')

    def init_bst_args(self) -> None:
        """Initialize bst converter's arguments
        """
        self.parser.add_argument(
            "--bst_input_dir",
            type=str,
            default=None,
            help='path to BST data folder')
        self.parser.add_argument(
            '--bst_save_path',
            type=str,
            default=None,
            help='path to save BST converter\'s output data')

    def init_fc_args(self) -> None:
        """Initialize fc converter's arguments
        """
        self.parser.add_argument(
            "--fc_input_dir",
            type=str,
            default=None,
            help='path to FC data folder')
        self.parser.add_argument(
            '--fc_save_path',
            type=str,
            default=None,
            help='path to save FC converter\'s output data')
        self.parser.add_argument(
            '--fc_conversion_guide_path',
            type=str,
            default=None,
            help='path to json file which have slot conversion guide for fused chat')

    def init_kt_args(self) -> None:
        """Initialize kt converter's arguments
        """
        self.parser.add_argument(
            "--kt_input_dir",
            type=str,
            default=None,
            help='path to KT data folder')
        self.parser.add_argument(
            '--kt_save_path',
            type=str,
            default=None,
            help='path to save KT converter\'s output data')
        self.parser.add_argument(
            '--kt_domain_path',
            type=str,
            default=None,
            help='path to domains map dict file')
        self.parser.add_argument(
            '--kt_slot_path',
            type=str,
            default=None,
            help='path to slots map dict file')

    def parse(self) -> object:
        args = self.parser.parse_args()
        return args