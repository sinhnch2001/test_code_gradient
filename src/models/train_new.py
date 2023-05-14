import sys
import os
import argparse
import logging
from pathlib import Path
sys.path.insert(0,r'./') #Add root directory here

from src.config.arguments import ModuleArguments
from src.data.dataloader_GradRes import GradResDataLoader
from training_loop_new import Trainer,setup_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def parse_args():
    """ Get argument object and do sanity checks

    Returns:
          args (object): Godel arugment object
    """
    args = ModuleArguments()
    args = args.parse()

    # sanity checks
    if args.output_dir is None:
        raise ValueError(
            "You are running training script without output dir. "
            "You can set output dir using --output_dir.")
    else:
        # check and create output directory if it doesn't already exist
        dir = Path(args.output_dir)
        if not dir.is_dir():
            dir.mkdir(parents=True, exist_ok=True)

    if args.log_file is not None:
        # check and create log directory if it doesn't already exist
        dir = Path(args.log_file).parents[0]
        if not dir.is_dir():
            dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from_checkpoint is not None:
        if not Path(args.resume_from_checkpoint).is_dir():
            raise ValueError("resume_from_checkpoint is not a folder")

    if args.train_files is None or args.val_files is None:
        raise ValueError(
            "You are running training script without input data. "
            "Make sure you set all input data to --train_files, --val_files")
    else:
        if args.train_files is not None:
            for file in args.train_files:
                extension = file.split(".")[-1]
                assert extension in ["json"], "`train_files` should be a json file."
        if args.val_files is not None:
            for file in args.val_files:
                extension = file.split(".")[-1]
                assert extension in ["json"], "`val_files` should be a json file."
        if args.test_files is not None:
            for file in args.test_files:
                extension = file.split(".")[-1]
                assert extension in ["json"], "`test_files` should be a json file."

    # validate and convert the input argument
    try:
        args.checkpointing_steps = int(args.checkpointing_steps)  # try to convert to int
    except:
        args.checkpointing_steps = args.checkpointing_steps  # if conversion fails, assume it's a string

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

def main():
    args = parse_args()

    # config logger
    config_logger(args.log_file)

    logger.info('START A NEW TRAINING')

    # log log_file
    if args.log_file is not None:
        logger.info("Store logged information in {}".format(args.log_file))

    # log source_prefix
    logger.info('Running model {} with source prefix: "{}"'.format(args.model_name_or_path, args.source_prefix))

    # load pretrained model and tokenizer
    config, model, tokenizer = setup_training(config_name=args.config_name,
                                              model_name_or_path=args.model_name_or_path,
                                              tokenizer_name=args.tokenizer_name,
                                              model_type=args.model_type,
                                              use_slow=args.use_slow_tokenizer)
    dataloader_args = {
        "tokenizer": tokenizer,
        "text_column": args.text_column,
        "target_column": args.target_column,
        "train_file": args.train_files,
        "val_file": args.val_files,
        "train_batch_size": args.per_device_train_batch_size,
        "val_batch_size":args.per_device_eval_batch_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "source_prefix": args.source_prefix,
        "seed": args.seed
    }
    dataloaders = GradResDataLoader(**dataloader_args)

    trainer_args = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "training_args":args,
        "dataloaders": dataloaders
    }
    trainer = Trainer(**trainer_args)
    trainer.train()

if __name__ == "__main__":
    main()