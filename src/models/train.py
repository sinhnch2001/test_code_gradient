import sys
import os
import argparse
sys.path.insert(0,r'./') #Add root directory here

from src.data.dataloader_GradRes import DataLoader_test
from training_loop import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--output_dir', type=str, help="The output directory to save")
    parser.add_argument('--train_files', nargs='+', help= "Directory to train file (can be multiple files)")
    parser.add_argument('--text_column', type=str, default='prompt',
                        help="The name of the column in the datasets containing the full texts .")
    parser.add_argument('--target_column', type=str, default='output',
                        help="The name of the column in the label containing the full texts .")
    parser.add_argument('--val_files', nargs='+',required=True,
                        help= "Directory to validation file (can be multiple files)")
    parser.add_argument('--test_files', nargs='+',
                        help= "Directory to test file (can be multiple files)")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for the dataloader")
    parser.add_argument('--max_train_samples', type=int, default=None, help="Number of training samples")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="Number of validation samples")
    parser.add_argument('--seed', type=int, default=42, help="A seed for reproducible training.")

    # Training
    parser.add_argument('--model_name', type=str, default="prakharz/DIAL-BART0",
                        help ="Model name for fine-tuning")
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help="number training epochs")
    parser.add_argument('--max_target_length', type=int, default=60,
                        help="max length labels tokenize")
    parser.add_argument('--num_beams', type=int, default=4,
                        help="number of beams")
    parser.add_argument('--with_tracking', action='store_true',
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument('--checkpointing_steps', type=str, default=None,
                        help="Whether the various states should be saved at the end of every n steps,"
                             " or 'epoch' for each epoch.(can be 'epoch' or int)")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help="If the training should continue from a checkpoint folder. (can be bool or string)")
    parser.add_argument('--do_eval_per_epoch', action='store_true',
                        help="Whether to run evaluate per epoch.")
    parser.add_argument('--report_to', type=str, default='wandb',help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`,'"mlflow"', `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ))

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--weight_decay', type=float, default=0.3,
                        help="Weight decay to use.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)

    args = parser.parse_args(args)

    # validate and convert the input argument
    try:
        args.checkpointing_steps = int(args.checkpointing_steps)  # try to convert to int
    except:
        args.checkpointing_steps = args.checkpointing_steps  # if conversion fails, assume it's a string

    return args


def main(args):

    args = parse_args(args)
    dataloader_args = {
        "model_name": args.model_name,
        "train_file": args.train_files,
        "val_file": args.val_files,
        "test_file": args.test_files,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "seed": args.seed
    }
    dataloaders = DataLoader_test(**dataloader_args)

    trainer_args = {
        "model_name_or_path": args.model_name,
        "output_dir": args.output_dir,
        "dataloaders": dataloaders,
        "lr_scheduler_type": args.lr_scheduler_type,
        "checkpointing_steps": args.checkpointing_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "seed": args.seed,
        "with_tracking": args.with_tracking,
        "report_to": args.report_to,
        "num_train_epochs": args.num_train_epochs,
        "max_target_length": args.max_target_length,
        "num_beams": args.num_beams,
        "weight_decay": args.weight_decay,
        "per_device_batch_size":dataloaders.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "do_eval_per_epoch": args.do_eval_per_epoch,
        "learning_rate": args.learning_rate
    }
    trainer = Trainer(**trainer_args)
    trainer.train()


if __name__ == "__main__":
    # main(sys.argv[1:])
    dataloader_args = {
        "model_name": 'google/flan-t5-small',
        "train_file": [
            'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\FUSEDCHAT\\train.json',
            'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\KETOD\\train.json',
            'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC\\train.json',
            'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\WOW\\train.json'],
        "val_file": ['C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\FUSEDCHAT\\val.json',
                     'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\KETOD\\val.json',
                     'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC\\val.json',
                     'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\WOW\\val.json'],
        "test_file": ['C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\FUSEDCHAT\\test.json',
                      'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\KETOD\\test.json',
                      'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC\\test.json',
                      'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\WOW\\test.json'],
        "batch_size": 4
    }
    dataloaders = DataLoader_test(**dataloader_args)

    trainer_args = {
        "model_name_or_path": 'google/flan-t5-small',
        "output_dir": "C:\ALL\OJT\gradients.baselinev1.dialogstate\reports",
        "dataloaders": dataloaders,
        "checkpointing_steps": "epoch",
        "report_to": "mlflow",
        "num_train_epochs": 10,
        "num_beams": 4,
        "weight_decay": 0.3,
        "per_device_batch_size": dataloaders.batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5
    }
    trainer = Trainer(**trainer_args)
    trainer.train()