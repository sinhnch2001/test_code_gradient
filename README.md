# State Prediction
This repository is to predict the state of user turn in the dialogue

## Install
```shell
pip install -r requirements.txt
```

## Convert dataset 
```shell
python src/features/data_converter.py --ketod_dataset <path_of_ketod> --ketod_sample <out_path_of_ketod> --fusedchat_dataset <path_of_fused> --fused_sample <out_path_of_fused> --woi_dataset <path_of_woi> --woi_sample <out_path_of_woi> --schema_guided <path_of_schema> --context_window <window_context>


```
```commandline
usage: Dataset Converter [-h] [--schema_guided SCHEMA_GUIDED] [--context_window CONTEXT_WINDOW] [--ketod_dataset KETOD_DATASET] [--ketod_sample KETOD_SAMPLE] [--fusedchat_dataset FUSEDCHAT_DATASET]
                         [--fusedchat_sample FUSEDCHAT_SAMPLE] [--woi_dataset WOI_DATASET] [--woi_sample WOI_SAMPLE]                                                                                 
                                                                                                                                                                                                     
options:                                                                                                                                                                                             
  -h, --help            show this help message and exit                                                                                                                                              
  --schema_guided SCHEMA_GUIDED                                                                                                                                                                      
                        the path of schema guided                                                                                                                                                    
  --context_window CONTEXT_WINDOW                                                                                                                                                                    
                        the maximum number of utterance of a dialogue history                                                                                                                        
  --ketod_dataset KETOD_DATASET                                                                                                                                                                      
                        the path of ketod dataset                                                                                                                                                    
  --ketod_sample KETOD_SAMPLE                                                                                                                                                                        
                        the path of ketod sample (out file)                                                                                                                                          
  --fusedchat_dataset FUSEDCHAT_DATASET                                                                                                                                                              
                        the path of fusedchat dataset                                                                                                                                                
  --fusedchat_sample FUSEDCHAT_SAMPLE
                        the path of fusedchat sample (out file)
  --woi_dataset WOI_DATASET
                        the path of Wizard of internet dataset
  --woi_sample WOI_SAMPLE
                        the path of Wizard of internet sample (out file)

```

## Dowload link

Dataset after converted: https://drive.google.com/drive/folders/1QWqcj9vRSWrW77DJhUxndpFQFlrX2kAp



## Training
```commandline
usage: train.py [-h] [--output_dir OUTPUT_DIR] [--train_files TRAIN_FILES [TRAIN_FILES ...]] [--text_column TEXT_COLUMN] [--target_column TARGET_COLUMN] --val_files VAL_FILES [VAL_FILES ...] [--test_files TEST_FILES [TEST_FILES ...]]
                [--batch_size BATCH_SIZE] [--max_train_samples MAX_TRAIN_SAMPLES] [--max_eval_samples MAX_EVAL_SAMPLES] [--seed SEED] [--model_name MODEL_NAME] [--num_train_epochs NUM_TRAIN_EPOCHS]                                    
                [--max_target_length MAX_TARGET_LENGTH] [--num_beams NUM_BEAMS] [--mixed_precision MIXED_PRECISION] [--with_tracking] [--checkpointing_steps CHECKPOINTING_STEPS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]      
                [--do_eval_per_epoch] [--report_to REPORT_TO] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]                                                  
                [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]                                                                                                                      
                                                                                                                                                                                                                                         
options:                                                                                                                                                                                                                                 
  -h, --help            show this help message and exit                                                                                                                                                                                  
  --output_dir OUTPUT_DIR                                                                                                                                                                                                                
                        The output directory to save                                                                                                                                                                                     
  --train_files TRAIN_FILES [TRAIN_FILES ...]                                                                                                                                                                                            
                        Directory to train file (can be multiple files)                                                                                                                                                                  
  --text_column TEXT_COLUMN                                                                                                                                                                                                              
                        The name of the column in the datasets containing the full texts .                                                                                                                                               
  --target_column TARGET_COLUMN                                                                                                                                                                                                          
                        The name of the column in the label containing the full texts .                                                                                                                                                  
  --val_files VAL_FILES [VAL_FILES ...]                                                                                                                                                                                                  
                        Directory to validation file (can be multiple files)                                                                                                                                                             
  --test_files TEST_FILES [TEST_FILES ...]                                                                                                                                                                                               
                        Directory to test file (can be multiple files)                                                                                                                                                                   
  --batch_size BATCH_SIZE                                                                                                                                                                                                                
                        Batch size for the dataloader                                                                                                                                                                                    
  --max_train_samples MAX_TRAIN_SAMPLES
                        Number of training samples
  --max_eval_samples MAX_EVAL_SAMPLES
                        Number of validation samples
  --seed SEED           A seed for reproducible training.
  --model_name MODEL_NAME
                        Model name for fine-tuning
  --num_train_epochs NUM_TRAIN_EPOCHS
                        number training epochs
  --max_target_length MAX_TARGET_LENGTH
                        max length labels tokenize
  --num_beams NUM_BEAMS
                        number of beams
  --mixed_precision MIXED_PRECISION
                        Whether to use mixed precision. Choosebetween fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.
  --with_tracking       Whether to enable experiment trackers for logging.
  --checkpointing_steps CHECKPOINTING_STEPS
                        Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.(can be 'epoch' or int)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        If the training should continue from a checkpoint folder. (can be bool or string)
  --do_eval_per_epoch   Whether to run evaluate per epoch.
  --report_to REPORT_TO
                        The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`,mlflow, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.Only applicable    
                        when `--with_tracking` is passed.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup period) to use.
  --weight_decay WEIGHT_DECAY
                        Weight decay to use.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use.

```

**Please config to your system in the dir src/config**
```commandline
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: trainer.train     # Main training function containing accelerate instance
mixed_precision: 'fp16'
num_machines: 1
num_processes: 2    # The number of GPUs on the system
use_cpu: false
```

**Training with Fully sharded data parallel enabled**
```commandline
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: FSDP    # Distributed type:  NO — Not a distributed environment, just a single process.
                                             # MULTI_CPU — Distributed on multiple CPU nodes.
                                             # MULTI_GPU — Distributed on multiple GPUs.
                                             # DEEPSPEED — Using DeepSpeed.
                                             # TPU — Distributed on TPUs.
                                             # FSDP - Fully sharded data parallel
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: BartEncoderLayer,BartDecoderLayer     # Name of transformer blocks of the model to warp
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: trainer.train   # Main training function containing accelerate instance
mixed_precision: 'fp16'
num_machines: 1
num_processes: 2    # The number of GPUs on the system
use_cpu: false
```
```shell
bash run_training.sh
```
* **Please update your parameters in the run_training.sh file**
