CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /kaggle/working/test_code_gradient/src/config/config_fsdp_t5.yaml /kaggle/working/test_code_gradient/src/models/train_new.py 	\
  --num_train_epochs 2 \
  --model_name_or_path "google/flan-t5-base" \
	--output_dir "/kaggle/working/"  \
	--log_file "./logs" \
	--train_files "/kaggle/input/data-test-gradient/test.json" "/kaggle/input/data-test-gradient/test - Copy.json"\
	--val_files "/kaggle/input/data-test-gradient/val.json" "/kaggle/input/data-test-gradient/val - Copy.json"\
	--test_files "/kaggle/input/data-test-gradient/test.json" "/kaggle/input/data-test-gradient/test - Copy.json"\
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
	--num_beams   4 \
	--weight_decay  0.3 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--with_tracking  \
	--report_to mlflow \
	--checkpointing_steps epoch \
	--do_eval_per_epoch \
	--exp_name test 
#	--max_train_samples 100 \
#  --max_eval_samples 50