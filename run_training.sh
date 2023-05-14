CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/train.py 	\
  --num_train_epochs 10 \
	--output_dir "./output"  \
	--log_file "./logs" \
	--train_files  'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\FUSEDCHAT\\train.json' \
	               'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\KETOD\\train.json'\
                 'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC\\train.json'\
                 'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\WOW\\train.json' \
	--val_files 'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\FUSEDCHAT\\val.json' \
	            'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\KETOD\\val.json'\
              'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\ORQUAC\\val.json'\
              'C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\WOW\\val.json' \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
	--num_beams   4 \
	--weight_decay  0.3 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--with_tracking  \
	--report_to wandb \
	--checkpointing_steps epoch \
	--do_eval_per_epoch \
	--exp_name test