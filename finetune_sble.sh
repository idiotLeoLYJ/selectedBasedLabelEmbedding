export CUDA_VISIBLE_DEVICES=2

python ./selectedBasedLabelEmbedding.py \
  --model_type roberta \
  --model_name_or_path ./roberta-base \
  --task_name mnli \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir glue_data/MNLI \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 1 \
  --save_steps 10000 \
  --logging_steps 10000 \
  --num_train_epochs 3 \
  --output_dir output \
  --overwrite_output_dir \
  --evaluate_during_training 