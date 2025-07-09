#!/bin/bash

data_path=cold_start_data
model_path=continual_pretrain_model_path
save_path=save_path

deepspeed --module openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset ${data_path} \
   --input_key input \
   --output_key output \
   --apply_chat_template \
   --train_batch_size 128 \
   --micro_train_batch_size 16 \
   --pretrain ${model_path} \
   --save_path ${save_path} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 3 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --gradient_checkpointing \
   --use_wandb "true"

# Support HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}

# Support RingAttention
# pip install ring_flash_attn
#   --ring_attn_size 2 \
#   --ring_head_stride 2 \

# Can also be used for continued pre-training
# --pretrain_mode