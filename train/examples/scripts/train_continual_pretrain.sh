#!/bin/bash

data_path=data_path
model_path=model_path
save_path=save_path

deepspeed --module openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset ${data_path} \
   --input_key input \
   --output_key output \
   --apply_chat_template \
   --train_batch_size 1024 \
   --micro_train_batch_size 32 \
   --pretrain ${model_path} \
   --save_path ${save_path} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
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