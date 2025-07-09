set -x

# dataset
prompt_data=USPTO_50K_prompt_data
# model
pretrain=model_path
# save path
save_path=save_path

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --enable_prefix_caching \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0 1 \
   --eps_clip_low_high 0.2 0.3 \
   --pretrain ${pretrain} \
   --remote_rm_url examples/python/reward_func.py \
   --save_path ${save_path} \
   --micro_train_batch_size 8 \
   --train_batch_size 512 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 2048 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 5000000 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --prompt_data ${prompt_data} \
   --input_key input \
   --label_key label \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --flash_attn \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb "true"

# You could also try
#   --kl_estimator k2 \
