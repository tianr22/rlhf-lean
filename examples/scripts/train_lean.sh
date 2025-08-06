set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.2 \
   --init_kl_coef 1e-3 \
   --flash_attn \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain /home/fit/wanghn/WORK/liguchan/models/kimina-7b \
   --agent_func_path /home/fit/wanghn/WORK/tianr/rlhf-lean/examples/python/agent_func.py \
   --save_path /home/fit/wanghn/WORK/liguchan/checkpoints/rlhf \
   --ckpt_path /home/fit/wanghn/WORK/liguchan/checkpoints/rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 1 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --generate_max_len 12288 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data /home/fit/wanghn/WORK/tianr/prompts.jsonl \
   --input_key prompt \
   --normalize_reward \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --max_rounds 2 \
   --ring_attn_size 4 \

# You could also try
#   --kl_estimator k2 \