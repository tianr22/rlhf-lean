set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm_lean \
   --save_path /home/fit/wanghn/WORK/tianr/models/reward/kimina-7b \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 16 \
   --gradient_checkpointing \
   --pretrain /home/fit/wanghn/WORK/tianr/models/kimina-7b-expert/expert \
   --bf16 \
   --max_epochs 1 \
   --max_len 16384 \
   --zero_stage 3 \
   --dataset /home/fit/wanghn/WORK/tianr/reward_data.jsonl \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --grad_accum_dtype float32 \
   --flash_attn \
   --packing_samples \
   --compute_fp32_loss
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands 2>&1 | tee train_rm_lean_kimina.log
fi
