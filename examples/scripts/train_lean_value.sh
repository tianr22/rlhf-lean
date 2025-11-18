set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path /data2/tianrui/rlhf/models/value_0_9_wo_01 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --pretrain /data2/liguchan/Lean/prover/local/models/expert \
   --bf16 \
   --max_epochs 1 \
   --max_len 16384 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset /data2/tianrui/lean/value_data_train.jsonl \
   --eval_dataset /data2/tianrui/lean/value_data_eval.jsonl \
   --prompt_key prompt \
   --value_key value \
   --flash_attn \
   --packing_samples \
   --adam_offload \
   --compute_fp32_loss
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


deepspeed --module $training_commands 2>&1 | tee train_lean_value.log

