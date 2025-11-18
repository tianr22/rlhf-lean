#!/bin/bash

deepspeed --module openrlhf.cli.train_sft \
   --max_len 40960 \
   --dataset /data2/liguchan/Lean/prover/local/results/goedel_sft_train.jsonl \
   --input_key question \
   --output_key response \
   --input_template $'User: {}\nAssistant: ' \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain /data2/liguchan/Lean/prover/local/models/Goedel-Prover-V2-8B \
   --save_path /data2/tianrui/checkpoints/goedel/sft_new/\
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --packing_samples \
   --grad_accum_dtype float32 \
   --bf16 \
   --learning_rate 5e-6 \

# Support HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}
