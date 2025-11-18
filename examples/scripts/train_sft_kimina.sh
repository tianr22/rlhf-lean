#!/bin/bash

deepspeed --module openrlhf.cli.train_sft \
   --max_len 40960 \
   --dataset /home/fit/wanghn/WORK/tianr/kimina_SFT_train_compiler_inside.jsonl \
   --flash_attn \
   --input_key prompt \
   --output_key response \
   --apply_chat_template \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain /home/fit/wanghn/WORK/checkpoints/Kimina-Prover-Distill-8B \
   --save_path /home/fit/wanghn/WORK/tianr/models/sft/kimina \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --ds_tensor_parallel_size 4 \
   --ring_attn_size 4 \
   --max_epochs 1 \
   --packing_samples \
   --grad_accum_dtype float32 \
   --bf16 \
   --learning_rate 5e-6 \
   --need_append_tokens \

# Support HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}
