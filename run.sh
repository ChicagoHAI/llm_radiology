#!/bin/bash

torchrun --nproc_per_node=4 --master_port=4001 finetune.py \
    --model_name_or_path /data/LLAMA_hf/llama-7b \
    --data_path /data/dangnguyen/report_generation/mimic_data/finetune_llm/finetune_imp_rec_10pc.json \
    --bf16 True \
    --output_dir /data/dangnguyen/report_generation/llm_radiology/radiology_models/7b_clean_rec/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True