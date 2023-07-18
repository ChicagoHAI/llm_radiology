torchrun --nproc_per_node=4 --master_port=4001 train.py \
    --model_name_or_path /data/LLAMA_hf/llama-7b \
    --data_path /data/chenghao/stanford_alpaca/alpaca_data.json \
    --bf16 True \
    --output_dir /data/alpaca_our_version_debug \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "/data/chenghao/stanford_alpaca/configs/default_offload_opt_param.json" \
    --tf32 True
