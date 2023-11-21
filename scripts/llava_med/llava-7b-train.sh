
#!/bin/bash

deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path FlagAlpha/Llama2-Chinese-7b-Chat\
    --version  llava_llama_2 \
    --data_path medical_image_dataset/test_image_text.json \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type linear \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 False \
    --fp16 True \
    --bits 16 \
    --output_dir ./checkpoints/llava-7b-train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


python -m llava.serve.cli --model-path ./checkpoints/llava-7b-train

