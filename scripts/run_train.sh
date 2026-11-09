# 数据与模型
dataset_path="/path/to/json"
output_dir="outputs/portraits"
pretrained_model_name_or_path="nvidia/difix_ref"

# GPU（单机；设为 1 使用单卡）
num_gpus="8"

# 训练
learning_rate="2e-5"
max_train_steps="100000"
resolution="512"
train_batch_size="1"
gradient_accumulation_steps="1"
mixed_precision="no"
dataloader_num_workers="8"

# 扩散与损失
timestep="199"
lambda_l2="1.0"
lambda_lpips="1.0"
lambda_gram="0.0"

# 验证、保存与日志
checkpointing_steps="10000"
eval_freq="10000"
num_samples_eval="100"
viz_freq="5000"
report_to="local"
tracker_run_name="portraits"

launch_args=()
if [ "$num_gpus" -gt 1 ]; then
    launch_args+=(--multi_gpu)
fi

accelerate launch \
    "${launch_args[@]}" \
    --num_machines 1 \
    --num_processes "$num_gpus" \
    --mixed_precision "$mixed_precision" \
    src/train_difix.py \
    --dataset_path "$dataset_path" \
    --output_dir "$output_dir" \
    --pretrained_model_name_or_path "$pretrained_model_name_or_path" \
    --learning_rate "$learning_rate" \
    --max_train_steps "$max_train_steps" \
    --resolution "$resolution" \
    --train_batch_size "$train_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --mixed_precision "$mixed_precision" \
    --mv_unet \
    --enable_xformers_memory_efficient_attention \
    --timestep "$timestep" \
    --lambda_l2 "$lambda_l2" \
    --lambda_lpips "$lambda_lpips" \
    --lambda_gram "$lambda_gram" \
    --checkpointing_steps "$checkpointing_steps" \
    --eval_freq "$eval_freq" \
    --num_samples_eval "$num_samples_eval" \
    --viz_freq "$viz_freq" \
    --dataloader_num_workers "$dataloader_num_workers" \
    --report_to "$report_to" \
    --tracker_run_name "$tracker_run_name"
