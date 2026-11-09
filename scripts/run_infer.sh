# 模型
model_name="nvidia/difix_ref"
model_path="/path/to/pkl"

# 输入与输出
input_image="/path/to/input"
ref_image="/path/to/ref"
output_dir="outputs/inference"

# 推理
height="512"
width="512"
timestep="199"
prompt="remove degradation"

python src/inference_difix.py \
    --model_name "$model_name" \
    --model_path "$model_path" \
    --input_image "$input_image" \
    --ref_image "$ref_image" \
    --output_dir "$output_dir" \
    --height "$height" \
    --width "$width" \
    --timestep "$timestep" \
    --prompt "$prompt"
