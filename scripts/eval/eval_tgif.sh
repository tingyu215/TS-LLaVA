gpt_version="gpt-3.5-turbo-0125"
num_tasks=25

CKPT_NAME="llava-v1.6-7b"
aggregation_method=$1
num_frames=$2
num_sampled_tokens=$3
prompt_version=$4
image_aspect_ratio=$5
api_key=$6
output_dir="./TS-LLaVA/output/TGIF_Zero_Shot_QA_TS/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}"
output_file=${output_dir}/merge.jsonl

python3 ./TS-LLaVA/scripts/eval/eval_video_qa.py \
    --pred_path ${output_file} \
    --output_dir ${output_dir}/${gpt_version} \
    --output_json ${output_dir}/results.json \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks} \
    --api_key ${api_key}