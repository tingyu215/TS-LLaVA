gpt_version="gpt-3.5-turbo-0125"
num_tasks=25

CKPT_NAME="llava-v1.6-7b"
aggregation_method=$1
num_frames=$2
num_sampled_tokens=$3
prompt_version=$4
image_aspect_ratio=$5
api_key=$6
output_dir="./TS-LLaVA/output/Video_ChatGPT"


echo "=================================="
echo "eval Consistency..."
python3 ./TS-LLaVA/eval/eval_generative_qa.py \
		--pred_path ${output_dir}/consistency/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/merge.jsonl \
		--output_dir ${output_dir}/consistency/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/${gpt_version} \
		--output_json ${output_dir}/consistency/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/results_consistency.json \
		--gpt_version ${gpt_version} \
		--num_tasks ${num_tasks} \
        --api_key ${api_key} \
		--prompt_mode consistency


echo "=================================="
echo "eval Correctness of Information..."

python3 ./TS-LLaVA/eval/eval_generative_qa.py \
    --pred_path ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/merge.jsonl \
    --output_dir ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/${gpt_version}/correctness \
    --output_json ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/results_correctness.json \
	--gpt_version ${gpt_version} \
	--num_tasks ${num_tasks} \
    --api_key ${api_key} \
	--prompt_mode correctness

echo "=================================="
echo "eval Detail Orientation..."

python3 ./TS-LLaVA/eval/eval_generative_qa.py \
    --pred_path ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/merge.jsonl \
    --output_dir ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/${gpt_version}/detailed_orientation \
    --output_json ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/results_detailed_orientation.json \
	--gpt_version ${gpt_version} \
	--num_tasks ${num_tasks} \
    --api_key ${api_key} \
	--prompt_mode detailed_orientation

echo "=================================="
echo "eval Contextual Understanding..."

python3 ./TS-LLaVA/eval/eval_generative_qa.py \
    --pred_path ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/merge.jsonl \
    --output_dir ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/${gpt_version}/contextual \
    --output_json ${output_dir}/generic/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/results_contextual.json \
	--gpt_version ${gpt_version} \
	--num_tasks ${num_tasks} \
    --api_key ${api_key} \
	--prompt_mode context


echo "=================================="
echo "eval Temporal Understanding..."

python3 ./TS-LLaVA/eval/eval_generative_qa.py \
    --pred_path ${output_dir}/temporal/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/merge.jsonl \
    --output_dir ${output_dir}/temporal/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/${gpt_version} \
    --output_json ${output_dir}/temporal/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}/results_temporal_understanding.json \
	--gpt_version ${gpt_version} \
	--num_tasks ${num_tasks} \
    --api_key ${api_key} \
	--prompt_mode temporal

