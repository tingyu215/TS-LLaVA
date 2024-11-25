CKPT_NAME="llava-v1.6-7b"
aggregation_method=$1
num_frames=$2
num_sampled_tokens=$3
prompt_version=$4
image_aspect_ratio=$5
model_path="./TS-LLaVA/ckpt/llava-v1.6-7b"
video_dir="BASE_VIDEO_DIR/Activitynet_Zero_Shot_QA/all_test"
GT_QA_DIR="./TS-LLaVA/playground/gt_qa_files"
output_dir="./TS-LLaVA/output/Activitynet_Zero_Shot_QA_TS/${CKPT_NAME}_u${num_frames}FRS_${aggregation_method}_${num_sampled_tokens}tokens_${prompt_version}_${image_aspect_ratio}"
CONV_MODE="image_seq_${prompt_version}"

# CONV_MODE=${CONV_MODE:-"image_seq_v3"}

################################# Run ##################################

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 ./TS-LLaVA/scripts/infer_video/run_inference_video_qa.py \
      --video_dir ${video_dir} \
      --gt_file_question ${GT_QA_DIR}/Activitynet_Zero_Shot_QA/test_q.json \
      --gt_file_answers ${GT_QA_DIR}/Activitynet_Zero_Shot_QA/test_a.json \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --model_path ${model_path} \
      --conv_mode ${CONV_MODE} \
      --num_chunks ${CHUNKS} \
      --chunk_idx ${IDX} \
      --num_frames ${num_frames} \
      --temperature 0 \
      --aggregation_method ${aggregation_method} \
      --num_sampled_tokens ${num_sampled_tokens} \
      --image_aspect_ratio ${image_aspect_ratio} &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "${output_file}"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "${output_file}"
done

