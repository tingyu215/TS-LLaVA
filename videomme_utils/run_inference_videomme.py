import sys
import os

# Add the parent directory of 'llava' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import os
import argparse
import json

from tqdm import tqdm
from llava.eval.model_utils import load_video

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import torch
import time

def llava_inference(video_frames, question, conv_mode, model, tokenizer, image_processor, image_sizes):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = process_images(video_frames, image_processor, model.config)  

    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=16,
                use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_qa', help='Path to the ground truth file containing question and answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--conv_mode", type=str, required=False, default='vicuna_v1')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)

    parser.add_argument("--aggregation_method", type=str, default=None)
    parser.add_argument("--post_prompt", type=str)

    return parser.parse_args()


def videomme_doc_to_text(doc, post_prompt=None):
    # option_prompt = f"This video's subtitles are listed below:\n{subtitles}\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    if post_prompt == "v1":
        post_prompt = "Answer with the option's letter from the given choices directly."
    elif post_prompt == "v2":
        post_prompt = "The best answer is:"
    else:
        raise NotImplementedError
    # post_prompt = post_prompt if post_prompt else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def run_inference(args):
    """
    Run inference on Video QA DataSetå.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # import ipdb
    # ipdb.set_trace()
    model.config.aggregation_method = args.aggregation_method


    gt_file_qa = json.load(open(args.gt_file_qa, "r"))
    gt_file_qa = get_chunk(gt_file_qa, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_file_qa):
        vid_name = sample['videoID']
        question = videomme_doc_to_text(sample, args.post_prompt)

        sample_set = {'video_id': sample["video_id"],
        'question_id': sample["question_id"],
        'duration': sample["duration"], 
        'question': sample["question"], 
        'answer': sample['answer'], 
        'domain': sample["domain"], 
        'sub_category': sample["sub_category"], 
        'task_type': sample["task_type"], 
        'options': sample["options"] }

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{vid_name}{fmt}")
                
            if os.path.exists(temp_path):
                # print(f'processing {idx}/{len(gt_questions)}')
                video_path = temp_path
                video_frames, sizes = load_video(video_path, num_frm=args.num_frames)                
                # Run inference on the video and add the output to the list
                output = llava_inference(video_frames, question, conv_mode, model,
                                                tokenizer, image_processor, sizes)
                # print(output)
                sample_set['response'] = output
                # output_list.append(sample_set)
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
