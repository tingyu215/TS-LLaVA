import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
from tqdm import tqdm
import torch
import math


from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# from llava.eval.model_utils import load_video
from prompt import get_multiple_choice_prompt, get_multiple_choice_prompt_onlychoice
from decord import VideoReader, cpu
from PIL import Image
import numpy as np


def get_index(bound, fps, max_frame, num_frms, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_frms
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_frms)
    ])
    return frame_indices


def load_video(video_path, num_frms, bound=None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    # max_frame = len(vr) - 1
    max_frame = len(vr)
    total_num_frm = min(max_frame, num_frms)
    fps = float(vr.get_avg_fps())
        
    frame_indices = get_index(bound, fps, max_frame, total_num_frm, first_idx=0) 
    img_array = vr.get_batch(frame_indices.tolist()).asnumpy()
    # images_group = list()
    # for frame_index in frame_indices:
    #     img = Image.fromarray(vr[frame_index].numpy())
    #     images_group.append(img)
    # torch_imgs = self.transform(images_group)
    # return torch_imgs
    original_size = (img_array.shape[-2], img_array.shape[-3])  # (width, height)
    original_sizes = (original_size,) * total_num_frm
    clip_imgs = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    
    return clip_imgs, original_sizes


def load_frame(video_path, num_frms, bound=None, fps=3):
    max_frame = len(os.listdir(video_path))
    total_num_frm = min(max_frame, num_frms)
    clip_imgs = list()
    frame_indices = get_index(bound, fps, max_frame, total_num_frm, first_idx=1) # frame_idx starts from 1
    for frame_index in frame_indices:
        img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
        clip_imgs.append(img)
    original_sizes = (clip_imgs[0].size,) * total_num_frm
    return clip_imgs, original_sizes



def llava_inference(
    video_frames,
    question,
    candidates,
    conv_mode,
    model,
    tokenizer,
    image_processor,
    image_sizes,
):
    # Get multiple choice prompt
    if args.use_onlychoice_prompt:
        prompt = get_multiple_choice_prompt_onlychoice(model, conv_mode, question, candidates)
    else:
        prompt = get_multiple_choice_prompt(model, conv_mode, question, candidates)

    # Get text inputs
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    # Get image inputs
    image_tensor = process_images(video_frames, image_processor, model.config)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    """
    Run inference on Video QA Dataset.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, rope_scaling_factor=args.rope_scaling_factor)

    # # Override image aspect ratio if needed
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio
    model.config.aggregation_method = args.aggregation_method
    model.config.num_sampled_tokens = args.num_sampled_tokens

    # Load questions and answers
    gt_file_qa = json.load(open(args.gt_file_qa, "r"))
    gt_file_qa = get_chunk(gt_file_qa, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Iterate over each sample in the ground truth file
    for index, sample in enumerate(tqdm(gt_file_qa)):
        video_name = sample["video"]
        question = sample["question"]
        candidates = sample["candidates"]
        answer = sample["answer"]

        sample_set = {
            "question": question,
            "candidates": candidates,
            "answer": answer,
        }

        if "start" in sample:
            bound = [sample["start"], sample["end"]]
        else:
            bound = None

        # Load video
        video_path = os.path.join(args.video_dir, video_name)
                
        if os.path.exists(video_path):
            try:
                if args.input_format == "video":
                    video_frames, sizes = load_video(video_path, num_frms=args.num_frames, bound=bound)
                else:
                    video_frames, sizes = load_frame(video_path, num_frms=args.num_frames, bound=bound)
            except Exception as e:
                print(f"Failed to load {video_path}, continue...")
                continue
            # if len(sizes) < 6:
            #     continue

            # Run inference on the video
            output = llava_inference(
                video_frames,
                question,
                candidates,
                args.conv_mode,
                model,
                tokenizer,
                image_processor,
                sizes,
            )
            output = output.replace("In the image", "In the video")
            # print(output)
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()

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
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file_qa", help="Path to the ground truth file containing question and answer.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--aggregation_method", type=str, default=None)
    parser.add_argument("--num_sampled_tokens", type=int, default=2880)
    parser.add_argument("--rope_scaling_factor", type=int, default=1)
    parser.add_argument("--use_onlychoice_prompt", action="store_true")
    parser.add_argument("--input_format", type=str, default="video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)