from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


def get_option_prompt(candidates, version="default"):
    option_prompt = ""
    options = []
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        if version == "v4":
            option_prompt += f"({choice}) {candidate}\n"
        else:
            option_prompt += f"({choice}):{candidate} "
        options.append(choice)
    options = "(" + ",".join(options) + ")"
    return option_prompt, options


def get_multiple_choice_prompt(model, conv_mode, question, candidates):
    if conv_mode == "multiple_choice_allvideo_v4":
        prompt = "You are a helpful expert in video analysis. Select the best option to answer the question. USER: <image>\nThe input consists of a sequence of key frames from a video.\nQuestion: %s\nOptions:\n%sOnly give the best option. \nASSISTANT:\nAnswer: Best option:("
        option_prompt, options = get_option_prompt(candidates, version="v4")
        prompt = prompt % (question, option_prompt)
    elif conv_mode == "multiple_choice_allvideo_34b_v4":
        prompt = "<|im_start|>system\n You are a helpful expert in video analysis. Select the best option to answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe input consists of a sequence of key frames from a video. Question: %s\nOptions:\n%sOnly give the best option. <|im_end|>\n<|im_start|>assistant\nAnswer: Best option:("
        option_prompt, options = get_option_prompt(candidates, version="v4")
        prompt = prompt % (question, option_prompt)
    elif conv_mode == "multiple_choice_allvideo_v5":
        prompt = "You are a helpful expert in video analysis. Select the best option to answer the question. USER: <image>\nThe input consists of an image arranging key frames from a video in a grid view, along with a sequence of key frames from the video.\nQuestion: %s\nOptions:\n%sOnly give the best option. \nASSISTANT:\nAnswer: Best option:("
        option_prompt, options = get_option_prompt(candidates, version="v4")
        prompt = prompt % (question, option_prompt)
    elif conv_mode == "multiple_choice_allvideo_34b_v5":
        prompt = "<|im_start|>system\n You are a helpful expert in video analysis. Select the best option to answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe input consists of an image arranging key frames from a video in a grid view, along with a sequence of key frames from the video. Question: %s\nOptions:\n%sOnly give the best option. <|im_end|>\n<|im_start|>assistant\nAnswer: Best option:("
        option_prompt, options = get_option_prompt(candidates, version="v4")
        prompt = prompt % (question, option_prompt)
    else:
        raise ValueError(f"Unknown conv_mode: {conv_mode}")
    return prompt


def get_prompt(model, conv_mode, question):
    if conv_mode == "image_seq_v3":
        prompt = "USER: <image>\nThe input consists of a sequence of key frames from a video. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the video. Question: %s \nASSISTANT:\nAnswer: In the video,"
        prompt = prompt % question
    elif conv_mode == "image_seq_34b_v3":
        prompt = "<|im_start|>system\n Answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe input consists of a sequence of key frames from a video. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the video. Question: %s <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"
        prompt = prompt % question
    elif conv_mode == "image_seq_v2":
        prompt = "You are a helpful expert in video analysis. USER: <image>\nThe input consists of a sequence of key frames from a video. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the video. Question: %s \nASSISTANT:\nAnswer: In the video,"
        prompt = prompt % question
    elif conv_mode == "image_seq_34b_v2":
        prompt = "<|im_start|>system\n You are a helpful expert in video analysis. Answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe input consists of a sequence of key frames from a video. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the video. Question: %s <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"
        prompt = prompt % question
    else:
        if model.config.mm_use_im_start_end:
            ques = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            ques = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], ques)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt



def get_option_prompt_onlychoice(candidates, version="default"):
    option_prompt = ""
    options = []
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        if version == "v4":
            option_prompt += f"{choice}. {candidate}\n"
        else:
            option_prompt += f"{choice}:{candidate} "
        options.append(choice)
    options = "(" + ",".join(options) + ")"
    return option_prompt, options


def get_multiple_choice_prompt_onlychoice(model, conv_mode, question, candidates):
    # give one choice as answer only
    # e.g. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nSelect the best answer to the following multiple-choice question based on the video. Respond with only the letter (A,B,C,D,E) of the correct option. \nQuestion: how do the two man play the instrument\nOptions:\nA. roll the handle\nB. tap their feet\nC. strum the string\nD. hit with sticks\nE. pat with hand\nOnly give the best option. \nAnswer with the option's letter from the given choices directly. ASSISTANT:
    if conv_mode == "multiple_choice_allvideo_v4":
        # vicuna-v1: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        option_prompt, options = get_option_prompt_onlychoice(candidates, version="v4")

        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nSelect the best answer to the following multiple-choice question based on the video. Respond with only the letter {options} of the correct option. \nQuestion: {question}\nOptions:\n{option_prompt}Only give the best option. \nAnswer with the option's letter from the given choices directly. ASSISTANT:"
        
        # prompt = prompt % (options, question, option_prompt)
    elif conv_mode == "multiple_choice_allvideo_34b_v4":
        option_prompt, options = get_option_prompt_onlychoice(candidates, version="v4")

        prompt = f"<|im_start|>system\n A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. <|im_end|>\n<|im_start|>user\n <image>\nSelect the best answer to the following multiple-choice question based on the video. Respond with only the letter {options} of the correct option.\nQuestion: {question}\nOptions:\n{option_prompt}Only give the best option. <|im_end|>\n<|im_start|>assistant\nAnswer with the option's letter from the given choices directly. ASSISTANT:"
        
        # prompt = prompt % (options, question, option_prompt)
    else:
        raise ValueError(f"Unknown conv_mode: {conv_mode}")
    return prompt