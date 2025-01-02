from collections import defaultdict
import argparse
import os
import json

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--merged_output_name', help='Name of the file for storing merged results JSON.', required=True)

    return parser.parse_args()



# def make_merged_list_videomme(samples):
def make_merged_list_videomme(args):
    file = open(os.path.join(args.output_dir, args.output_name))
    samples = [eval(i.strip()) for i in file.readlines()]

    # A defaultdict to accumulate merged data
    merged_data = defaultdict(lambda: {
        'video_id': None,
        'duration': None,
        'domain': None,
        'sub_category': None,
        'questions': []
    })

    # Iterate over each sample and merge them based on video_id
    for sample in samples:
        video_id = sample['video_id']
        
        if not merged_data[video_id]['video_id']:  # If this video_id hasn't been initialized yet
            merged_data[video_id]['video_id'] = video_id
            merged_data[video_id]['duration'] = sample['duration']
            merged_data[video_id]['domain'] = sample['domain']
            merged_data[video_id]['sub_category'] = sample['sub_category']
        
        question_data = {
            'question_id': sample['question_id'],
            'task_type': sample['task_type'],
            'question': sample['question'],
            'options': sample['options'],
            'answer': sample['answer'],
            'response': sample.get('response', '')  # Assuming response can be missing
        }
        
        merged_data[video_id]['questions'].append(question_data)

    # Convert the defaultdict to a regular list of dicts
    merged_list = list(merged_data.values())
    return merged_list

if __name__ == "__main__":
    args = parse_args()
    merged_list = make_merged_list_videomme(args)
    with open(os.path.join(args.output_dir, args.merged_output_name), "w") as f:
        json.dump(merged_list, f)
    
    # samples = [
    #     {"video_id": "001",
    #     "duration": "short",
    #     "domain": "Knowledge",
    #     "sub_category": "Humanity & History",
    #     "question_id": "001-1",
    #     "task_type": "Counting Problem",
    #     "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?",
    #     "options": ["A. Apples.",
    #                 "B. Candles.",
    #                 "C. Berries.",
    #                 "D. The three kinds are of the same number."
    #             ],
    #     "answer": "C",
    #     "response": "C. Berries.",},
    #     {"video_id": "001",
    #     "duration": "short",
    #     "domain": "Knowledge",
    #     "sub_category": "Humanity & History",
    #             "question_id": "001-2",
    #             "task_type": "Information Synopsis",
    #             "question": "What is the genre of this video?",
    #             "options": [
    #                 "A. It is a news report that introduces the history behind Christmas decorations.",
    #                 "B. It is a documentary on the evolution of Christmas holiday recipes.",
    #                 "C. It is a travel vlog exploring Christmas markets around the world.",
    #                 "D. It is a tutorial on DIY Christmas ornament crafting."
    #             ],
    #             "answer": "A",
    #             "response": "D.",
    #         },
    #         {"video_id": "001",
    #     "duration": "short",
    #     "domain": "Knowledge",
    #     "sub_category": "Humanity & History",
    #             "question_id": "001-3",
    #             "task_type": "Counting Problem",
    #             "question": "How many red socks are above the fireplace at the end of this video?",
    #             "options": [
    #                 "A. 1.",
    #                 "B. 4.",
    #                 "C. 2.",
    #                 "D. 3."
    #             ],
    #             "answer": "D",
    #             "response": "D. 3",
    #         },
    #     {"video_id": "002",
    #     "duration": "short",
    #     "domain": "Knowledge",
    #     "sub_category": "Humanity & History",
    #             "question_id": "002-1",
    #             "task_type": "Object Recognition",
    #             "question": "Which of the following features/items is not discussed in the video in relation to the tomb?",
    #             "options": [
    #                 "A. Inkstone.",
    #                 "B. Niche.",
    #                 "C. Jade.",
    #                 "D. Sacrificial table."
    #             ],
    #             "answer": "C",
    #             "response": "Answer: C. Jade.",
    #         },
    # ]
    # merged_list = make_merged_list_videomme(samples)

    # print(merged_list)