import argparse
from tqdm import tqdm
import numpy as np
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default=r"", help="The path to file containing prediction.")
    parser.add_argument("--save_dir", default=r"", help="The path to save results")
    parser.add_argument("--gt_dir", default="./playground/gt_qa_files/NExTQA/val_qa_full.json", help="The path to save results")
    args = parser.parse_args()
    return args


def map_prediction_to_option(pred):
    pred_option = "none"
    if isinstance(pred, str):
        prediction_letter = pred[0]
        if prediction_letter in "abcdefABCDEF":
            pred_option = prediction_letter.lower()
        if "answer is " in pred:
            pred = pred[pred.index("answer is"):]
        if "A:" in pred or "A)" in pred:
            pred_option = "a"
        elif "B:" in pred or "B)" in pred:
            pred_option = "b"
        elif "C:" in pred or "C)" in pred:
            pred_option = "c"
        elif "D:" in pred or "D)" in pred:
            pred_option = "d"
        elif "E:" in pred or "E)" in pred:
            pred_option = "e"
        elif "F:" in pred or "F)" in pred:
            pred_option = "f"
    return pred_option


def check_ans(pred, gt):
    flag = False

    pred_option = map_prediction_to_option(pred)

    if pred_option not in "abcde":
        print(f"Model does not follow the instruction: {pred}")
    elif pred_option == gt.lower():
        flag = True

    return flag


def rearrange_annotation_dict(anno_dict):
    new_dict = {}
    for sample in anno_dict:
        new_dict[sample["question_id"]] = {}
        new_dict[sample["question_id"]]["question_type"] = sample["question_type"]
        new_dict[sample["question_id"]]["question"] = sample["question"]
        new_dict[sample["question_id"]]["answer"] = sample["answer"]
    return new_dict


def main():
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    with open(args.gt_dir, "r") as f:
        full_annotation = json.load(f)
    
    full_anno_dict = rearrange_annotation_dict(full_annotation)

    for new_pred_content in new_pred_contents:
        new_pred_content["question_type"] = full_anno_dict[new_pred_content["id"]]["question_type"]


    task_accuracy = {}
    for new_pred_content in tqdm(new_pred_contents):
        task_name = new_pred_content["task_name"]
        if task_name not in task_accuracy:
            task_accuracy[task_name] = {
                "yes_count": 0,
                "no_count": 0,
                "accuracy": 0,
            }

        gt = chr(ord("a") + new_pred_content["answer_number"])
        if check_ans(
            pred=new_pred_content["pred"],
            gt=gt,
        ):
            task_accuracy[task_name]["yes_count"] += 1
        else:
            task_accuracy[task_name]["no_count"] += 1
    
    ### question_type
    for new_pred_content in tqdm(new_pred_contents):
        question_type = new_pred_content["question_type"]
        if question_type not in task_accuracy:
            task_accuracy[question_type] = {
                "yes_count": 0,
                "no_count": 0,
                "accuracy": 0,
            }
        gt = chr(ord("a") + new_pred_content["answer_number"])
        if check_ans(
            pred=new_pred_content["pred"],
            gt=gt,
        ):
            task_accuracy[question_type]["yes_count"] += 1
        else:
            task_accuracy[question_type]["no_count"] += 1
    
    task_accuracy["Causal"] = {"yes_count": 0, "no_count": 0,"accuracy": 0,}
    task_accuracy["Descriptive"] = {"yes_count": 0, "no_count": 0,"accuracy": 0,}
    task_accuracy["Temporal"] = {"yes_count": 0, "no_count": 0,"accuracy": 0,}
    
    for task_stat in task_accuracy:
        if task_stat.startswith("C") and task_stat!="Causal":
            task_accuracy["Causal"]["yes_count"] += task_accuracy[task_stat]["yes_count"]
            task_accuracy["Causal"]["no_count"] += task_accuracy[task_stat]["no_count"]
        elif task_stat.startswith("D") and task_stat!="Descriptive":
            task_accuracy["Descriptive"]["yes_count"] += task_accuracy[task_stat]["yes_count"]
            task_accuracy["Descriptive"]["no_count"] += task_accuracy[task_stat]["no_count"]
        elif task_stat.startswith("T") and task_stat!="Temporal":
            task_accuracy["Temporal"]["yes_count"] += task_accuracy[task_stat]["yes_count"]
            task_accuracy["Temporal"]["no_count"] += task_accuracy[task_stat]["no_count"]
        else:
            continue

    

    accuracy_list = []
    for task_name in task_accuracy:
        yes_count = task_accuracy[task_name]["yes_count"]
        no_count = task_accuracy[task_name]["no_count"]
        accuracy = yes_count / (yes_count + no_count)
        task_accuracy[task_name]["accuracy"] = accuracy
        accuracy_list.append(accuracy)
        print(task_name)
        print("\tYes count:", yes_count)
        print("\tNo count:", no_count)
        print("\tAccuracy:", accuracy)
    # if len(accuracy_list) > 1:
    #     print("Average accuracy:", np.mean(accuracy_list))

    #     task_accuracy["overall"] = {
    #         "accuracy": np.mean(accuracy_list),
    #     }

    with open(os.path.join(args.save_dir, "results_categories.json"), "w") as f:
        json.dump(task_accuracy, f)


if __name__ == "__main__":
    main()