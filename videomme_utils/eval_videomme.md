## Introduction

* This document provides a brief guide for evaluating our model on the VideoMME dataset. A more detailed documentation may be updated in the future as time allows.

* Please note that the code is not yet fully cleaned and is taken directly from our old repository. If you encounter any errors, kindly inform us, and we will address them accordingly.

* Alternatively, you can evaluate VideoMME using [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). To do this, replace llava_arch.py in their codebase, along with making a few minor corrections, to ensure compatibility.


## Download data

* To download the dataset, please visit https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main

* Store the dataset in either the ``playground`` folder or the ``ckpt`` folder.

* Run the following command to convert the annotations to JSON format: ``python convert_parquet_to_json.py``


## Inference and evaluation

* Inference:
    * Use the same bash script from multiple-choice VideoQA inference (``run_inference_videomme.py``) 
    * For reference, check the scripts ``run_inference_multiple_choice_qa.py`` and ``run_qa_nextqa.sh`` for details
* Pos-inference proccessing: 
    * After obtaining the .jsonl file, run ``merge_outputs.py`` to convert it into JSON format.
    * Alternatively, you can directly work with the .jsonl file by modifying ``eval_your_results.py`` as needed.
* Evaluation:
    * Run ``eval_your_results.py`` to evaluate the results.