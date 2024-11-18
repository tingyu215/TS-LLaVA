# TS-LLaVA

**Full Readme and code will be available soon.**

This is the official implementation for [TS-LLaVA: Constructing Visual Tokens through Thumbnail-and-Sampling for Training-Free Video Large Language Models]()

by [Tingyu Qu](https://tingyu215.github.io), [Mingxiao Li](https://mingxiao-li.github.io), [Tinne Tuytelaars](https://www.esat.kuleuven.be/psi/TT), [Marie-Francine Moens](https://people.cs.kuleuven.be/~sien.moens/).


![](figures/method.png)


We explore various visual tokens compression strategies. Our TS-LLaVA achieves the state-of-the-art performance among trianing-free video LLMs.


## Installation

To create conda env, please run:

    conda env create -n llava --file environment.yml
    conda activate llava

The checkpoints for LLaVA-v1.6 can be found here:
    
    git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b MODEL_PATH/llava-v1.6-vicuna-7b
    git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b MODEL_PATH/llava-v1.6-34b

\[Optional] To enable GPT evaluation for open-ended video QA, please do the following:

    export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY

## Multiple Choice VideoQA Data


1. We prepare the ground-truth question and answer files based on [`IG-VLM`](https://github.com/imagegridworth/IG-VLM/tree/main) and [`SF-LLaVA`](https://github.com/apple/ml-slowfast-llava/tree/main), and put them under [playground/gt_qa_files](playground/gt_qa_files).

    - **NExT-QA**:  Download the `NExT_QA.csv` from [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/NExT_QA.csv)
    - **EgoSchema**: Download the `EgoSchema.csv` from [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/EgoSchema.csv)
    - **IntentQA**: Download the `IntentQA.csv` from [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/IntentQA.csv)
    
    If you want to run our model for Open-Ended VideoQA and video-based Text Generation, please download the datasets as:
    - **MSVD-QA**: Download the `MSVD_QA.csv` from [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/MSVD_QA.csv)
    - **MSRVTT-QA**: Download the `MSRVTT_QA.csv` from [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/MSRVTT_QA.csv)
    - **TGIF-QA**: Download the `TGIF_FrameQA.csv` from [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/TGIF_FrameQA.csv)
    - **Activitynet-QA**: Download the `Activitynet_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/ActivityNet_QA.csv)
    - **VCGBench**
        - Download all files under [`text_generation_benchmark`](https://github.com/imagegridworth/IG-VLM/blob/main/data/text_generation_benchmark)
        - Reformat the files by running
            ```
            python scripts/data/prepare_vcgbench_qa_file.py --qa_folder $TEXT_GENERATION_BENCHMARK
            ```
2. Reformatting the files:
    - After getting the csv files, please reformat the files (apart from VCGBench) by running
        ```
        python scripts/data/prepare_{DATASET}_file.py --qa_file $PATH_TO_CSV_FILE
        ```
    - replace DATASET with the names of the dataset. Check the ``scripts/data`` to make sure the name is correct.

3. Download the raw videos from the official websites.

    - Multiple Choice VideoQA

        - Download datasets from the data owners.
            - [`NExT-QA`](https://github.com/doc-doc/NExT-QA)
            - [`EgoSchema`](https://egoschema.github.io)
            - [`IntentQA`](https://github.com/JoseponLee/IntentQA)

    - Open-Ended VideoQA & video-based Text Generation:

        - [Recomanded] Option 1: Follow the instruction in [`Video-LLaVA`](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) to download raw videos.
        - Option 2: Download videos from the data owners.
            - [`MSVD-QA`](https://github.com/xudejing/video-question-answering?tab=readme-ov-file)
            - [`MSRVTT-QA`](https://github.com/xudejing/video-question-answering?tab=readme-ov-file)
            - [`TGIF-QA`](https://github.com/YunseokJANG/tgif-qa?tab=readme-ov-file)
            - [`ActivityNet-QA`](https://github.com/MILVLG/activitynet-qa)



## Multitask Benchmarks

1. Download the data:
    - MVBench
        - Download the data from [`here`](https://huggingface.co/datasets/OpenGVLab/MVBench)
        - The official repo can be found [`here`](https://github.com/OpenGVLab/Ask-Anything)

    - MLVU
        - Download the data from [`here`](https://huggingface.co/datasets/MLVU/MVLU)
        - The official repo can be found [`here`](https://github.com/JUNJIE99/MLVU)
