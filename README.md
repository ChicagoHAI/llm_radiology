# Codebase for Pragmatic Radiology Report Generation

We expose the following functionalities discussed in the paper
* Report cleaning with large language models
* Training an image classifier for detecting positive findings
* Finetuning LLaMA on predicted conditions and indications
* Generating reports with Pragmatic-LLaMA
* Evaluating generated reports

This code mainly works for [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) [1], but can be adapted slightly to work with other chest X-ray datasets. We will discuss the relevant changes in each section. Make sure to obtain the license if you are working with MIMIC-CXR.

TODO: update requirements.txt

## Obtaining the relevant data and models

The input to our model is a tuple of (image, indication). Please follow the relevant dataset instructions to obtain them, especially the indication section. For MIMIC-CXR, you can use [create_section_files.py](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt) to extract the indication, findings, and impression sections.

Our code uses CheXbert [2] to label reports and RadGraph [3] for evaluation. Please download the two model checkpoints [here](https://github.com/rajpurkarlab/CXR-Report-Metric) and put them in "./models/".

## Report cleaning with large language models

```
deepspeed --num_gpus=<insert> report_cleaning.py --chexbert_path <insert> --dataset_path <insert> --output_dir <insert>
```

This cleaning method works best for one sentence at a time. The process to split sentences and keep track of their report IDs can be specific to the dataset, so we leave that implementation to the user.

## Finetuning LLaMA on predicted conditions and indications

We first format the indication and groundtruth CheXbert labels into a JSON file that can be used to finetune LLaMA.

```
python format_llama_input.py --indication_path <insert> --impression_path <insert> --outpath <insert>
```

Follow [Alpaca's](https://github.com/tatsu-lab/stanford_alpaca) finetuning instructions to finetune a LLaMA model to generate radiology reports. Put the path to the above JSON file for --data_path.

## Generating reports with Pragmatic-LLaMA

Insert the path to your finetuned Pragmatic-LLaMA model, path to indications, path to the directory containing the vision model and tuned classification thresholds, and specify an output path for predicted vision labels. This helps save time on subsequent runs on the same images by not having to re-run the classifier.

```
python pragmatic_llama_inference.py --llama_path <insert> --indication_path <insert> --vision_path <insert> --image_path <insert> --vision_out_path <insert> --outpath <insert>
```

## Evaluating generated reports

Once you have generated reports using Pragmatic-LLaMA or any other model, they can be evaluate using the command below. Note that --out_path should be a CSV file. If it ends with "filename.csv", there will also be an output file that ends with "filename_avg.csv" that contains the average (over all evaluated reports) scores of metrics. filename.csv itself saves the scores per-report. We graciously borrow much of the evaluation code from [Yu, Endo, and Krishnan et al.](https://github.com/rajpurkarlab/CXR-Report-Metric/blob/main/CXRMetric/run_eval.py) [4]

```
python evaluate.py --gt_path <insert> --gen_path <insert> --out_path <insert>
```

[1] Johnson, Alistair, Pollard, Tom, Mark, Roger, Berkowitz, Seth, and Steven Horng. "MIMIC-CXR Database" (version 2.0.0). PhysioNet (2019). https://doi.org/10.13026/C2JT1Q.

[2] Smit, Akshay, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y. Ng, and Matthew P. Lungren. "CheXbert: combining automatic labelers and expert annotations for accurate radiology report labeling using BERT." arXiv preprint arXiv:2004.09167 (2020).

[3] Jain, Saahil, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon et al. "Radgraph: Extracting clinical entities and relations from radiology reports." arXiv preprint arXiv:2106.14463 (2021).

[4] Yu, Feiyang, Mark Endo, Rayan Krishnan, Ian Pan, Andy Tsai, Eduardo Pontes Reis, Eduardo Kaiser Ururahy Nunes Fonseca et al. "Evaluating progress in automatic chest x-ray radiology report generation." Patterns 4, no. 9 (2023).