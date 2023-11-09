import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
from transformers import pipeline
import deepspeed

import json
import gzip
import pandas as pd
import numpy as np
import torch
import math
import os

import sys
sys.path.append('/data/dangnguyen/report_generation/report-generation')
from CXRMetric.CheXbert.src.label import label

CHEXBERT_PATH = '/data/dangnguyen/report_generation/models/chexbert.pth'
DEVICE = int(sys.argv[1])
PARTITION = DEVICE
# TOTAL_EXAMPLES = 71989
TOTAL_EXAMPLES = 160
NUM_GPUS = 4

workload = math.ceil(TOTAL_EXAMPLES / NUM_GPUS)
# inpath = "/data/dangnguyen/report_generation/mimic_data/report_cleaning/train_gt_imp_sen_72k_uniq.csv"
# outpath = '/data/dangnguyen/report_generation/mimic_data/report_cleaning/clean_72k/cleaned_frag_{}.csv'.format(PARTITION)

inpath = "/data/dangnguyen/report_generation/mimic_data/report_cleaning/test_cleaning_gt_200.csv"
outpath = '/data/dangnguyen/report_generation/mimic_data/report_cleaning/test_flan_frag_{}.csv'.format(PARTITION)

# gt_pos_path = './gt_pos_labels_72k.pt'
# gt_neg_path = './gt_neg_labels_72k.pt'

gt_pos_path = './gt_pos_labels_200.pt'
gt_neg_path = './gt_neg_labels_200.pt'

torch.cuda.set_device(DEVICE)
print('Currently using CUDA: {}'.format(DEVICE))

# Cleans reports according to rules
def clean(instructions, examples, input_list, pipe):
    BATCH_SIZE = 64
    num_batches = len(input_list) // BATCH_SIZE + 1
    prompts = [instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent) for input_sent in input_list]

    output_batches = []
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        batch = prompts[start:end]
        output = pipe(batch, max_length=200)
        output_batches += output

        print('Completed batch: {}/{}'.format(i+1, num_batches))

    output = [example['generated_text'] for example in output_batches]
    return output

# Checks which examples' labels have been changed due to cleaning
def labels_changed(y_gt, y_gt_neg, output_list):
    df_gen = pd.DataFrame(output_list, columns=['report'])
    df_gen = df_gen.replace('REMOVED', '_')

    pre_chexb_path = './gen_pre_chexbert_{}.csv'.format(PARTITION)

    df_gen.to_csv(pre_chexb_path, index=False)
    y_gen = label(CHEXBERT_PATH, pre_chexb_path, use_gpu=False)
    y_gen = np.array(y_gen).T

    y_gen_neg = y_gen.copy()
    y_gen[(y_gen == 2) | (y_gen == 3)] = 0
    y_gen_neg[(y_gen_neg == 1) | (y_gen_neg == 3)] = 0
    y_gen_neg[y_gen_neg == 2] = 1

    pos_is_diff = np.logical_xor(y_gt, y_gen).any(axis=1)
    neg_is_diff = np.logical_xor(y_gt_neg, y_gen_neg).any(axis=1)
    print(pos_is_diff)
    print(neg_is_diff)
    assert len(pos_is_diff) == len(output_list)

    label_is_diff = np.logical_or(pos_is_diff, neg_is_diff)
    return label_is_diff

if __name__ == "__main__":
    start = PARTITION * workload
    end = (PARTITION + 1) * workload
    print('Data partition: [{}, {}]'.format(start, end))

    # label heuristic
    if not (os.path.exists(gt_pos_path) and os.path.exists(gt_neg_path)):
        data_full = pd.read_csv(inpath).fillna('_')
        df_gt = data_full[['report']] # make sure the CSV file has a column named 'report'
        df_gt.to_csv('./gt_pre_chexbert.csv', index=False) # temporary file for the purpose of labeling. Will be deleted.
        y_gt = label(CHEXBERT_PATH, './gt_pre_chexbert.csv', use_gpu=True)
        y_gt = np.array(y_gt).T

        # Note on labels:
        # 0: unmentioned ; 1: positive ; 2: negative ; 3: uncertain
        y_gt_neg = y_gt.copy()
        y_gt[(y_gt == 2) | (y_gt == 3)] = 0
        y_gt_neg[(y_gt_neg == 1) | (y_gt_neg == 3)] = 0
        y_gt_neg[y_gt_neg == 2] = 1

        torch.save(y_gt, gt_pos_path)
        torch.save(y_gt_neg, gt_neg_path)

    y_gt = torch.load(gt_pos_path)[start:end]
    y_gt_neg = torch.load(gt_neg_path)[start:end]

    pipe = pipeline("text2text-generation", model="google/flan-t5-XXL", device=DEVICE)
    pipe.model = deepspeed.init_inference(
        pipe.model,
        mp_size=1,
        dtype=torch.float,
        injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')},
    )

    data = pd.read_csv(inpath)[start:end].fillna('')
    input_list = list(data["report"])

    RULES = [1, 2, 3, 4, 5, 6, 7]
    # RULES = [1, 5, 6, 7, 2, 3, 4]
    for i in RULES:
        rule = 'rewrite' + str(i)
        print(rule)
        instruct_path = '/data/dangnguyen/report_generation/llm_radiology/prompts/report_clean_rules/{}_instructions.txt'.format(rule)
        examples_path = '/data/dangnguyen/report_generation/llm_radiology/prompts/report_clean_rules/{}_sen_fewshot.txt'.format(rule)

        instructions = open(instruct_path).read()
        examples = open(examples_path).read()

        output_list = clean(instructions, examples, input_list, pipe)
        assert len(output_list) == len(input_list)
        
        # checking if cleaning has changed the labels
        label_is_diff = labels_changed(y_gt, y_gt_neg, output_list)
        for j in range(len(label_is_diff)):
            if label_is_diff[j]:
                output_list[j] = input_list[j] # we revert back to the input if the output changes a label

        input_list = output_list

        # outputting intermediate results
        tmp_path = '/data/dangnguyen/report_generation/mimic_data/report_cleaning/clean_test/test_clean_frag_{}_rule_{}.csv'.format(PARTITION, rule)
        data_tmp = data.copy()
        data_tmp['llm_rewritten'] = output_list
        data_tmp.to_csv(tmp_path, index=False)

    data['llm_rewritten'] = output_list
    data.to_csv(outpath, index=False)