import sys

import transformers
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

sys.path.append('/data/dangnguyen/report_generation/report-generation/')
from CXRMetric.run_eval import calc_metric

cxr_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
        'Pneumothorax', 'Support Devices']

# converts a label vector to English
def labels_to_eng(labels):
    diag = ''
    for i in range(len(labels)):
        label = labels[i]
        cond = cxr_labels[i]
        if label == 1:
            diag += cond
            diag += ', '
    return diag

def format_input(df):
    inputs = []
    for _, row in df.iterrows():
        ind = row['indication']
        labels = labels_to_eng(row[cxr_labels])[:-2]
        inp = 'Indication: {}\nPositive labels: {}'.format(ind, labels)
        inputs.append(inp)
    return inputs

def write_report(instructions, input_list, model, tokenizer):
    # BATCH_SIZE = 32
    # num_batches = len(input_list) // BATCH_SIZE + 1
    # prompts = [instructions.format(input=input_example) for input_example in input_list]
    # output = []
    # for i in range(num_batches):
    #     start = i * BATCH_SIZE
    #     end = (i + 1) * BATCH_SIZE
    #     batch = prompts[start:end]
    #     if len(batch) == 0:
    #         continue
    #     with torch.no_grad():
    #         input_tokens = tokenizer(batch, padding=True, return_tensors="pt")
    #         generate_ids = model.generate(input_tokens['input_ids'].to('cuda:0'), max_length=200)
    #         output_batch = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    #         output_cleaned = [example.split('Response:')[1] for example in output_batch]
    #         output += output_cleaned
    #     print('Completed batch: {}/{}'.format(i+1, num_batches))
    # return output

    prompts = [instructions.format(input=input_example) for input_example in input_list]
    output = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        with torch.no_grad():
            input_tokens = tokenizer(prompt, return_tensors="pt")
            generate_ids = model.generate(input_tokens['input_ids'].to('cuda:0'), max_length=200)
            output_batch = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            output_cleaned = [example.split('Response:')[1] for example in output_batch]
            output += output_cleaned
        print('Completed example: {}/{}'.format(i+1, len(prompts)))
    return output

if __name__ == "__main__":
    xray_llama = transformers.LlamaForCausalLM.from_pretrained('./radiology_models/xray_llama_7b/').to('cuda:0')
    llama_tokenizer = transformers.LlamaTokenizer.from_pretrained('./radiology_models/xray_llama_7b/')

    data = pd.read_csv('/data/dangnguyen/report_generation/mimic_data/test_ind_imp_chexb.csv').fillna('')
    input_list = format_input(data)

    instruct_path = '/data/dangnguyen/report_generation/llm_radiology/prompts/report_writing/instructions.txt'
    instructions = open(instruct_path).read()

    output_list = write_report(instructions, input_list, xray_llama, llama_tokenizer)

    data['generated'] = output_list
    data[['study_id','report','generated']].to_csv('/data/dangnguyen/report_generation/mimic_data/finetune_llm/test_gen_use-gt_imp.csv', index=False)
