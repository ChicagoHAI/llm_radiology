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

def clean(instructions, examples, input_list, pipe):
    # output_list = []
    # for input_sent in input_list:
    #     input_text = instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent)
    #     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    #     outputs = model.generate(input_ids, max_length=200, bos_token_id=0)
    #     result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     output_list.append(result)
    # return output_list

    BATCH_SIZE = 128
    num_batches = len(input_list) // BATCH_SIZE
    prompts = [instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent) for input_sent in input_list]

    output_batches = []
    for i in range(num_batches+1):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        batch = prompts[start:end]
        output = pipe(batch)
        output_batches += output

    return output_batches

if __name__ == "__main__":
    pipe = pipeline("text2text-generation", model="google/flan-t5-XXL", device='cuda:0')
    pipe.model = deepspeed.init_inference(
        pipe.model,
        mp_size=1,
        dtype=torch.float,
        injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')},
    )

    test_file = "/data/dangnguyen/report_generation/mimic_data/report_cleaning/test_cleaning_gt.csv"
    outpath = '/data/dangnguyen/report_generation/mimic_data/report_cleaning/test_deepspeed.csv'

    # test_file = "/data/dangnguyen/report_generation/mimic_data/report_cleaning/rewrite6_intermediate.csv"
    # outpath = "/data/dangnguyen/report_generation/mimic_data/report_cleaning/rewrite6_intermediate.csv"

    data = pd.read_csv(test_file)
    print(len(data))
    input_list = list(data["report"])
    # input_list = list(data["llm_rewritten"])

    NUM_RULES = 7
    for i in range(NUM_RULES):
    # for i in range(6, 7):
        rule = 'rewrite' + str(i + 1)
        print(rule)
        instruct_path = '/data/dangnguyen/report_generation/llm_radiology/prompts/report_clean_rules/{}_instructions.txt'.format(rule)
        examples_path = '/data/dangnguyen/report_generation/llm_radiology/prompts/report_clean_rules/{}_sen_fewshot.txt'.format(rule)

        instructions = open(instruct_path).read()
        examples = open(examples_path).read()

        output_list = clean(instructions, examples, input_list, pipe)
        assert len(output_list) == len(input_list)
        input_list = output_list

        # outputting intermediate results
        tmp_path = '/data/dangnguyen/report_generation/mimic_data/report_cleaning/{}_intermediate.csv'.format(rule)
        data_tmp = data.copy()
        data_tmp['llm_rewritten'] = output_list
        data_tmp.to_csv(tmp_path, index=False)

    data['llm_rewritten'] = output_list
    data.to_csv(outpath, index=False)