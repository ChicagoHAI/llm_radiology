from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import gzip
import pandas as pd

def clean(instructions, examples, input_list, model, tokenizer):
    output_list = []
    for input_sent in input_list:
        input_text = instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_length=200, bos_token_id=0)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_list.append(result)
    return output_list
            

if __name__ == "__main__":    
    model_name = "google/flan-t5-XXL"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    test_file = "/data/dangnguyen/report_generation/mimic_data/finetune_llm/test_gt_cleaning.csv"
    data = pd.read_csv(test_file)
    print(len(data))
    input_list = list(data["original"])

    outpath = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/test_cleaned_2.csv'

    NUM_RULES = 7
    for i in range(NUM_RULES):
        rule = 'rewrite' + str(i + 1)
        print(rule)
        instruct_path = '/data/dangnguyen/report_generation/XrayGPT/prompts/mimic/report_clean_rules/{}_instructions.txt'.format(rule)
        examples_path = '/data/dangnguyen/report_generation/XrayGPT/prompts/mimic/report_clean_rules/{}_sen_fewshot.txt'.format(rule)
        instructions = open(instruct_path).read()
        examples = open(examples_path).read()

        output_list = clean(instructions, examples, input_list, model, tokenizer)
        assert len(output_list) == len(input_list)
        input_list = output_list

    data['llm_rewritten'] = output_list
    data.to_csv(outpath, index=False)
        
