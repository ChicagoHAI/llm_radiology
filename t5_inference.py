
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import gzip
import pandas as pd


def inference(instructions, examples, input_list, model_name, output_file):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    with open(output_file, "w") as fout:
        for input_sent in input_list:
            input_text = instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent)
            # print(input_text)
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_length=200, bos_token_id=0)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            fout.write(json.dumps({"input": input_sent, "output": result}) + "\n")

if __name__ == "__main__":
    test_file = "/data/dangnguyen/report_generation/mimic_data/mimic_train_impressions_sentence.csv"
    data = pd.read_csv(test_file, nrows=100)
    input_list = list(data["report"])
    instructions = open("instructions.txt").read()
    examples = open("examples.txt").read()
    output_file = "generated_sentence_t5_xxl.jsonl"
    model_name = "google/flan-t5-XXL"
    inference(instructions, examples, input_list, model_name, output_file)

