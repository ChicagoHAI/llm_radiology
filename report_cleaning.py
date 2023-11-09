# import os
# import transformers
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline, set_seed
# from transformers.deepspeed import HfDeepSpeedConfig
# import deepspeed
# from datasets import Dataset
# import argparse
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
# import torch
# import pandas as pd
# import numpy as np

# import sys
# sys.path.append('/data/dangnguyen/report_generation/report-generation')
# from CXRMetric.CheXbert.src.label import label

# CHEXBERT_PATH = '/data/dangnguyen/report_generation/models/chexbert.pth'

# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [label.strip() for label in labels]

#     # rougeLSum expects newline after each sentence
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
#     return preds, labels

# def parse_arge():
#     """Parse the arguments."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_path", type=str, default="/data/dangnguyen/report_generation/mimic_data/report_cleaning/test_cleaning_gt_200.csv", help="Path to the already processed dataset.")
#     parser.add_argument("--output_dir", type=str, default="/data/dangnguyen/report_generation/mimic_data/report_cleaning", help="Path to save the running results.")
#     parser.add_argument("--outfile", type=str, default="output.csv", help="Name of the output file.")
#     parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size to use for testing.")
#     parser.add_argument("--model_id", type=str, default="google/flan-t5-XXL", help="Model id to use for training.")
#     parser.add_argument("--generation_max_length", type=int, default=200, help="Maximum length to use for generation")
#     parser.add_argument("--top_k", type=int, default=1, help="Top k to use for generation.")
#     parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
#     parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
#     parser.add_argument(
#         "--bf16",
#         type=bool,
#         default=True if torch.cuda.get_device_capability()[0] == 8 else False,
#         help="Whether to use bf16.",
#     )
#     # parser.add_argument("--generation_min_length", type=int, default=55, help="Minimum length to use for generation")
#     # parser.add_argument("--generation_no_repeat_ngram_size", type=int, default=3, help="No repeat ngram size to use for generation.")
#     # parser.add_argument("--generation_early_stopping", type=bool, default=True, help="Early stopping to use for generation.")
#     # parser.add_argument("--generation_length_penalty", type=float, default=2.0, help="Length penalty to use for generation.")
#     # parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
#     # parser.add_argument("--top_p", type=float, default=0.95, help="Top p to use for generation.")
#     args = parser.parse_known_args()
#     return args

# # Checks which examples' labels have been changed due to cleaning
# def labels_changed(y_gt, y_gt_neg, output_list):
#     df_gen = pd.DataFrame(output_list, columns=['report'])
#     df_gen = df_gen.replace('REMOVED', '_') # the token REMOVED can stand in for removed parts

#     pre_chexb_path = os.path.join(args.output_dir, 'gen_pre_chexbert.csv')
#     df_gen.to_csv(pre_chexb_path, index=False)
#     y_gen = label(CHEXBERT_PATH, pre_chexb_path, use_gpu=False)
#     y_gen = np.array(y_gen).T

#     y_gen_neg = y_gen.copy()
#     y_gen[(y_gen == 2) | (y_gen == 3)] = 0
#     y_gen_neg[(y_gen_neg == 1) | (y_gen_neg == 3)] = 0
#     y_gen_neg[y_gen_neg == 2] = 1

#     os.system('rm {}'.format(pre_chexb_path))

#     pos_is_diff = np.logical_xor(y_gt, y_gen).any(axis=1)
#     neg_is_diff = np.logical_xor(y_gt_neg, y_gen_neg).any(axis=1)
#     assert len(pos_is_diff) == len(output_list)

#     label_is_diff = np.logical_or(pos_is_diff, neg_is_diff)
#     return label_is_diff

# def label_heuristic(output_list):
#     gt_pos_path = os.path.join(args.output_dir, "gt_pos_labels.pt")
#     gt_neg_path = os.path.join(args.output_dir, "gt_neg_labels.pt")

#     if not (os.path.exists(gt_pos_path) and os.path.exists(gt_neg_path)):
#         data_full = pd.read_csv(args.dataset_path).fillna('_')
#         pre_chexb_path = os.path.join(args.output_dir, "gt_pre_chexbert.csv")

#         data_full[['report']].to_csv(pre_chexb_path, index=False) # temporary file for the purpose of labeling. Will be deleted.
#         y_gt = label(CHEXBERT_PATH, pre_chexb_path, use_gpu=False)
#         y_gt = np.array(y_gt).T

#         # Note on labels:
#         # 0: unmentioned ; 1: positive ; 2: negative ; 3: uncertain
#         y_gt_neg = y_gt.copy()
#         y_gt[(y_gt == 2) | (y_gt == 3)] = 0
#         y_gt_neg[(y_gt_neg == 1) | (y_gt_neg == 3)] = 0
#         y_gt_neg[y_gt_neg == 2] = 1

#         torch.save(y_gt, gt_pos_path)
#         torch.save(y_gt_neg, gt_neg_path)

#     y_gt = torch.load(gt_pos_path)
#     y_gt_neg = torch.load(gt_neg_path)
#     label_is_diff = labels_changed(y_gt, y_gt_neg, output_list)
#     return label_is_diff

# def predict(args, model, tokenizer, data_collator, 
#             instructions, examples, report_list, outfile):
    
#     def preprocess_function(examples):
#         model_inputs = tokenizer(examples["input_text"], return_tensors="pt", padding=True)
#         model_inputs["labels"] = model_inputs.input_ids.detach().clone() # dummy target, copy from inputs
#         return model_inputs
    
#     input_list = [instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent) for input_sent in report_list]
#     dataset = Dataset.from_dict({"input_text": input_list})
#     # this is faster and more reliable -- you should not only include input_ids, but also the masks used for padding for best performance
#     dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False, remove_columns=["input_text"], desc="Running tokenizer on dataset")

#     output_dir = args.output_dir
#     os.makedirs(output_dir, exist_ok=True)

#     training_args = Seq2SeqTrainingArguments(
#         output_dir=output_dir,
#         per_device_eval_batch_size=args.per_device_eval_batch_size,
#         predict_with_generate=True,
#         generation_max_length=args.generation_max_length,
#         fp16=False,  # T5 overflows with fp16
#         bf16=args.bf16,  # Use BF16 if available
#         deepspeed=args.deepspeed,
#         logging_dir=f"{output_dir}/logs",
#         logging_strategy="steps",
#         logging_steps=500,
#     )
#     # Create Trainer instance
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         eval_dataset=dataset,
#         data_collator=data_collator,
#     )

#     predict_results = trainer.predict(dataset, max_length=args.generation_max_length, top_k=args.top_k)

#     if trainer.is_world_process_zero():
#         preds = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)
#         predictions = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

#         label_is_diff = label_heuristic(predictions)
#         for i in range(len(label_is_diff)):
#             if label_is_diff[i]:
#                 predictions[i] = report_list[i]

#         df_res = pd.DataFrame(predictions, columns=['report'])
#         df_res.to_csv(os.path.join(output_dir, outfile), index=False)


# if __name__ == '__main__':
#     args, _ = parse_arge()
#     set_seed(args.seed)

#     model_name = args.model_id
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#     model = model.eval()
#     embedding_size = model.get_input_embeddings().weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         model.resize_token_embeddings(len(tokenizer))

#     data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8)

#     data = pd.read_csv(args.dataset_path)
#     report_list = list(data["report"])

#     RULES = [1, 2, 3, 4, 5, 6, 7]
#     for i in RULES:
#         rule = 'rewrite' + str(i)
#         print(rule)

#         instruct_path = './prompts/report_clean_rules/{}_instructions.txt'.format(rule)
#         examples_path = './prompts/report_clean_rules/{}_sen_fewshot.txt'.format(rule)
#         outfile = '{}_intermediate.csv'.format(rule)
#         instructions = open(instruct_path).read()
#         examples = open(examples_path).read()
        
#         predict(args, model, tokenizer, data_collator, 
#                 instructions, examples, report_list, outfile)

#         torch.distributed.barrier()

#         report_list = pd.read_csv(os.path.join(args.output_dir, outfile))['report'].to_list()


import os
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline, set_seed
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
from datasets import Dataset
import argparse
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch
import pandas as pd
import numpy as np
import nltk

from CXRMetric.CheXbert.src.label import label

MISSING = 0
POSITIVE = 1
NEGATIVE = 2
UNCERTAIN = 3

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexbert_path", type=str, default=None, 
                        help="Path the CheXbert model")
    parser.add_argument("--dataset_path", type=str, default=None, 
                        help="Path to the CSV file of sentences to clean. Ensure there is a `report` column")
    parser.add_argument("--output_dir", type=str, default="./report_cleaning", 
                        help="Path to save the running results.")
    parser.add_argument("--outfile", type=str, default="clean_output.csv", 
                        help="Name of the output file.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, 
                        help="Batch size to use for testing.")
    parser.add_argument("--model_id", type=str, default="google/flan-t5-XXL", 
                        help="Model id to use for training.")
    parser.add_argument("--generation_max_length", type=int, default=200, 
                        help="Maximum length to use for generation")
    parser.add_argument("--top_k", type=int, default=1, 
                        help="Top k to use for generation.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, 
                        help="Path to deepspeed config file.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args

# Checks which examples' labels have been changed due to cleaning
def labels_changed(args, y_gt, y_gt_neg, output_list):
    df_gen = pd.DataFrame(output_list, columns=['report'])
    df_gen = df_gen.replace('REMOVED', '_')

    pre_chexb_path = os.path.join(args.output_dir, 'gen_pre_chexbert.csv')
    df_gen.to_csv(pre_chexb_path, index=False)
    y_gen = label(args.chexbert_path, pre_chexb_path, use_gpu=False)
    y_gen = np.array(y_gen).T

    y_gen_neg = y_gen.copy()
    y_gen[(y_gen == NEGATIVE) | (y_gen == UNCERTAIN)] = 0
    y_gen_neg[(y_gen_neg == POSITIVE) | (y_gen_neg == UNCERTAIN)] = 0
    y_gen_neg[y_gen_neg == NEGATIVE] = 1

    os.system('rm {}'.format(pre_chexb_path))

    pos_is_diff = np.logical_xor(y_gt, y_gen).any(axis=1)
    neg_is_diff = np.logical_xor(y_gt_neg, y_gen_neg).any(axis=1)
    assert len(pos_is_diff) == len(output_list)

    label_is_diff = np.logical_or(pos_is_diff, neg_is_diff)
    return label_is_diff

def label_heuristic(args, output_list):
    gt_pos_path = os.path.join(args.output_dir, "gt_pos_labels.pt")
    gt_neg_path = os.path.join(args.output_dir, "gt_neg_labels.pt")

    if not (os.path.exists(gt_pos_path) and os.path.exists(gt_neg_path)):
        data_full = pd.read_csv(args.dataset_path).fillna('_')
        pre_chexb_path = os.path.join(args.output_dir, "gt_pre_chexbert.csv")

        data_full[['report']].to_csv(pre_chexb_path, index=False)
        y_gt = label(args.chexbert_path, pre_chexb_path, use_gpu=False)
        y_gt = np.array(y_gt).T

        y_gt_neg = y_gt.copy()
        y_gt[(y_gt == NEGATIVE) | (y_gt == UNCERTAIN)] = 0
        y_gt_neg[(y_gt_neg == POSITIVE) | (y_gt_neg == UNCERTAIN)] = 0
        y_gt_neg[y_gt_neg == NEGATIVE] = 1

        torch.save(y_gt, gt_pos_path)
        torch.save(y_gt_neg, gt_neg_path)

    y_gt = torch.load(gt_pos_path)
    y_gt_neg = torch.load(gt_neg_path)
    label_is_diff = labels_changed(args, y_gt, y_gt_neg, output_list)
    return label_is_diff

def predict(args, model, tokenizer, data_collator, 
            instructions, examples, report_list, outfile):
    
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], 
                                 return_tensors="pt", padding=True)
        model_inputs["labels"] = model_inputs.input_ids.detach().clone() # dummy target, copy from inputs
        return model_inputs
    
    input_list = [instructions.format(EXAMPLES=examples, INPUT_QUERY=input_sent) 
                  for input_sent in report_list]
    dataset = Dataset.from_dict({"input_text": input_list})
    dataset = dataset.map(preprocess_function, batched=True, 
                          load_from_cache_file=False, 
                          remove_columns=["input_text"], 
                          desc="Running tokenizer on dataset")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        deepspeed=args.deepspeed,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
    )
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
    )

    predict_results = trainer.predict(dataset, 
                                      max_length=args.generation_max_length, 
                                      top_k=args.top_k)

    if trainer.is_world_process_zero():
        preds = np.where(predict_results.predictions != -100, 
                         predict_results.predictions, 
                         tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(preds, 
                                             skip_special_tokens=True, 
                                             clean_up_tokenization_spaces=True)

        label_is_diff = label_heuristic(args, predictions)
        for i in range(len(label_is_diff)):
            if label_is_diff[i]:
                predictions[i] = report_list[i]

        df_res = pd.DataFrame(predictions, columns=['report'])
        df_res.to_csv(os.path.join(output_dir, outfile), index=False)

if __name__ == '__main__':
    args, _ = parse_args()
    set_seed(args.seed)

    model_name = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                           model=model, 
                                           label_pad_token_id=tokenizer.pad_token_id, 
                                           pad_to_multiple_of=8)

    data = pd.read_csv(args.dataset_path)
    report_list = list(data["report"])

    RULES = [1, 2, 3, 4, 5, 6, 7]
    for i in RULES:
        rule = 'rewrite' + str(i)
        print(rule)

        instruct_path = './prompts/report_clean_rules/{}_instructions.txt'.format(rule)
        examples_path = './prompts/report_clean_rules/{}_sen_fewshot.txt'.format(rule)
        outfile = '{}_intermediate.csv'.format(rule)
        instructions = open(instruct_path).read()
        examples = open(examples_path).read()
        
        predict(args, model, tokenizer, data_collator, 
                instructions, examples, report_list, outfile)

        torch.distributed.barrier()

        report_list = pd.read_csv(os.path.join(args.output_dir, outfile))['report'].to_list()

# deepspeed --num_gpus=4 report_cleaning.py --chexbert_path /data/dangnguyen/report_generation/models/chexbert.pth --dataset_path /data/dangnguyen/report_generation/mimic_data/report_cleaning/test_cleaning_gt_200.csv --output_dir /data/dangnguyen/report_generation/mimic_data/report_cleaning/github_ready/