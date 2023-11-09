# import sys
# import transformers
# import torch
# from torch.utils.data import DataLoader
# import pandas as pd
# import numpy as np
# from collections import OrderedDict

# sys.path.append('/data/dangnguyen/report_generation/report-generation/')
# # sys.path.remove('/data/chacha/CXR-Report-Metric')
# from CXRMetric.run_eval import calc_metric
# from CXRMetric.CheXbert.src.label import label

# from model.my_models import DenseChexpertModel

# LLAMA_PATH = './radiology_models/xray_llama_7b/'
# INSTRUCTIONS_PATH = './prompts/report_writing/instructions.txt'

# GT_PATH = '/data/dangnguyen/report_generation/mimic_data/test3_ind_imp_chexbert.csv'
# GEN_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/test3_gen_imp.csv'

# VISION_OUT_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/test3_pred_viz.csv'
# IMG_PATH = '/data/dangnguyen/report_generation/mimic_data/test3_images.pt'

# cxr_labels = [
#         'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#         'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
#         'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
#         'Pneumothorax', 'Support Devices']

# # Takes in a list of images and returns the predicted labels for the
# # 14 MIMIC-CXR conditions. Negative, uncertain, and missing labels are
# # mapped to 0
# def predict_img_labels(images):
#     VISION_MODEL_PATH = './radiology_models/vision/model_best.pth'
#     THRESHOLDS_PATH = './radiology_models/vision/tuned_thresholds.pt'

#     device = 'cuda:1'
#     checkpoint = torch.load(VISION_MODEL_PATH)
#     state_dict = checkpoint['state_dict']
#     model = DenseChexpertModel()
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)
#     model = model.to(device)
#     model.eval()

#     BATCH_SIZE = 128
#     num_batches = len(images) // BATCH_SIZE + 1
#     outputs = []
#     with torch.no_grad():
#         for i in range(num_batches):
#             start = i * BATCH_SIZE
#             end = (i + 1) * BATCH_SIZE
#             img_batch = images[start:end].to(device)
#             output, _ = model(img_batch)
#             outputs.append(output)
#     outputs = torch.cat(outputs, dim=0) 

#     # rounding based on tuned thresholds
#     thresholds = torch.load(THRESHOLDS_PATH).tolist()
#     outputs_pred = torch.sigmoid(outputs).detach().cpu().numpy() 
#     for i in range(len(thresholds)):
#         outputs_pred[:, i] = (outputs_pred[:, i] >= thresholds[i]).astype(int)

#     return outputs_pred

# # Converts a label vector to English
# def labels_to_eng(labels):
#     diag = ''
#     for i in range(len(labels)):
#         label = labels[i]
#         cond = cxr_labels[i]
#         if label == 1:
#             diag += cond
#             diag += ', '
#     return diag

# # Takes in a DataFrame containing the indication and labels (can be predicted or GT)
# # and returns a list of formatted inputs for the model
# def format_input(df):
#     inputs = []
#     for _, row in df.iterrows():
#         ind = row['indication']
#         labels = labels_to_eng(row[cxr_labels])[:-2]
#         inp = 'Indication: {}\nPositive labels: {}'.format(ind, labels)
#         inputs.append(inp)
#     return inputs

# # Takes in the list of formatted input (indication + labels) and writes the report
# def write_report(input_list):
#     # BATCH_SIZE = 32
#     # num_batches = len(input_list) // BATCH_SIZE + 1
#     # prompts = [instructions.format(input=input_example) for input_example in input_list]
#     # output = []
#     # for i in range(num_batches):
#     #     start = i * BATCH_SIZE
#     #     end = (i + 1) * BATCH_SIZE
#     #     batch = prompts[start:end]
#     #     if len(batch) == 0:
#     #         continue
#     #     with torch.no_grad():
#     #         input_tokens = tokenizer(batch, padding=True, return_tensors="pt")
#     #         generate_ids = model.generate(input_tokens['input_ids'].to('cuda:0'), max_length=200)
#     #         output_batch = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
#     #         output_cleaned = [example.split('Response:')[1] for example in output_batch]
#     #         output += output_cleaned
#     #     print('Completed batch: {}/{}'.format(i+1, num_batches))
#     # return output

#     device = 'cuda:0'
#     model = transformers.LlamaForCausalLM.from_pretrained(LLAMA_PATH).to(device)
#     tokenizer = transformers.LlamaTokenizer.from_pretrained(LLAMA_PATH)

#     instructions = open(INSTRUCTIONS_PATH).read()
#     prompts = [instructions.format(input=input_example) for input_example in input_list]

#     output = []
#     with torch.no_grad():
#         for i in range(len(prompts)):
#             prompt = prompts[i]
#             input_tokens = tokenizer(prompt, return_tensors="pt")
#             generate_ids = model.generate(input_tokens['input_ids'].to(device), max_length=200)
#             output_batch = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
#             output_cleaned = [example.split('Response:')[1] for example in output_batch]
#             output += output_cleaned
#             print('Completed example: {}/{}'.format(i+1, len(prompts)))

#     return output

# if __name__ == "__main__":
#     data = pd.read_csv(GT_PATH).fillna('') # some indications will be empty
#     # model_input = data[['indication']]

#     # images = torch.load(IMG_PATH)
#     # predicted_labels = predict_img_labels(images)
#     # df_pred_labels = pd.DataFrame(predicted_labels, columns=cxr_labels)
    
#     # model_input = pd.concat([model_input, df_pred_labels], axis=1) # columns: indication, [cxr_labels]
#     # model_input.to_csv(VISION_OUT_PATH, index=False)

#     model_input = pd.read_csv(VISION_OUT_PATH)

#     input_list = format_input(model_input)
#     output_list = write_report(input_list)

#     data['generated'] = output_list
#     data[['study_id','report','generated']].rename(columns={'report':'original', 'generated':'report'}).to_csv(GEN_PATH, index=False)



import transformers
import torch
import pandas as pd
from collections import OrderedDict
import argparse
import os

from model.my_models import DenseChexpertModel
from format_llama_input import labels_to_eng

# LLAMA_PATH = './radiology_models/xray_llama_7b/'
# INSTRUCTIONS_PATH = './prompts/report_writing/instructions.txt'

# INDICATION_PATH = '/data/dangnguyen/report_generation/mimic_data/test3_ind_imp_chexbert.csv'
# GEN_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/test3_gen_imp.csv'

# VISION_OUT_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/test3_pred_viz.csv'
# IMG_PATH = '/data/dangnguyen/report_generation/mimic_data/test3_images.pt'

# VISION_MODEL_PATH = './radiology_models/vision/model_best.pth'
# THRESHOLDS_PATH = './radiology_models/vision/tuned_thresholds.pt'

CXR_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
              'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
              'Pneumothorax', 'Support Devices']

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--llama_path", type=str, help="Path to the finetuned LLaMA model.")
    parser.add_argument("--instruct_path", type=str, 
                        default="./prompts/report_writing/instructions.txt",
                        help="Path to the prompt .txt file.")
    parser.add_argument("--vision_path", type=str,
                        help="""Path to the directory containing the vision classifier checkpoint
                        and tuned thresholds for each label.""")
    parser.add_argument("--image_batch_size", type=int, default=128,
                        help="Batch size for image classifier inference.")

    # Data
    parser.add_argument("--indication_path", type=str, help="""Path to indications CSV file.
                        Should contain `study_id` and `report` columns.""")
    parser.add_argument("--vision_out_path", type=str,
                        help="""Path to the CSV file containing predicted labels.
                        Specify this to store the predicted labels to avoid
                        re-running the classifier on the same data afterwards.""")
    parser.add_argument("--image_path", type=str, help="Path to .pt images file.")
    parser.add_argument("--outpath", type=str, default="generated_reports.csv", 
                        help="Path to file storing generated reports.")

    args = parser.parse_known_args()
    return args


# Takes in a list of images and returns the predicted labels for the
# 14 MIMIC-CXR conditions. Negative, uncertain, and missing labels are
# mapped to 0
def predict_img_labels(images, args):
    vision_model_path = os.path.join(args.vision_path, "model_best.pth")
    thresholds_path = os.path.join(args.vision_path, "tuned_thresholds.pt")

    device = 'cuda:0'
    checkpoint = torch.load(vision_model_path)
    state_dict = checkpoint['state_dict']
    model = DenseChexpertModel()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    num_batches = len(images) // args.image_batch_size + 1
    outputs = []
    with torch.no_grad():
        for i in range(num_batches):
            start = i * args.image_batch_size
            end = (i + 1) * args.image_batch_size
            img_batch = images[start:end].to(device)
            output, _ = model(img_batch)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)

    # rounding based on tuned thresholds
    thresholds = torch.load(thresholds_path).tolist()
    outputs_pred = torch.sigmoid(outputs).detach().cpu().numpy() 
    for i in range(len(thresholds)):
        outputs_pred[:, i] = (outputs_pred[:, i] >= thresholds[i]).astype(int)

    return outputs_pred

# # Converts a label vector to English
# def labels_to_eng(labels):
#     diag = ''
#     for i in range(len(labels)):
#         label = labels[i]
#         cond = cxr_labels[i]
#         if label == 1:
#             diag += cond
#             diag += ', '
#     return diag

# Takes in a DataFrame containing the indication and labels (can be predicted or GT)
# and returns a list of formatted inputs for the model
def format_input(df):
    inputs = []
    for _, row in df.iterrows():
        ind = row['indication']
        labels = labels_to_eng(row[CXR_LABELS])[:-2]
        inp = 'Indication: {}\nPositive labels: {}'.format(ind, labels)
        inputs.append(inp)
    return inputs

# Takes in the list of formatted input (indication + labels) and writes the report
def write_report(input_list, args):
    device = 'cuda:0'
    model = transformers.LlamaForCausalLM.from_pretrained(args.llama_path).to(device)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.llama_path)

    instructions = open(args.instruct_path).read()
    prompts = [instructions.format(input=input_example) for input_example in input_list]

    output = []
    with torch.no_grad():
        for i in range(len(prompts)):
            prompt = prompts[i]
            input_tokens = tokenizer(prompt, return_tensors="pt")
            generate_ids = model.generate(input_tokens['input_ids'].to(device), max_length=200)
            output_batch = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            output_cleaned = [example.split('Response:')[1] for example in output_batch]
            output += output_cleaned
            print('Completed example: {}/{}'.format(i+1, len(prompts)))

    return output

if __name__ == "__main__":
    args, _ = parse_args()

    data = pd.read_csv(args.indication_path).fillna('')[:100]
    
    if not os.path.exists(args.vision_out_path):
        model_input = data[['study_id','indication']]

        images = torch.load(args.image_path)
        predicted_labels = predict_img_labels(images, args)
        df_pred_labels = pd.DataFrame(predicted_labels, columns=CXR_LABELS)
        
        model_input = pd.concat([model_input, df_pred_labels], axis=1)
        model_input.to_csv(args.vision_out_path, index=False)

    model_input = pd.read_csv(args.vision_out_path)[:100]
    input_list = format_input(model_input)
    output_list = write_report(input_list, args)

    data['generated'] = output_list
    data[['study_id','generated']].rename(columns={'generated':'report'}) \
                                  .to_csv(args.outpath, index=False)


