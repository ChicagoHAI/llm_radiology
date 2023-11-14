import transformers
import torch
import pandas as pd
from collections import OrderedDict
import argparse
import os

from image_model.model.my_models import DenseChexpertModel
from format_llama_input import labels_to_eng

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


