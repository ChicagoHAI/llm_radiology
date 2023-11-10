import pandas as pd
import json
import argparse

CXR_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
              'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
              'Pneumothorax', 'Support Devices']

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--indication_path", type=str, default=None, 
                        help="""Path to the indications. Should contain `study_id`
                                and `report` (storing indications) columns.""")
    parser.add_argument("--impression_path", type=str, default=None, 
                        help="""Path to the impressions. Should contain `study_id`
                                and `report`(storing impressions) columns,
                                and columns of CheXbert report labels.""")
    parser.add_argument("--outpath", type=str, default="llama_input.json",
                        help="JSON file to save formatted input.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed to use for training.")
    parser.add_argument("--sample_percent", type=float, default=1.0,
                        help="The percentage of reports to sample.")
    args = parser.parse_known_args()
    return args

# Converts a label vector to English
def labels_to_eng(labels):
    diag = ''
    for i in range(len(labels)):
        label = labels[i]
        cond = CXR_LABELS[i]
        if label == 1:
            diag += cond
            diag += ', '
    return diag

def format_input(df):
    instruction = "Write a radiology report responding to the indication. Include all given positive labels."
    finetune_data = []

    for _, row in df.iterrows():
        ind = row['indication']
        imp = row['impression']
        labels = labels_to_eng(row[CXR_LABELS])[:-2]
        
        inp = 'Indication: {}.\nPositive labels: {}'.format(ind, labels)
        sample = {
            'instruction': instruction,
            'input': inp,
            'output': imp
        }
        finetune_data.append(sample)
    return finetune_data


if __name__ == '__main__':
    args, _ = parse_args()

    df_ind = pd.read_csv(args.indication_path)[['study_id','report']].drop_duplicates()
    df_ind = df_ind.rename(columns={'report': 'indication'}).fillna('')

    df_imp_chexb = pd.read_csv(args.impression_path)[['study_id','report'] + CXR_LABELS]
    df_imp_chexb = df_imp_chexb.rename(columns={'report': 'impression'})

    df_ind_imp = df_imp_chexb.merge(df_ind, on='study_id')
    df_ind_imp_sample = df_ind_imp.sample(frac=args.sample_percent, 
                                          random_state=args.seed)
    finetune_data = format_input(df_ind_imp_sample)

    with open(args.outpath, 'w') as json_file:
        json.dump(finetune_data, json_file)