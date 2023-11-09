import nltk.data
import pandas as pd
import argparse
import os

def section_start(lines, section='IMPRESSION:'):
    """Finds line index that is the start of the section."""
    for idx, line in enumerate(lines):
        if section in line:
            return idx
    return -1

def generate_section_csv(df, section='IMPRESSION:'):
    """Generates a csv containing report sections."""
    df_report = df.copy()
    for index, row in df_report.iterrows():
        report = row['report'].splitlines()
        section_idx = section_start(report, section)
        seperator = ''
        if section_idx != -1:
            section = seperator.join(report[section_idx:]).replace(section, '').replace('\n', '').strip()
        else:
            section = ''
        df_report.at[index,'report'] = section
    return df_report

def generate_sentence_level_section_csv(df, split, dir, tokenizer):
    """Generates a csv containing all impression sentences."""
    df_imp = []
    for index, row in df.iterrows():
        report = row['report'].splitlines()

        impression_idx = section_start(report)
        seperator = ''
        if impression_idx != -1:
            impression = seperator.join(report[impression_idx:]).replace('IMPRESSION:', '').replace('\n', '').strip()
        else:
            impression = ''
        
        for sent_index, sent in enumerate(split_sentences(impression, tokenizer)):
            df_imp.append([row['dicom_id'], row['study_id'], row['subject_id'], sent_index, sent])
    
    df_imp = pd.DataFrame(df_imp, columns=['dicom_id', 'study_id', 'subject_id', 'sentence_id', 'report'])

    out_name = f'mimic_{split}_sentence_impressions.csv'
    out_path = os.path.join(dir, out_name)
    df_imp.to_csv(out_path, index=False)

def split_sentences(report, tokenizer):
    """Splits sentences by periods and removes numbering and nans."""
    sentences = []
    if not (isinstance(report, float) and math.isnan(report)):
        for sentence in tokenizer.tokenize(report):
            try:
                float(sentence)  # Remove numbering
            except ValueError:
                sentences.append(sentence)
    return sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a report section and generate csvs for report level and sentence level.')
    parser.add_argument('--dir', type=str, required=True, help='directory where train and test report reports are stored and where impression sections will be stored')
    parser.add_argument('--section', type=str, help='the section of the report to be extracted') # TO-DO
    args = parser.parse_args()

    train_path = os.path.join(args.dir, 'mimic_train_full.csv')
    test_path = os.path.join(args.dir, 'mimic_test_full.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if args.section == 'impression':
        keywords = ['IMPRESSION:', 'FINDINGS AND IMPRESSION:'] 
    elif args.section == 'findings':
        keywords = ['FINDINGS:', 'FINDINGS AND IMPRESSION:']
    elif args.section == 'indication':
        keywords = ["INDICATION:"]
    else:
        raise ValueError("section not supported")

    df_section_trains = [generate_section_csv(train_df, kw) for kw in keywords]
    df_section_tests = [generate_section_csv(test_df, kw) for kw in keywords]
    df_section_train = pd.concat(df_section_trains)
    df_section_test = pd.concat(df_section_tests)

    train_out_name = f'mimic_train_{args.section}.csv'
    train_out_path = os.path.join(args.dir, train_out_name)

    test_out_name = f'mimic_test_{args.section}.csv'
    test_out_path = os.path.join(args.dir, test_out_name)

    df_section_train.to_csv(train_out_path)
    df_section_test.to_csv(test_out_path)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    generate_sentence_level_impression_csv(train_df, 'train', args.dir, tokenizer)



