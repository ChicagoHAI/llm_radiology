import json
import numpy as np
import os
import re
import pandas as pd
import pickle
import torch

from bert_score import BERTScorer
from fast_bleu import BLEU
from nltk.translate.bleu_score import sentence_bleu
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import config
from CXRMetric.radgraph_evaluate_model import run_radgraph

import sys
sys.path.append('/data/dangnguyen/report_generation/report-generation/CXRMetric/')
sys.path.append('/data/dangnguyen/report_generation/report-generation/CXRMetric/dygiepp/')
from CheXbert.src.label import label # a function to label reports using CheXbert

"""Computes 4 individual metrics and a composite metric on radiology reports."""

# Paths to chexbert and radgraph models
CHEXBERT_PATH = '/data/dangnguyen/report_generation/models/chexbert.pth'
RADGRAPH_PATH ='/data/dangnguyen/report_generation/models/radgraph.tar.gz'

CHEXPERT_LABELS_PATH = '/data/mimic_data/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'

NORMALIZER_PATH = "CXRMetric/normalizer.pkl"
COMPOSITE_METRIC_V0_PATH = "CXRMetric/composite_metric_model.pkl"
COMPOSITE_METRIC_V1_PATH = "CXRMetric/radcliq-v1.pkl"

REPORT_COL_NAME = "report"
STUDY_ID_COL_NAME = "study_id"
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]

cache_path = "./cache/"
pred_embed_path = os.path.join(cache_path, "pred_embeddings.pt")
gt_embed_path = os.path.join(cache_path, "gt_embeddings.pt")

weights = {"bigram": (1/2., 1/2.)}
composite_metric_col_v0 = "RadCliQ-v0"
composite_metric_col_v1 = "RadCliQ-v1"

cxr_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', \
'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion','Lung Opacity', \
'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

# for some reason, there is another ordering (unsorted) of the labels
cxr_labels_2 = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',\
'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',\
'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']

# cxr_labels_2 but without No Finding
cxr_labels_3 = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',\
'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',\
'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

# # labels that have a non-trivial number of negative mentions
# cxr_negative = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Pneumothorax', 'Pleural Effusion']


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """
    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

def add_bleu_col(gt_df, pred_df):
    """Computes BLEU-2 and adds scores as a column to prediction df."""
    # pred_df["bleu_score"] = [0.0] * len(pred_df)
    # for i, row in gt_df.iterrows():
    #     gt_report = prep_reports([row[REPORT_COL_NAME]])[0]
    #     pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
    #     predicted_report = \
    #         prep_reports([pred_row[REPORT_COL_NAME].values[0]])[0]
        
    #     if len(pred_row) == 1:
    #         # bleu = BLEU([gt_report], weights) # for some reason I cannot use this package due to a bug
    #         # score = bleu.get_score([predicted_report])["bigram"]
    #         score = [sentence_bleu([gt_report], predicted_report, weights=(1/2, 1/2))] # to use BLEU-2
    #         assert len(score) == 1
    #         _index = pred_df.index[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]].tolist()[0]
    #         pred_df.at[_index, "bleu_score"] = score[0]

    pred_df["bleu_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row[REPORT_COL_NAME]])[0]
        predicted_report = prep_reports([pred_df.loc[i][REPORT_COL_NAME]])[0]
        
        score = [sentence_bleu([gt_report], predicted_report, weights=(1/2, 1/2))] # to use BLEU-2
        assert len(score) == 1
        # _index = pred_df.index[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]].tolist()[0]
        pred_df.at[i, "bleu_score"] = score[0]

    return pred_df

def add_bertscore_col(gt_df, pred_df, use_idf):
    """Computes BERTScore and adds scores as a column to prediction df."""
    test_reports = gt_df[REPORT_COL_NAME].tolist()
    test_reports = [re.sub(r' +', ' ', test) for test in test_reports]
    method_reports = pred_df[REPORT_COL_NAME].tolist()
    method_reports = [re.sub(r' +', ' ', report) for report in method_reports]

    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=True,
        idf=use_idf,
        idf_sents=test_reports)
    _, _, f1 = scorer.score(method_reports, test_reports)
    pred_df["bertscore"] = f1
    return pred_df

def add_semb_col(pred_df, semb_path, gt_path):
    """Computes s_emb and adds scores as a column to prediction df."""
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    list_label_embeds = []
    list_pred_embeds = []
    for data_idx in sorted(label_embeds.keys()):
        list_label_embeds.append(label_embeds[data_idx])
        list_pred_embeds.append(pred_embeds[data_idx])
    np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
    np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum() / (
            np.linalg.norm(label) * np.linalg.norm(pred))
        scores.append(sim_scores)
    pred_df["semb_score"] = scores
    return pred_df

def add_radgraph_col(pred_df, entities_path, relations_path):
    """Computes RadGraph F1 and adds scores as a column to prediction df."""
    study_id_to_radgraph = {}
    with open(entities_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            try:
                study_id_to_radgraph[int(study_id)] = float(f1)
            except:
                continue
    with open(relations_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            try:
                study_id_to_radgraph[int(study_id)] += float(f1)
                study_id_to_radgraph[int(study_id)] /= float(2)
            except:
                continue
    radgraph_scores = []
    count = 0
    for i, row in pred_df.iterrows():
        try:
            radgraph_scores.append(study_id_to_radgraph[int(row[STUDY_ID_COL_NAME])])
        except KeyError:
            radgraph_scores.append(0)
    pred_df["radgraph_combined"] = radgraph_scores
    return pred_df

# Computes negative F1 and negative F1-5 for the labels:
# Edema, Consolidation, Pneumonia, Pneumothorax, Pleural Effusion.
# Also returns a list of Negative F1's for each label
def negative_f1(gt, pred):
    labels = range(13)
    labels_five = [4, 5, 6, 8, 9]
    f1_scores = []

    for i in labels:
        score = f1_score(gt[:, i], pred[:, i], zero_division=0)
        f1_scores.append(score)
    f1_scores = np.array(f1_scores)

    neg_f1 = f1_scores.mean()
    neg_f1_five = f1_scores[labels_five].mean()
    return neg_f1, neg_f1_five, f1_scores

# Computes positive F1 and positive F1-5 for all labels except No Finding
# When `use_five` is True, we only calculate F1 with the labels:
# Atelectasis, Consolidation, Edema, Pleural Effusion, Cardiomegaly
def positive_f1(gt, pred):
    labels_five = [1, 4, 5, 7, 9] # the indices of the five labels according to the cxr_labels_2 ordering
    gt_five = gt[:, labels_five]
    pred_five = pred[:, labels_five]

    pos_f1 = f1_score(gt, pred, average='macro', zero_division=0)
    pos_f1_five = f1_score(gt_five, pred_five, average='macro', zero_division=0)
    return pos_f1, pos_f1_five

# Computes the positive and negative F1
def compute_f1(df_gt, df_pred):
    # need to make sure df_gt and df_pred has a column called "report"
    gt_pre_chexb = './gt_pre-chexbert.csv'
    df_gt.to_csv(gt_pre_chexb, index=False)

    y_gt = label(CHEXBERT_PATH, gt_pre_chexb, use_gpu=True)
    y_gt = np.array(y_gt).T
    y_gt = y_gt[:, :] # excluding No Finding

    # Note on labels:
    # 0: unmentioned ; 1: positive ; 2: negative ; 3: uncertain
    y_gt_neg = y_gt.copy()
    y_gt_neg[(y_gt_neg == 1) | (y_gt_neg == 3)] = 0
    y_gt_neg[y_gt_neg == 2] = 1
    
    y_gt[(y_gt == 2) | (y_gt == 3)] = 0

    torch.save(y_gt, './gt_chexb.pt')

    pred_pre_chexb = './pred_pre-chexbert.csv'
    df_pred.to_csv(pred_pre_chexb, index=False)

    # the labels are according to the 2nd ordering (see run_eval.py)
    y_pred = label(CHEXBERT_PATH, pred_pre_chexb, use_gpu=True)
    y_pred = np.array(y_pred).T
    y_pred = y_pred[:, :]

    y_pred_neg = y_pred.copy()
    y_pred_neg[(y_pred_neg == 1) | (y_pred_neg == 3)] = 0
    y_pred_neg[y_pred_neg == 2] = 1
    
    y_pred[(y_pred == 2) | (y_pred == 3)] = 0

    torch.save(y_pred, './pred_chexb.pt')
    
    assert y_gt.shape == y_pred.shape

    os.system('rm {}'.format(gt_pre_chexb))
    os.system('rm {}'.format(pred_pre_chexb))

    pos_f1, pos_f1_five = positive_f1(y_gt, y_pred)
    neg_f1, neg_f1_five, label_neg_f1 = negative_f1(y_gt_neg, y_pred_neg)
    return pos_f1, pos_f1_five, neg_f1, neg_f1_five, label_neg_f1

def hallucination_prop(df_pred):
    type_keywords = ["compar","interval","new","increas","worse","chang",
                      "persist","improv","resol","disappear",
                      "prior","stable","previous","again","remain","remov",
                      "similar","earlier","decreas","recurr","redemonstrate",
                      "status",
                      "findings","commun","report","convey","relay","enter","submit",
                      "recommend","suggest","should",
                      " ap "," pa "," lateral ","view"]
    reports = df_pred['report'].to_list()
    has_hallucination = []

    for report in reports:
        report = report.lower()
        has_hallu = 0
        for keyword in type_keywords:
            if keyword in report:
                has_hallu = 1
                break
        has_hallucination.append(has_hallu)
    has_hallu_np = np.array(has_hallucination)
    
    hallu_freqs = has_hallu_np.sum()
    hallu_props = hallu_freqs / len(has_hallu_np)
    return hallu_props

def calc_metric(gt_csv, pred_csv, out_csv, use_idf=False): # TODO: support single metrics at a time
    """Computes four metrics and composite metric scores."""
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    cache_gt_csv = os.path.join(
        os.path.dirname(gt_csv), f"cache_{os.path.basename(gt_csv)}")
    cache_pred_csv = os.path.join(
        os.path.dirname(pred_csv), f"cache_{os.path.basename(pred_csv)}")
    
    # Must ensure that there is no empty impression in the input
    gt = pd.read_csv(gt_csv).sort_values(by=[STUDY_ID_COL_NAME])
    pred = pd.read_csv(pred_csv).sort_values(by=[STUDY_ID_COL_NAME]).fillna('_')

    # Keep intersection of shared indices
    gt_study_ids = set(gt['study_id'])
    pred_study_ids = set(pred['study_id'])

    shared_study_ids = gt_study_ids.intersection(pred_study_ids)
    print(f"Number of shared indices: {len(shared_study_ids)}")
    gt = gt.loc[gt['study_id'].isin(shared_study_ids)].reset_index()
    pred = pred.loc[pred['study_id'].isin(shared_study_ids)].reset_index()

    print('GT: {} Gen: {}'.format(len(gt), len(pred)))

    gt.to_csv(cache_gt_csv)
    pred.to_csv(cache_pred_csv)

    # check that length and study IDs are the same
    assert len(gt) == len(pred)
    assert (REPORT_COL_NAME in gt.columns) and (REPORT_COL_NAME in pred.columns)
    assert (gt[STUDY_ID_COL_NAME].equals(pred[STUDY_ID_COL_NAME]))

    # add bleu column to the eval df
    pred = add_bleu_col(gt, pred)

    # add bertscore column to the eval df
    pred = add_bertscore_col(gt, pred, use_idf)

    # run encode.py to make the semb column
    os.system(f"mkdir -p {cache_path}")
    os.system(f"python /data/dangnguyen/report_generation/report-generation/CXRMetric/CheXbert/src/encode.py \
              -c {CHEXBERT_PATH} -d {cache_pred_csv} -o {pred_embed_path}") # can change this path to wherever you store CheXbert
    os.system(f"python /data/dangnguyen/report_generation/report-generation/CXRMetric/CheXbert/src/encode.py \
              -c {CHEXBERT_PATH} -d {cache_gt_csv} -o {gt_embed_path}")
    pred = add_semb_col(pred, pred_embed_path, gt_embed_path)

    # run radgraph to create that column
    entities_path = os.path.join(cache_path, "entities_cache.json")
    relations_path = os.path.join(cache_path, "relations_cache.json")
    run_radgraph(cache_gt_csv, cache_pred_csv, cache_path, RADGRAPH_PATH,
                 entities_path, relations_path)
    pred = add_radgraph_col(pred, entities_path, relations_path)

    # computing macro F1
    pos_f1, pos_f1_five, neg_f1, neg_f1_five, label_neg_f1 = compute_f1(gt, pred)

    # computing mean hallucination proportion
    hallu_prop = hallucination_prop(pred)

    bleuscore = pred['bleu_score'].mean()
    bertscore = pred['bertscore'].mean()

    # print('{} {}'.format(bleuscore, bertscore))
    # print('{} {} {}'.format(pos_f1, pos_f1_five, hallu_prop))

    # save results in the out folder
    pred.to_csv(out_csv, index=False)
    
    out_csv_avg = out_csv[:-4] + '_avg.csv'
    metrics_avg = pred[COLS].mean().to_list()
    metrics_avg += [pos_f1, pos_f1_five, neg_f1, neg_f1_five, hallu_prop]
    metrics_avg = np.array(metrics_avg)
    metrics_avg = np.concatenate([metrics_avg, label_neg_f1])

    COLS_2 = COLS + ['positive_f1','positive_f1_5','negative_f1','negative_f1_5','hall_prop']
    COLS_2 += cxr_labels_2[:-1]
    df_metrics_avg = pd.DataFrame(COLS_2, columns=['metrics'])
    df_metrics_avg['score'] = metrics_avg
    df_metrics_avg.round(3).to_csv(out_csv_avg, index=False)

