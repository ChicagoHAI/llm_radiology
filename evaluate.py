import sys
import transformers
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from collections import OrderedDict

sys.path.append('/data/dangnguyen/report_generation/report-generation/')
sys.path.remove('/data/chacha/CXR-Report-Metric')
from CXRMetric.run_eval import calc_metric
from CXRMetric.CheXbert.src.label import label

GT_PATH = '/data/dangnguyen/report_generation/mimic_data/val_ind_imp.csv'
GEN_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/val_gen_imp_clean_ind.csv'
OUT_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/val_gen_imp_clean_ind_metrics.csv'

calc_metric(GT_PATH, GEN_PATH, OUT_PATH, use_idf=False)