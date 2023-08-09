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

GT_PATH = '/data/dangnguyen/report_generation/mimic_data/test_ind_imp.csv'
GEN_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/baselines/test_xraygpt-fewshot-ind_imp.csv'
OUT_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/baselines/test_xraygpt-fewshot-ind_imp_metrics.csv'

# # This is for validation
# GT_PATH = '/data/dangnguyen/report_generation/mimic_data/val_ind_imp.csv'
# GEN_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/val_gen_imp.csv'
# OUT_PATH = '/data/dangnguyen/report_generation/mimic_data/finetune_llm/val_gen_imp_metrics.csv'

calc_metric(GT_PATH, GEN_PATH, OUT_PATH, use_idf=False)