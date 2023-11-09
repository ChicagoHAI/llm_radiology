import sys
sys.path.append('/data/dangnguyen/report_generation/report-generation/')
from CXRMetric.run_eval import calc_metric

# We need 3 files: GT, prediction, and output, to input into calc_metric()
# calc_metric() will save the evaluations in the output CSV
save_gt_path = '/net/scratch/chacha/future_of_work/report-generation/xraygpt/xraygpt_beam_0_temperature_1_gt_test_cleaned.csv'
save_pred_path = '/net/scratch/chacha/future_of_work/report-generation/xraygpt/xraygpt_beam_0_temperature_1_preds_test_cleaned.csv'
out_file = '/net/scratch/chacha/future_of_work/report-generation/xraygpt/xraygpt_beam_0_temperature_1_test_matching_metrics.csv'

calc_metric(save_gt_path, save_pred_path, out_file, use_idf=False)