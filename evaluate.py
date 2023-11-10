import argparse
from CXRMetric.run_eval import calc_metric

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, 
                        help="Path to CSV file of groundtruth reports.")
    parser.add_argument("--gen_path", type=str, 
                        help="Path to CSV file of generated reports.")
    parser.add_argument("--out_path", type=str, default="gen_metrics.csv",
                        help="Path to CSV file containing performance scores.")

    args = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args, _ = parse_args()
    calc_metric(args.gt_path, args.gen_path, args.out_path, use_idf=False)