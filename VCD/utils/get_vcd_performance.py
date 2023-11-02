import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', default='data/Clothfold_GNS_12.07.10.59_11_planning/progress.csv', type=str, help="VCD planning progress.csv file")
args = parser.parse_args()
data = pd.read_csv(args.csv_file)

print(args.csv_file)
print(f'info_final_normalized_performance_mean: ', data['info_final_normalized_performance'].mean())
print(f'info_final_normalized_performance_std: ', data['info_final_normalized_performance'].std())
