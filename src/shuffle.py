
import sys
sys.path.append("/home/chri6578/Documents/TLPCF/") 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()
from tqdm import tqdm
import pandas as pd
import copy
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description="Shuffle.")
parser.add_argument('-d', '--dataset', type=str, help='The directory path')
parser.add_argument('-s', '--sample', type=int, help='The sample id')

parser.add_argument("args", nargs='*', help="List of arguments")
args = parser.parse_args()

dataset = args.dataset
sample = args.sample

DATA_DIR = "/home/chri6578/Documents/gttp/data"
DF = pd.read_csv(f"{DATA_DIR}/{dataset}/ml_{dataset}.csv")   

test_time_start = {
    "reddit": 2261813.658,
    "Contacts": 2047800,
    "wikipedia": 2218288.6,
    "uci": 6714558.3,
    "SocialEvo": 18711359,
    "mooc": 2250151.6,
    "lastfm": 120235473,
    "enron": 93431801,
    "Flights": 106,
    "UNvote": 2019686400,
    "CanParl": 347155200,
    "USLegis": 10,
    "UNtrade": 883612800
}

DF_test_orig = DF[DF['ts'] > test_time_start[dataset]]

lambda_orig = len(DF_test_orig)/(DF_test_orig['ts'].max() - DF_test_orig['ts'].min())

DF_r1 = copy.copy(DF_test_orig)
ts_shuffled = list(DF_r1['ts']) 
np.random.shuffle(ts_shuffled)

# assign the ts_shuffles back to the dataframe
DF_r1['ts'] = ts_shuffled

X = np.load(f"{DATA_DIR}/{dataset}/ml_{dataset}.npy")
X_trainval = X[:1+len(DF)-len(DF_test_orig)]
X_test = X[1+len(DF)-len(DF_test_orig):]
len(X_trainval) + len(X_test)

X_test_new = np.zeros(X_test.shape)
test_new_df = DF_r1.sort_values(by='ts')

for j, idx in enumerate(test_new_df['idx']):
    X_test_new[j] = X[idx]

X_new = np.vstack([X_trainval, X_test_new])

np.save(f"{DATA_DIR}/{dataset}/shuffle_{sample}_ml_{dataset}.npy", X_new)

DF_trainval = DF[DF['ts'] <= test_time_start[dataset]]
DF_new = pd.concat([DF_trainval, test_new_df ], ignore_index=True)

DF_new.reset_index(drop=True, inplace=True)

DF_new['Unnamed: 0'] = DF_new.index
DF_new['idx'] = DF_new.index + 1

DF_new.to_csv(f"{DATA_DIR}/{dataset}/shuffle_{sample}_ml_{dataset}.csv", index=False)






