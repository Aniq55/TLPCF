import sys
sys.path.append("/home/chri6578/Documents/gttp/") 
from src.utils import *
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()
from tqdm import tqdm
import pandas as pd
import copy
import argparse

parser = argparse.ArgumentParser(description="Intensify.")
parser.add_argument('-d', '--dataset', type=str, help='The directory path')
parser.add_argument('-m', '--m', type=int, help='The multiples of intensity')
parser.add_argument('-s', '--sample', type=int, help='The sample id')

parser.add_argument("args", nargs='*', help="List of arguments")
args = parser.parse_args()

dataset = args.dataset
m = args.m
sample = args.sample

DF = pd.read_csv(f"/home/chri6578/Documents/gttp/data/{dataset}/ml_{dataset}.csv")   

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

def add_random(row):
    return row['ts'] + np.random.uniform(-1/lambda_orig, 1/lambda_orig)

DF_all = {}
for i in range(m):
    DF_all[i] = copy.copy(DF_test_orig)
    DF_all[i]['ts'] = DF_all[i].apply(add_random, axis=1)
    DF_all[i]['id_old'] = DF_all[i]['Unnamed: 0']

DF_list = [DF_all[i] for i in range(m)]
union_df = pd.concat(DF_list, ignore_index=True)

X = np.load(f"/home/chri6578/Documents/gttp/data/{dataset}/ml_{dataset}.npy")
X_trainval = X[:1+len(DF)-len(DF_test_orig)]
X_test = X[1+len(DF)-len(DF_test_orig):]

X_test_new = np.zeros((X_test.shape[0]*m, X_test.shape[1]))

test_new_df = union_df.sort_values(by='ts')

for j, idx in enumerate(test_new_df['idx']):
    X_test_new[j] = X[idx]

X_new = np.vstack([X_trainval, X_test_new])

np.save(f"/home/chri6578/Documents/gttp/data/{dataset}/intense_{m}_{sample}_ml_{dataset}.npy", X_new)

DF_trainval = DF[DF['ts'] <= test_time_start[dataset]]
DF_new = pd.concat([DF_trainval, test_new_df ], ignore_index=True)


DF_new.reset_index(drop=True, inplace=True)

DF_new['Unnamed: 0'] = DF_new.index
DF_new['idx'] = DF_new.index + 1

DF_new.to_csv(f"/home/chri6578/Documents/gttp/data/{dataset}/intense_{m}_{sample}_ml_{dataset}.csv", index=False)



