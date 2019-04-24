#!/usr/bin/env python3
""" Plots the accuracy distribution """
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import sys

def select(data, setting, model, pretraining=200):
    df = data
    df = df[df["setting"] == setting]
    df = df[df['model'] == model]
    df = df[df['pretraining'] == pretraining]
    return df['accuracy'].values



import os

if len(sys.argv) < 2:
    raise ValueError("Too few arguments")

data = pd.read_csv(sys.argv[1])
outdir = sys.argv[1] + '-dists.d'
os.makedirs(outdir, exist_ok=True)

import itertools
i = 0
for setting, model, pretraining in itertools.product("AB",
                                                     ["MLP","GCN","GCN-64","GraphSAGE","GAT"],
                                                     [0, 200]):
    plt.figure(100+i)
    x = select(data, setting, model, pretraining)
    sns.distplot(x)
    fname = "{}-{}-{}.png".format(setting, model, pretraining)
    plt.savefig(os.path.join(outdir,fname))
    i+=1





