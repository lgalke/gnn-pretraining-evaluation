#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Computes the Jenson-Shannon divergence between Settings A and B """
import sys
import pandas as pd
import numpy as np

from scipy.spatial.distance import jensenshannon
data = pd.read_csv(sys.argv[1])

pretraining = int(sys.argv[2]) if len(sys.argv) > 2 else 200

print("Pretraining =", pretraining)
data = data[data['pretraining'] == pretraining]


for model in ["MLP","GCN","GCN-64","GraphSAGE", "GAT"]:
    df_model = data[data['model'] == model]
    # jsdivs = []
    # for epoch in range(33):
    #     relevant = subset[subset['epoch'] == epoch]
    settingA = df_model[df_model['setting'] == 'A']
    settingB = df_model[df_model['setting'] == 'B']

    accuracyA = settingA['accuracy'].values
    accuracyB = settingB['accuracy'].values
    # jsdivs.append(jensenshannon(accuracyA, accuracyB))

    print("\tModel:", model)
    print("\tJensen-Shannon divergence between A/B: %.4f" % jensenshannon(accuracyA, accuracyB))

