#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import sys
data = pd.read_csv(sys.argv[1])
sns.relplot(x="epoch", y="accuracy", kind='line', col='dataset', row='setting', data=data, hue="model", style="pretraining", markers=False, ci="sd")
plt.savefig(sys.argv[1] + '.png')

