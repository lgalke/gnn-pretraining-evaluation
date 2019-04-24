#!/usr/bin/env python3
""" This tiny script checks how many rows are there per dataset/setting/model/pretraining config in full results file """
import pandas as pd
import sys
ROWS_PER_RUN = 33
data = pd.read_csv(sys.argv[1])
for key, group in data.groupby(["dataset", "setting", "model","pretraining"]):
    print("Runs for", key, ":", (group.count()/ROWS_PER_RUN).mean())

# sns.relplot(x="epoch", y="accuracy", kind='line', col='dataset', row='setting', data=data, hue="model", style="pretraining", markers=False, ci="sd")
# plt.savefig(sys.argv[1] + '.png')


