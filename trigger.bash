#!/bin/bash
# Script to reproduce all experiments from the paper


RUNS=100
INFERENCE=32
OUTFILE=results.txt
for DATASET in "cora" "citeseer" "pubmed"; do
	for MODEL in "GCN" "GCN-64" "GraphSAGE" "GAT"; do
		for PRETRAINING in 0 200; do
			# Setting A
			python3 evaluate.py --dataset "$DATASET" --model "$MODEL" --inference "$INFERENCE" --epochs "$PRETRAINING" --runs "$RUNS" --outfile "$OUTFILE"
			# Setting B
			python3 evaluate.py --dataset "$DATASET" --invert --model "$MODEL" --inference "$INFERENCE" --epochs "$PRETRAINING" --runs "$RUNS" --outfile "$OUTFILE"
		done
	done
done
