#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../

# Creating directories to keep data results and figures
#
# Data will be downloaded automatically
mkdir -p /tmp/dl_bounds/data
mkdir -p /tmp/dl_bounds/weights
mkdir -p /tmp/dl_bounds/results/raw
mkdir -p /tmp/dl_bounds/results/db
mkdir -p /tmp/dl_bounds/results/figures

# Running three types of experiments: computing norms,
# hessian spectral norms, and the sharpness,
# in both "bad" and "good" minimum type of experiment (see accomp. pdf)
#
for experiment in norms hessian sharpness
do
    for bad_min in false true
    do
      python src/eval.py \
	    --dataset=mnist \
	    --network=mlp_20-20 \
	    --passes=32 \
	    --batch_size=128 \
	    --learning_rate=0.05 \
	    --experiment=${experiment} \
	    --result_filename=mnist_demo \
	    --optimizer=sgd \
	    --log2_snapshots \
	    --flip_labels=0 \
	    --bad_min=${bad_min} \
	    --data_path=/tmp/dl_bounds/data \
	    --weight_path=/tmp/dl_bounds/weights \
	    --result_path=/tmp/dl_bounds/results/raw \
	    --result_filename=/tmp/dl_bounds/results/raw/demo_mnist_${experiment}_bad_min_${bad_min} \
	    --logtostderr
    done
done

# Collects results (pickled dicts into sqlite DB)
#
python src/collect_results.py \
      --do=collect \
      --source_dir=/tmp/dl_bounds/results/raw \
      --target_db=/tmp/dl_bounds/results/db/demo_mnist.db \
      --logtostderr

# Produces (LaTeX) tables as PDF files
#
for metric_type in norms phased
do
    python src/results.py \
	  --do=table_${metric_type} \
	  --db=/tmp/dl_bounds/results/db/demo_mnist.db \
	  --network='mlp%' \
	  --dataset=mnist \
	  --groupby_param=learning_rate \
	  --pdf=/tmp/dl_bounds/results/figures/demo_mnist_${metric_type}.pdf \
	  --logtostderr
done
