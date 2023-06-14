#!/bin/bash

out_dir=xai_output
# timestamp = scaling_experiment

# i = num_experiments
for ((i=0;i<=0;i++)); do
    for scenario in linear multiplicative translations_rotations xor; do
        for background in uncorrelated correlated imagenet; do
            pipenv run python -m eval.quantitative_metrics $scenario $background
            pipenv run python -m eval.quantitative_metrics_keras $scenario $background 
        done
    done
done