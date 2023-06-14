#!/bin/bash

out_dir=training_output_8by8
# timestamp = scaling_experiment

# i = num_experiments
for ((i=0;i<=0;i++)); do
    for scenario in linear multiplicative translations_rotations xor; do
        for background in uncorrelated correlated imagenet; do
            pipenv run python -m training.train_models $scenario $background $i $out_dir
            pipenv run python -m training.train_models_keras $scenario $background $i $out_dir
        done
    done
done