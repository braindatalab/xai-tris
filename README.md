# XAI-Tris

This repository contains all the experiments presented in the paper: *"XAI-TRIS: Non-linear benchmarks to quantify ML explanation performance"*.

We use `Pipfiles` to create Python environments. Our main experiments use [Captum](https://github.com/pytorch/captum), however inclusion of some methods (PatternNet, PatternAttribution, DTD) from the [innvestigate](https://github.com/albermax/innvestigate) library requires particular dependencies, such as `Python=3.7`, `tensorflow==1.13.1/1.14`, and `keras==2.2.4`. If you do not wish to include these methods, you are able to remove `tensorflow` and `keras` from the Pipfile and change any specific package requirements to `*` along with the Python version of choice (we have also tested this with `Python 3.10`). 

To setup pipenv, run:
```shell
pipenv install
```
and then
```shell
pipenv shell
```

In four steps we can reproduce the results: (i) generate tetromino data, (ii) train models (iii) apply XAI methods, (iv) run the evaluation steps and generate plots.

#### Generate data 

##### Download ImageNet data
If one wishes to use the data from the ImageNet-1k, we sourced it from [HuggingFace](https://huggingface.co/datasets/imagenet-1k) by making an account and agreeing to the license terms. Please respect these license terms when using this data. Download and extract images into a folder `imagenet_images/` in the top level of this repository. We used the `N=50,000` validation set due to the appropriate volume size for our datasets and benchmarks. 

##### 64x64 Data
Once ImageNet data is downloaded and extracted above (if using it), configuration is set by the `data/data_config.json` file, with fields explained in data/README.md. Generate data via:
```shell
python -m data.generate_data 
```

##### 8x8 Data
One can also generate the original 8x8 data (without ImageNet backgrounds), which is quicker and less computationally demanding to run, and also produces interesting results. Configuration here is set by the `data/data_config_8by8.json` file. Generate data via:
```shell
python -m data.generate_data data_config_8by8
```

#### Train models

Update the `data_path` parameter of the `training/training_config.json` with the path to the folder of freshly generated pickle files containing data, of the form `artifacts/tetris/data/YYYY-MM-DD-HH-mm`.

To train models for a particular scenario, background, and experiment number:

##### Torch
```shell
python -m training.train_models SCENARIO BACKGROUND EXP_NUM OUT_DIR 
```
where `SCENARIO=[linear|multiplicative|translations_rotations|xor]`, `BACKGROUND=[uncorrelated|correlated|imagenet]`, `EXP_NUM=[0,...,N]`, and `OUT_DIR=[str]`. For example `python -m training.train_models linear uncorrelated 0 neurips_training` will train models for the `linear` scenario with `uncorrelated` (WHITE) background for the `0`th experiment and output models and logs to the `artifacts/tetris/training/neurips_training` directory.

##### Keras
```shell
python -m training.train_models_keras SCENARIO BACKGROUND EXP_NUM OUT_DIR 
```
where input args are the same as specified above.

##### Train all models
As you may have noticed, you will have to run this individually for all scenario-background pairs. You can also input regex into the `SCENARIO` and `BACKGROUND` input args, so running `python -m training.train_models * * EXP_NUM OUT_DIR` will train all scenario and background types for PyTorch.

One can use a somewhat approach like `./train_all.sh` which simply runs all parameterizations one after another. We can also provide a script combining all parameterizations for each frameworks if needed.

#### XAI Methods
Update the `data_path`, `training_output`, and `num_experiments` parameters of the `xai/xai_config.json` file with the same `data_path` as used in the training step, and the corresponding training output path of the form `artifacts/tetris/training/OUT_DIR`.

To calculate explanations for a particular scenario and background and across all experiments:
##### Torch
```shell
python -m xai.calculate_explanations SCENARIO BACKGROUND OUT_DIR 
```
where `SCENARIO=[linear|multiplicative|translations_rotations|xor]`, `BACKGROUND=[uncorrelated|correlated|imagenet]`, and `OUT_DIR=[str]` as in the training step. Note: as of 14/06/23, you can't use regex for `SCENARIO` and `BACKGROUND` as in the training step, as it is incompatible with the input to the next step. If/when changed, this README will be updated.

##### Keras
```shell
python -m xai.calculate_explanations_keras SCENARIO BACKGROUND OUT_DIR 
```
where input args are the same as specified above.


#### Run evaluation metrics
Update the parameters `data_path`, `training_path`, `xai_path`, and `num_experiments` of `eval/eval_config.json`, and specify the desired output directory here too, in the form `./artifacts/tetris/eval/OUT_DIR`.
##### Torch
```shell
python -m eval.quantitative_metrics SCENARIO BACKGROUND 
```
where `SCENARIO=[linear|multiplicative|translations_rotations|xor]` and `BACKGROUND=[uncorrelated|correlated|imagenet]`. Note: as of 14/06/23, you can't use regex for `SCENARIO` and `BACKGROUND` as in the training step, as it is incompatible with the input to the next step. If/when changed, this README will be updated.

##### Keras
```shell
python -m eval.quantitative_metrics_keras SCENARIO BACKGROUND OUT_DIR 
```
where input args are the same as specified above.

#### Generate plots
With the same parameters of  `eval/eval_config.json` as before, simply run:
##### Qualitative plots (local and global)
```shell
python -m eval.plot_qualitative
```
##### Quantitative plots
```shell
python -m eval.plot_quantitative
```

#### Questions and Issues
If you have any questions and/or issues with the above, please do not hesitate to contact the authors!
