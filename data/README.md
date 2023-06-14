# Data Generation

This contains the steps required to generate data

#### Generate data 

##### Download ImageNet data
If one wishes to use the data from the ImageNet-1k, we sourced it from [HuggingFace](https://huggingface.co/datasets/imagenet-1k) by making an account and agreeing to the license terms. Please respect these license terms when using this data. Download and extract images into a folder `imagenet_images/` in the top level of this repository. We used the `N=50,000` validation set due to the appropriate volume size for our datasets and benchmarks. 

##### 64x64 Data
Configuration is set by the `data/data_config.json` file, with fields explained in data/README.md. Generate data via:
```shell
python -m data.generate_data 
```

##### 8x8 Data
One can also generate the original 8x8 data (without ImageNet backgrounds), which is quicker and less computationally demanding to run, and also produces interesting results. Configuration here is set by the `data/data_config_8by8.json` file. Generate data via:
```shell
python -m data.generate_data data_config_8by8
```

#### Config fields

The configuration files provided are given with the values used in our analyses, but for the curious, we detail each parameter here. Each config file is composed of two notable sections, with the second section outlined in the second table and the bottom row of the first table.

| Field           	| Recommended Values      	| Description                                                                                                                                                                                  	|
|-----------------	|-------------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| data_scenario   	| "tetris_data"           	| str, currently defunct, identifier for datasets (for future inclusion of other benchmark datasets of different types)                                                                        	|
| output_dir      	| "artifacts/tetris/data" 	| path to the output folder, where generated data will be stored in a timestamped subfolder in the form YYYY-MM-DD-HH-mm                                                                       	|
| num_experiments 	| 1...N                   	| number of experiments to run (datasets to generate per scenario), we used 1 for 64x64 results and 25 for 8x8. Note that minimal change was seen in results as num 8x8 experiments increased. 	|
| sample_size     	| 10000...N               	| total number of samples to generate, we generated 40000 for 64x64 results and 10000 for 8x8.                                                                                                 	|
| test_split      	| 0.1                     	| validation AND test split, so 0.1 = 90/5/5 train/val/test split. 0.2 was used for the 8x8 experiments.                                                                                                                            	|
| image_shape     	| [8,8]                   	| base image shape, leave fixed.                                                                                                                                                               	|
| image_scale     	| 8                       	| scale factor from base shape, i.e. image_scale=8 => 64x64 images, image_scale=1 => 8x8 images                                                                                                	|
| mean_data       	| 0                       	| mean of the Gaussian used to generate WHITE noise                                                                                                                                            	|
| var_data        	| 0.5                     	| variance of the Gaussian used to generate WHITE noise                                                                                                                                        	|
| smoothing_sigma 	| 3...10                  	| smoothing sigma value for the smoothing filter for CORR noise, where we used 10 for 64x64 data and 3 for 8x8.                                                                                	|
| manipulation    	| 1.0                     	| manipulation of tetromino patterns to the background, weighted by SNR in the data gen process. best to leave fixed.                                                                          	|
| positions       	| \[[8,8],[32,40]]         	| co-ordinate positions of the tetromino patterns specified below, we set as \[[8,8],[32,40]] for the 64x64 data and \[[1,1],[4,5]] for the 8x8 data.                                            	|
| patterns        	| ["t", "l"]              	| which tetromino patterns to incorporate. right now we support just "t" and "l" as they have four unique appearances for each 90 degree rotation.                                             	|
| use_imagenet    	| true                    	| boolean for whether or not to use imagenet. we used true for 64x64 data and false for 8x8 data.                                                                                              	|
| parameterizations | {'linear'\|'multiplicative'\|'translations_rotations'\|'xor'\|: {...}}                     | each key of the dict given as the scenario names listed here, followed by a sub-dict as broken down into each scenario in the table given below.

For the parameterizations sub-dict:

| Field             	| Recommended Values                                          	| Description                                                                                                                                                                                                                                         	|
|-------------------	|-------------------------------------------------------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| scenario          	| 'linear'\|'multiplicative'\|'translations_rotations'\|'xor' 	| name of the scenario, choose one as given (match with the higher-level key)                                                                                                                                                                         	|
| manipulation_type 	| 'additive'\|'multiplicative'                                	| manipulation type for data gen, either multiplicative or additive. Only use if multiplicative is specified above.                                                                                                                                   	|
| pattern_scales    	| [1,...,8]                                                     	| list of pattern scales defining the scaling of the tetromino pattern, where multiple can be given. We used just 8 for the 64x64 data (4 for the RIGID aka translations_rotations) to match the image_scale. In the 8x8 case where pattern_scale=1.           	|
| snrs              	| \[\[0.0,...,1.0],\[0.0,...,1.0],\[0.0,...,1.0]]                   	| list of snrs to generate for uncorrelated (WHITE), correlated (CORR), imagenet (IMAGENET) respectively in each sub-array. Multiple values between 0.0 and 1.0 can be given in each sub array. Leave the final sub-array blank if use_imagenet=false 	|