import os
import sys
import random
import numpy as np
import torch
from typing import Dict
from datetime import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedShuffleSplit

from common import DataRecord, SEED
from utils import load_json_file, dump_as_pickle
from data.data_utils import (
    generate_backgrounds,
    generate_imagenet,
    generate_fixed,
    generate_translations_rotations,
    generate_xor,
    generate_distractors,
    normalise_data,
    scale_to_bound
)

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

data_generators = {
    'linear': generate_fixed,
    'multiplicative': generate_fixed,
    'translations_rotations': generate_translations_rotations,
    'xor': generate_xor
}

def get_background(bg_type: str, base: np.ndarray, imagenet: np.ndarray, config: Dict, shape: tuple) -> np.ndarray:
    if bg_type == 'correlated':
        return np.array([
            gaussian_filter(np.reshape(bg, shape), config['smoothing_sigma']).reshape(-1)
            for bg in base
        ])
    elif bg_type == 'imagenet':
        return imagenet
    elif bg_type == 'white':
        return base
    else:
        raise ValueError(f"Unknown background type: {bg_type}")

def combine_inputs(signal, background, manipulation_type, alpha, distractor=None):
    if isinstance(alpha, list) and len(alpha) == 1 and isinstance(alpha[0], list):
        alpha = alpha[0]

    if manipulation_type == 'multiplicative':
        if isinstance(alpha, list):
            raise ValueError("Multiplicative expects scalar alpha")
        signal = 1 - alpha * signal
        signal, background, _ = normalise_data(signal, background)
        return signal * background

    elif manipulation_type == 'additive':
        if isinstance(alpha, list):
            raise ValueError("Additive expects scalar alpha")
        signal, background, _ = normalise_data(signal, background)
        return alpha * signal + (1 - alpha) * background

    elif manipulation_type == 'division':
        if isinstance(alpha, list):
            raise ValueError("Division expects scalar alpha")
        signal, background, _ = normalise_data(signal, background)
        scale = 100.0
        signal = scale * signal
        background = scale * background

        with np.errstate(divide='ignore', invalid='ignore'):
            result = background / (alpha * signal + (1 - alpha) + 1e-8)
            result[np.isnan(result)] = 0.0
            result[np.isinf(result)] = 0.0
        return result

    elif manipulation_type == 'distractor_additive':
        if not isinstance(alpha, list) or len(alpha) != 3:
            raise ValueError(f"Expected list of 3 alphas for distractor_additive, got {alpha}")
        signal, background, distractor = normalise_data(signal, background, distractor)
        a_s, a_d, a_b = alpha
        return a_s * signal + a_d * distractor + a_b * background

    elif manipulation_type == 'distractor_multiplicative':
        if not isinstance(alpha, list) or len(alpha) != 3:
            raise ValueError(f"Expected list of 3 alphas for distractor_multiplicative, got {alpha}")
        a_s, a_d, a_b = alpha
        signal = 1 - a_s * signal
        distractor = 1 - a_d * distractor
        signal, background, distractor = normalise_data(signal, background, distractor)
        return signal * distractor * background
    
    elif manipulation_type == 'distractor_division':
        if not isinstance(alpha, list) or len(alpha) != 3:
            raise ValueError(f"Expected list of 3 alphas for distractor_division, got {alpha}")
        a_s, a_d, a_b = alpha
        signal = a_s * signal
        distractor = a_d * distractor
        signal, background, distractor = normalise_data(signal, background, distractor)
        scale = 100.0
        signal = scale * signal
        distractor = scale * distractor 
        background = scale * background
        return (a_b * background) / (a_s * signal + (1 - a_s) + 5e-9) / (a_d * distractor + (1 - a_d) + 5e-9) 

    else:
        raise ValueError(f"Unknown manipulation type: {manipulation_type}")

def data_generation_process(config: Dict, output_dir: str):
    image_shape = (np.array(config["image_shape"]) * config["image_scale"]).astype(int).tolist()
    base_shape = tuple(image_shape)

    for i in range(config['num_experiments']):
        base_backgrounds = generate_backgrounds(config['sample_size'], config['mean_data'], config['var_data'], image_shape)
        imagenet_backgrounds = (
            generate_imagenet(config['sample_size'], image_shape)
            if config['use_imagenet'] else None
        )

        for param_name, params in config['parameterizations'].items():
            for pattern_scale in params['pattern_scales']:
                config['pattern_scale'] = pattern_scale
                patterns = data_generators[param_name](params=config, image_shape=image_shape)
                ground_truths = patterns.copy()

                distractors = generate_distractors(config, image_shape=image_shape, N=config['sample_size'])

                background_types = ["white", "correlated", "imagenet"]

                for manip_type, snr_by_bg in params['snrs'].items():
                    for bg_type in background_types:
                        if bg_type == 'imagenet' and not config['use_imagenet']:
                            continue

                        if bg_type in ['white', 'correlated']:
                            background_data = get_background(bg_type, base_backgrounds, imagenet_backgrounds, config, base_shape)
                        else:
                            background_data = imagenet_backgrounds

                        # alpha_set = snr_by_bg.get(bg_type)
                        if isinstance(snr_by_bg, dict):
                            alpha_set = snr_by_bg.get(bg_type)
                            if alpha_set is None:
                                print(f"No SNRs defined for {param_name} / {manip_type} / {bg_type}")
                                continue
                        elif isinstance(snr_by_bg, list):
                            alpha_set = snr_by_bg  # assume this is directly the list of alphas
                        else:
                            raise ValueError(f"Unexpected snr_by_bg type for {param_name} / {manip_type}: {type(snr_by_bg)}")


                        if alpha_set is None:
                            raise ValueError(f"No SNRs defined for {param_name} / {manip_type} / {bg_type}")

                        for alpha in alpha_set:
                            if "distractor" in manip_type and not (isinstance(alpha, list) and len(alpha) == 3):
                                raise ValueError(f"Expected list of 3 alphas for {manip_type}, got {alpha}")

                            x = combine_inputs(
                                signal=patterns.copy(),
                                background=background_data.copy(),
                                manipulation_type=manip_type,
                                alpha=alpha,
                                distractor=distractors if "distractor" in manip_type else None
                            )

                            scale = 1 / np.max(np.abs(x))
                            x = np.apply_along_axis(scale_to_bound, 1, x, scale)

                            y = torch.ravel(torch.cat((torch.zeros(config['sample_size'] // 2, 1), torch.ones(config['sample_size'] // 2, 1))))
                            x = torch.as_tensor(x, dtype=torch.float16)

                            sss = StratifiedShuffleSplit(n_splits=1, test_size=config['test_split'], random_state=SEED)
                            train_idx, val_test_idx = next(sss.split(x, y))

                            x_train, y_train = x[train_idx], y[train_idx]
                            masks_train = ground_truths[train_idx]

                            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
                            val_idx, test_idx = next(sss2.split(x[val_test_idx], y[val_test_idx]))

                            x_val, y_val = x[val_test_idx][val_idx], y[val_test_idx][val_idx]
                            x_test, y_test = x[val_test_idx][test_idx], y[val_test_idx][test_idx]
                            masks_val, masks_test = ground_truths[val_test_idx][val_idx], ground_truths[val_test_idx][test_idx]

                            alpha_str = "_".join([f"{a:.2f}" for a in alpha]) if isinstance(alpha, list) else f"{alpha:.2f}"
                            
                            manip_str = manip_type
                            # if 'distractor' in manip_type:
                            #     manip_str = 'distractor'
                            #     if manip_type == 'distractor_division':
                            #         manip_str = 'distractor_division'
                            exp_str = ''
                            if config['num_experiments'] > 1:
                                exp_str = f'_{str(i)}'
                            scenario_key = f"{param_name}_{manip_str}_{config['image_scale']}d{pattern_scale}p_{alpha_str}_{bg_type}{exp_str}"
                            record = DataRecord(x_train, y_train, x_val, y_val, x_test, y_test, masks_train, masks_val, masks_test)
                            dump_as_pickle(record, output_dir, scenario_key)


# Usage: python -m data.generate_data {data_config} {mode} {optional: output file dir}
# data_config is the path to your data_config file; mode = 8x8 or 64x64;
# if you don't include the third argument then the data will save to a folder named with current timestamp
def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'data_config'
    mode = sys.argv[2] if len(sys.argv) > 2 else '64x64'
    output_override = sys.argv[3] if len(sys.argv) > 3 else None

    base_config = load_json_file(f'data/{config_file}.json')
    config = base_config.copy()
    mode_config = config["modes"][mode]

    for k, v in mode_config.items():
        config[k] = v
    config["parameterizations"] = mode_config["parameterizations"]

    required = ["mean_data", "var_data", "sample_size", "manipulation", "patterns", "output_dir"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    output_path = os.path.join(config['output_dir'], output_override) if output_override else os.path.join(config['output_dir'], datetime.now().strftime(f"%Y-%m-%d-%H-%M-{mode}"))
    Path(output_path).mkdir(parents=True, exist_ok=True)

    data_generation_process(config, output_path)

if __name__ == '__main__':
    main()
