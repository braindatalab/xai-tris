import json
import itertools
import os
import argparse
import shutil
from subprocess import run
import numpy as np

BASE_CONFIG = "data/data_config_linesearch.json"
MODE = "8x8"
TARGET_SCENARIOS = ["linear", "multiplicative", "translations_rotations", "xor"]
# TARGET_SCENARIOS = ["multiplicative"]

# OUT_DIR = "artifacts/tetris/data/linesearch"
OUT_DIR = "artifacts/tetris/data/line_search"
CONFIG_TMP_DIR = "data/temp_configs"

def load_config():
    with open(BASE_CONFIG) as f:
        return json.load(f)

def save_config(config, path):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def run_generation(config_file_base, output_folder):
    # run(["python", "-m", "data.generate_data", config_file_base, MODE, output_folder])
    run(["python", "-m", "data.generate_data", f"temp_configs/{config_file_base}", MODE, output_folder])


def format_snr(snr):
    if isinstance(snr, list):
        return "_".join(f"{v:.2f}" for v in snr)
    return f"{snr:.2f}"

def snr_output_key(scenario, manipulation_type, scale, snr_val, bg_type):
    snr_str = format_snr(snr_val)
    return f"{scenario}_{manipulation_type}_{scale}d1p_{snr_str}_{bg_type}.pkl"

# def run_sweep(dry_run=False, skip_existing=False, keep_configs=False, n_snr_vals=10):
#     os.makedirs(CONFIG_TMP_DIR, exist_ok=True)
#     os.makedirs(OUT_DIR, exist_ok=True)

#     # Sweep ranges
#     scalar_snr_values = [round(v, 3) for v in np.linspace(0.05, 1.0, n_snr_vals)]
#     signal_alphas = [round(v, 3) for v in np.linspace(0.05, 0.95, n_snr_vals)]
#     triplet_snr_values = [[a, round((1 - a) / 2, 3), round((1 - a) / 2, 3)] for a in signal_alphas]

#     # manipulation_type_lookup = {
#     #     "linear": ["distractor_additive", "distractor_division"],
#     #     "multiplicative": ["distractor_multiplicative", "division", "distractor_division"],
#     #     "translations_rotations": ["distractor_additive", "division", "distractor_division"],
#     #     "xor": ["distractor_additive", "division", "distractor_division"]
#     # }

#     manipulation_type_lookup = {
#         "linear": ["additive", "distractor_additive"],
#         "multiplicative": ["multiplicative", "distractor_multiplicative"],
#         "translations_rotations": ["additive", "distractor_additive"],
#         "xor": ["additive", "distractor_additive"]
#     }

#     for scenario in TARGET_SCENARIOS:
#         for manip_type in manipulation_type_lookup[scenario]:
#             is_triplet = manip_type.startswith("distractor")
#             snr_list = triplet_snr_values if is_triplet else scalar_snr_values

#             for snr_val in snr_list:
#                 config = load_config()
#                 mode_cfg = config["modes"][MODE]
#                 image_scale = mode_cfg["image_scale"]
#                 param = mode_cfg["parameterizations"][scenario]

#                 param["manipulation_type"] = manip_type

#                 if is_triplet:
#                     param["snrs"] = {
#                         manip_type: {
#                             "white": [snr_val],
#                             "correlated": [snr_val]
#                             # "imagenet": [snr_val]
#                         }
#                     }
#                 else:
#                     scalar = snr_val if isinstance(snr_val, float) else snr_val[0]
#                     param["snrs"] = {
#                         manip_type: {
#                             "white": [scalar],
#                             "correlated": [scalar]
#                             # "imagenet": [scalar]
#                         }
#                     }

#                 mode_cfg["parameterizations"][scenario] = param
#                 config["modes"][MODE] = mode_cfg

#                 fname_base = f"data_config_search_{scenario}_{manip_type}_{format_snr(snr_val)}"
#                 config_path = os.path.join(CONFIG_TMP_DIR, f"{fname_base}.json")
#                 save_config(config, config_path)

#                 print(f"→ Running {scenario} / {manip_type} with SNR = {snr_val}")
#                 if not dry_run:
#                     run_generation(fname_base, "line_search")

#                 if not keep_configs:
#                     os.remove(config_path)


def run_sweep(dry_run=False, skip_existing=False, keep_configs=False, n_snr_vals=None):
    os.makedirs(CONFIG_TMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Use explicit SNR values
    scalar_snr_values = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    triplet_snr_values = [[a, (1 - a) / 2, (1 - a) / 2] for a in scalar_snr_values]

    manipulation_type_lookup = {
        "linear": ["additive", "distractor_additive"],
        "multiplicative": ["multiplicative", "distractor_multiplicative"],
        "translations_rotations": ["additive", "distractor_additive"],
        "xor": ["additive", "distractor_additive"]
    }

    for scenario in TARGET_SCENARIOS:
        for manip_type in manipulation_type_lookup[scenario]:
            is_triplet = manip_type.startswith("distractor")
            snr_list = triplet_snr_values if is_triplet else scalar_snr_values

            for snr_val in snr_list:
                config = load_config()
                mode_cfg = config["modes"][MODE]
                image_scale = mode_cfg["image_scale"]
                param = mode_cfg["parameterizations"][scenario]
                param["manipulation_type"] = manip_type

                snr_value_for_config = snr_val if is_triplet else snr_val  # already scalar or triplet

                param["snrs"] = {
                    manip_type: {
                        "white": [snr_value_for_config],
                        "correlated": [snr_value_for_config]
                    }
                }

                mode_cfg["parameterizations"][scenario] = param
                config["modes"][MODE] = mode_cfg

                fname_base = f"data_config_search_{scenario}_{manip_type}_{format_snr(snr_val)}"
                config_path = os.path.join(CONFIG_TMP_DIR, f"{fname_base}.json")
                save_config(config, config_path)

                print(f"→ Running {scenario} / {manip_type} with SNR = {snr_val}")
                if not dry_run:
                    run_generation(fname_base, "line_search")

                if not keep_configs:
                    os.remove(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without running generation")
    parser.add_argument("--skip-existing", action="store_true", help="Skip SNR configs that already have .pkl outputs")
    parser.add_argument("--keep-configs", action="store_true", help="Retain intermediate JSON config files")
    parser.add_argument("--n-snr", type=int, default=10, help="Number of SNR steps to search")
    args = parser.parse_args()

    print("=== Unified SNR Sweep ===")
    run_sweep(dry_run=args.dry_run, skip_existing=args.skip_existing, keep_configs=args.keep_configs, n_snr_vals=args.n_snr)
