import os
import pickle
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as sk_metrics
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models import FeatureSubNet, InteractionSubNet
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils import measure_interactions
from nam.wrapper import NAMClassifier

device = (
    # torch.device("mps") if torch.backends.mps.is_available()
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


use_amp = device.type == "cuda"


class MLP(nn.Module):   
    def __init__(self, n_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dim, n_dim // 2),
            nn.ReLU(),
            nn.Linear(n_dim // 2, n_dim // 4),
            nn.ReLU(),
            nn.Linear(n_dim // 4, n_dim // 8),
            nn.ReLU(),
            nn.Linear(n_dim // 8, 1)  # Single output logit for binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape: (N,)
    
def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=100, batch_size=512, lr=1e-3, patience=50):
    model = MLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_weights = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def parse_filename(path):
    base = os.path.basename(path).replace(".pkl", "")

    # Split based on known format ending: _<snr>_<background>
    parts = base.split("_1d1p_")
    if len(parts) != 2:
        raise ValueError(f"Filename format not recognized: {base}")

    scenario_and_manip, rest = parts
    snr_and_bg = rest.rsplit("_", 1)  # split SNR vs background

    if len(snr_and_bg) != 2:
        raise ValueError(f"Could not parse SNR and background from: {rest}")
    
    snr_str, background = snr_and_bg

    # Now split scenario and manip from the RIGHT
    scenario_parts = scenario_and_manip.rsplit("_", 1)
    if len(scenario_parts) != 2:
        raise ValueError(f"Could not parse scenario and manip_type from: {scenario_and_manip}")
    
    scenario, manip_type = scenario_parts

    try:
        snr_value = eval(snr_str.replace("_", ","))
    except Exception:
        snr_value = snr_str

    return scenario, manip_type, snr_value, background

# def parse_filename(path):
#     base = os.path.basename(path).replace(".pkl", "")

#     # Updated regex to allow compound manipulation types like "distractor_additive"
#     match = re.match(
#         r"(.+?)_([a-zA-Z0-9]+)_\d+d\d+p_([0-9_.\[\], ]+?)_([a-zA-Z]+)$",
#         base
#     )

#     if not match:
#         raise ValueError(f"Filename format not recognized: {base}")

#     scenario, manip_type, snr_str, background = match.groups()

#     try:
#         snr_value = eval(snr_str.replace("_", ","))
#     except Exception:
#         snr_value = snr_str

#     return scenario, manip_type, snr_value, background


def quadratic_features(xs):
    inds_0, inds_1 = np.triu_indices(xs.shape[1], 0)
    quadratic = np.zeros((xs.shape[0], xs.shape[1] + len(inds_0)))
    for i, x in enumerate(xs):
        outer = np.outer(x, x)
        quadratic[i] = np.concatenate([x, outer[inds_0, inds_1]])
    return quadratic


def compile_or_script(model):
    if device.type == "cuda":
        return torch.compile(model)
    else:
        return torch.jit.script(model)
    

def FAST(X_train, y_train, n_interactions, init_score=None, feature_names=None, feature_types=None):
    import time
    t0 = time.time()
    interactions = measure_interactions(
        X_train,
        y_train,
        interactions=n_interactions, 
        init_score=init_score,  # Can be a model or initial scores; set to None if not used
        feature_names = feature_names,
        feature_types = feature_types
    )
    
    pairs = []
    for (i, j), _ in interactions:
        pairs.append((i,j))
    return pairs, time.time() - t0

def evaluate_models(pkl_path):
    with open(pkl_path, "rb") as f:
        record = pickle.load(f)

    x_train = record.x_train.detach().numpy()
    y_train = record.y_train.detach().numpy().astype(int)
    x_val = record.x_val.detach().numpy()
    y_val = record.y_val.detach().numpy().astype(int)
    x_test = record.x_test.detach().numpy()
    y_test = record.y_test.detach().numpy().astype(int)

    results = {}


    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    model = NAMClassifier(
            num_epochs=500,
            num_learners=1,
            metric='auroc',
            interaction_pairs=[],
            hidden_sizes = [16,16,16],
            early_stop_mode='max',
            device='cpu',
            monitor_loss=False,
            n_jobs=10,
            random_state=2025
        )

    model.fit(X_train_tensor, y_train_tensor)
    pred = model.predict_proba(X_test_tensor)

    results['nam'] = sk_metrics.accuracy_score(y_test, pred)

    interaction_pairs = FAST(X_train_tensor, y_train, n_interactions=128)

    model128 = NAMClassifier(
            num_epochs=500,
            num_learners=1,
            metric='auroc',
            interaction_pairs=interaction_pairs,
            hidden_sizes = [16,16,16],
            early_stop_mode='max',
            device='cuda',
            monitor_loss=False,
            n_jobs=10,
            random_state=2025
        )

    model128.fit(X_train_tensor, y_train_tensor)
    pred128 = model128.predict_proba(X_test_tensor)

    results['nam128'] = sk_metrics.accuracy_score(y_test, pred128)

    # # ==== Quadratic model ====
    Xq_train = quadratic_features(x_train)
    Xq_test = quadratic_features(x_test)
    quad_clf = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000)
    quad_clf.fit(Xq_train, y_train)
    results['qlr'] = quad_clf.score(Xq_test, y_test)

    # ==== MLP ====
    # mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000)
    # mlp.fit(x_train, y_train)
    # results['mlp'] = mlp.score(x_test, y_test)
    
    # ==== PyTorch MLP ====
    mlp_model = train_mlp(
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        input_dim=X_train_tensor.shape[1]
    )
    mlp_model.eval()
    with torch.no_grad():
        test_logits = mlp_model(X_test_tensor)
        test_preds = (torch.sigmoid(test_logits) > 0.5).int().cpu().numpy()

    results['mlp'] = sk_metrics.accuracy_score(y_test, test_preds)


    return results

out_fname = "training_results_new.csv"
if os.path.exists(out_fname):
    df = pd.read_csv(out_fname)
    done_keys = set(zip(df["scenario"], df["manip_type"], df["snr_value_str"], df["background"]))
else:
    df = pd.DataFrame()
    done_keys = set()

records = []

for pkl_file in glob.glob("artifacts/tetris/data/line_search/*.pkl"):
    scenario, manip_type, snr_value, background = parse_filename(pkl_file)
    snr_value_str = str(snr_value) if isinstance(snr_value, (list, tuple)) else str(round(snr_value, 4))
    
    key = (scenario, manip_type, snr_value_str, background)
    if key in done_keys:
        continue

    model_scores = evaluate_models(pkl_file)

    record = {
        "scenario": scenario,
        "manip_type": manip_type,
        "snr_value": snr_value,
        "snr_value_str": snr_value_str,
        "background": background,
        **model_scores
    }

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(out_fname, index=False)
    done_keys.add(key)

    print(record)


# df = pd.DataFrame(records)

# # Clean up and formatting
# df["target_diff"] = np.abs(df["ebm"] - 0.80)
# df["snr_value_str"] = df["snr_value"].apply(lambda x: str(x) if isinstance(x, list) else x)

# # Check for duplicate (scenario, background, snr_value) combos
# dups = df[df.duplicated(subset=["scenario", "manip_type", "snr_value_str", "background"], keep=False)]
# if not dups.empty:
#     print("Duplicate entries:")
#     print(dups)


# out_fname = "training_results"
# df.to_csv(f"{out_fname}.csv", index=False)
# print(f"Saved {out_fname}.csv")
