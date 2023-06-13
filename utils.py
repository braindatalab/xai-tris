import json
import pickle as pkl
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Dict, Any

def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file

def load_pickle(file_path: str) -> None:
    with open(file_path, 'rb') as file:
        return pkl.load(file)

def dump_as_pickle(data: Any, output_dir: str, file_name: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = join(output_dir, f'{file_name}.pkl')
    print(output_file)
    with open(output_file, 'wb') as file:
        pkl.dump(data, file)


def append_date(s: str) -> str:
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f'{s}-{date}'
