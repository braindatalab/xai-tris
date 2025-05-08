# Refactored data_utils.py with distractor support and modular pattern generation

import os
import random
import numpy as np
from PIL import Image
from glob import glob
from typing import List, Dict, Tuple
from scipy.ndimage import gaussian_filter
from common import SEED

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def get_patterns(params: Dict) -> List[np.ndarray]:
    manip = params['manipulation']
    scale = params['pattern_scale']

    t = np.array([[manip, 0], [manip, manip], [manip, 0]])
    l = np.array([[manip, 0], [manip, 0], [manip, manip]])

    pattern_dict = {
        't': np.kron(t, np.ones((scale, scale))),
        'l': np.kron(l, np.ones((scale, scale)))
    }

    return [pattern_dict[p] for p in params['patterns']]

def get_distractors(params: Dict) -> List[np.ndarray]:
    manip = params['manipulation']
    scale = params['pattern_scale']

    sq = np.array([[manip, manip], [manip, manip]])

    # pattern_dict = {
    #     'sq': np.kron(sq, np.ones((scale, scale))),
    # }

    return np.kron(sq, np.ones((scale, scale)))

def normalise_data(signal: np.ndarray, background: np.ndarray, distractor: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal /= np.linalg.norm(signal, ord='fro')
    b_norm = np.linalg.norm(background, ord='fro')
    background = background if b_norm == 0 else background / b_norm

    if distractor is not None:
        d_norm = np.linalg.norm(distractor, ord='fro')
        distractor = distractor if d_norm == 0 else distractor / d_norm
        return signal, background, distractor

    return signal, background, None

def scale_to_bound(row, scale):
    return row * scale

def generate_backgrounds(sample_size: int, mean_data: int, var_data: float, image_shape: List[int]) -> np.ndarray:
    shape = image_shape[0] * image_shape[1]
    return np.random.normal(mean_data, var_data, size=(sample_size, shape))

def generate_imagenet(sample_size: int, image_shape: List[int]) -> np.ndarray:
    image_paths = glob('./imagenet_images/*')
    h, w = image_shape
    flat_dim = h * w
    backgrounds = np.zeros((sample_size, flat_dim))
    i = 0

    for path in random.sample(image_paths, sample_size + len(image_paths) // 10):
        if i == sample_size:
            break

        img = Image.open(path)
        if img.size[0] < w or img.size[1] < h:
            continue

        scale = max(w / img.size[0], h / img.size[1])
        new_size = (int(scale * img.size[0]), int(scale * img.size[1]))
        img = img.resize(new_size)

        left = (img.size[0] - w) // 2
        top = (img.size[1] - h) // 2
        img = img.crop((left, top, left + w, top + h))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        grey = np.dot(np.array(img)[...,:3], [0.299, 0.587, 0.114]).reshape(flat_dim)
        backgrounds[i] = grey - grey.mean()
        i += 1

    return backgrounds


def embed_patterns(params: Dict, image_shape: List[int], patterns: List[np.ndarray], N: int, positions: List[Tuple[int, int]]) -> np.ndarray:
    out = np.zeros((N, image_shape[0], image_shape[1]))
    idx = 0
    for k, pattern in enumerate(patterns):
        pos = positions[k]
        for _ in range(N // len(patterns)):
            out[idx][pos[0]:pos[0]+pattern.shape[0], pos[1]:pos[1]+pattern.shape[1]] = pattern
            if params['pattern_scale'] > 3:
                out[idx] = gaussian_filter(out[idx], 1.5)
            idx += 1
    return out.reshape(N, -1)

def generate_fixed(params: Dict, image_shape: List[int]) -> np.ndarray:
    return embed_patterns(params, image_shape, get_patterns(params), params['sample_size'], params['positions'])

def generate_translations_rotations(params: Dict, image_shape: List[int]) -> np.ndarray:
    N = params['sample_size']
    out = np.zeros((N, image_shape[0], image_shape[1]))
    patterns = get_patterns(params)
    idx = 0
    for pat in patterns:
        for _ in range(N // len(patterns)):
            p = np.rot90(pat, k=np.random.randint(4))
            y = np.random.randint(0, image_shape[0] - p.shape[0] + 1)
            x = np.random.randint(0, image_shape[1] - p.shape[1] + 1)
            out[idx][y:y+p.shape[0], x:x+p.shape[1]] = p
            if params['pattern_scale'] > 3:
                out[idx] = gaussian_filter(out[idx], 1.5)
            idx += 1
    return out.reshape(N, -1)

def generate_xor(params: Dict, image_shape: List[int]) -> np.ndarray:
    N = params['sample_size']
    out = np.zeros((N, image_shape[0], image_shape[1]))
    pats = get_patterns(params)
    poses = params['positions']
    combos = [[1, 1], [-1, -1], [1, -1], [-1, 1]]

    for k, (i_start, signs) in enumerate(zip(range(0, N, N//4), combos)):
        p1 = pats[0] * signs[0]
        p2 = pats[1] * signs[1]
        for i in range(i_start, i_start + N//4):
            out[i][poses[0][0]:poses[0][0]+p1.shape[0], poses[0][1]:poses[0][1]+p1.shape[1]] = p1
            out[i][poses[1][0]:poses[1][0]+p2.shape[0], poses[1][1]:poses[1][1]+p2.shape[1]] = p2
            if params['pattern_scale'] > 3:
                out[i] = gaussian_filter(out[i], 1.5)
    return out.reshape(N, -1)

def generate_distractors(params: Dict, image_shape: List[int], N: int) -> np.ndarray:
    patterns = np.zeros((N, image_shape[0], image_shape[1]))
    pat = get_distractors(params)
    signs = [-1, 1]

    for j in range(N):
        # inds = np.random.choice(len(pat), 2)
        poses = params['distractor_positions']
        rand_signs = np.random.choice(signs, len(poses))
        for i, pos in enumerate(poses):
            patterns[j][pos[0]:pos[0]+pat.shape[0], pos[1]:pos[1]+pat.shape[1]] = pat * rand_signs[i]

        # pat1, pat2 = chosen[inds[0]], chosen[inds[1]]
        # pos1, pos2 = poses[0], poses[1]
        # poses = params['distractor_positions'][np.random.choice([0, 1])]
        # patterns[j][pos1[0]:pos1[0]+pat1.shape[0], pos1[1]:pos1[1]+pat1.shape[1]] = pat1 * rand_signs[0]
        # patterns[j][pos2[0]:pos2[0]+pat2.shape[0], pos2[1]:pos2[1]+pat2.shape[1]] = pat2 * rand_signs[1]

        if params['pattern_scale'] > 3:
            patterns[j] = gaussian_filter(patterns[j], 1.5)

    return patterns.reshape(N, -1)
