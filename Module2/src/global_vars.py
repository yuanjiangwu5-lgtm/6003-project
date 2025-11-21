# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import multiprocessing as mp
import numpy as np
import random

# To ensure reproducible results to help debugging, set seeds for randomness.
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
np.random.seed(seed)
random.seed(seed)

# sizes is the number of relus in each layer
sizes = list(map(int,sys.argv[1].split("-")))
dimensions = [tuple([x]) for x in sizes]
neuron_count = sizes

DIM = sizes[0]

# __cheat_A, __cheat_B = np.load("models/" + str(seed) + "_" + "-".join(map(str,sizes))+".npy", allow_pickle=True)
# Robust loader for ground-truth model (tries .npz -> .npy -> .p)
import os
import numpy as _np

model_basename = "models/" + str(seed) + "_" + "-".join(map(str, sizes))
__cheat_A = None
__cheat_B = None

def _try_load_npz(path):
    try:
        data = _np.load(path, allow_pickle=True)
        # if saved with savez, keys likely like A_0, A_1, ..., B_0, B_1, ...
        files = list(data.files)
        if not files:
            return None, None
        # collect A_* and B_* in order
        A_list = []
        B_list = []
        # sort keys so A_0,A_1... then B_0,B_1...
        def sort_key(k):
            parts = k.split('_')
            if len(parts) >= 2 and parts[0] in ('A','B'):
                return (parts[0], int(parts[1]))
            return (k, 0)
        for k in sorted(files, key=sort_key):
            if k.startswith("A_"):
                A_list.append(_np.array(data[k]))
            elif k.startswith("B_"):
                B_list.append(_np.array(data[k]))
            else:
                # ignore unexpected keys
                pass
        if len(A_list) > 0 and len(B_list) > 0:
            return A_list, B_list
        # if not matching A_/B_ pattern, fall through
        return None, None
    except Exception as e:
        print("Warning: failed to load .npz:", path, "error:", e)
        return None, None

def _try_load_npy(path):
    try:
        loaded = _np.load(path, allow_pickle=True)
        # several possible formats; try to interpret common ones
        if isinstance(loaded, _np.ndarray) and loaded.dtype == object:
            lst = loaded.tolist()
            if isinstance(lst, (list, tuple)) and len(lst) >= 2:
                return lst[0], lst[1]
        # if it's array-like with two elements
        try:
            a, b = loaded
            return a, b
        except Exception:
            return None, None
    except Exception as e:
        print("Warning: failed to load .npy:", path, "error:", e)
        return None, None

def _try_load_pickle(path):
    try:
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            return obj[0], obj[1]
        return None, None
    except Exception as e:
        print("Warning: failed to load pickle:", path, "error:", e)
        return None, None

npz_path = model_basename + ".npz"
npy_path = model_basename + ".npy"
p_path = model_basename + ".p"

# Try .npz first (most likely)
if os.path.exists(npz_path):
    __cheat_A, __cheat_B = _try_load_npz(npz_path)
    if __cheat_A is not None:
        print("Loaded ground-truth model from", npz_path)

# Then try .npy
if __cheat_A is None and os.path.exists(npy_path):
    __cheat_A, __cheat_B = _try_load_npy(npy_path)
    if __cheat_A is not None:
        print("Loaded ground-truth model from", npy_path)

# Then try .p (pickle)
if __cheat_A is None and os.path.exists(p_path):
    __cheat_A, __cheat_B = _try_load_pickle(p_path)
    if __cheat_A is not None:
        print("Loaded ground-truth model from", p_path)

if __cheat_A is None:
    print("No ground-truth model found at", npz_path, "or", npy_path, "or", p_path)
    print("If you want to use ground-truth (cheating) values for debugging, run train_models.py first to create them, for example:")
    print("  python3 train_models.py", "-".join(map(str, sizes)), seed)

# In order to help debugging, we're going to log what lines of code
# cause lots of queries to be generated. Use this to improve things.
query_count = 0
query_count_at = {}

# HYPERPARAMETERS. Change these at your own risk. It may all die.

PARAM_SEARCH_AT_LOCATION = 1e2
GRAD_EPS = 1e-4
SKIP_LINEAR_TOL = 1e-8
BLOCK_ERROR_TOL = 1e-3
BLOCK_MULTIPLY_FACTOR = 2
DEAD_NEURON_THRESHOLD = 1000
MIN_SAME_SIZE = 4 # this is most likely what should be changed

if len(sizes) == 3:
    PARAM_SEARCH_AT_LOCATION = 1e4
    GRAD_EPS = 1e1
    SKIP_LINEAR_TOL = 1e-7
    BLOCK_MULTIPLY_FACTOR = 8

# When we save the results, we're going to use this to make sure that
# (a) we don't trash over old results, but
# (b) we don't keep stale results around
name_hash = "-".join(map(str,sizes))+str(hash(tuple(np.random.get_state()[1])))

# CHEAT MODE. Turning on lets you read the actual weight matrix.

# Enable IDDQD mode
# In order to debug sometimes it helps to be able to look at the actual values of the
# true weight matrix.
# When we're allowed to do that, assign them from __cheat_A and __cheat_B
# When we're not, then just give them a constant 0 so the code doesn't crash
CHEATING = False
    
if CHEATING:
    A = [np.array(x) for x in __cheat_A]
    B = [np.array(x) for x in __cheat_B]
else:
    A = [np.zeros_like(x) for x in __cheat_A]
    B = [np.zeros_like(x) for x in __cheat_B]

MPROC_THREADS = max(mp.cpu_count(),1)
pool = []

# ============================================================
# Default oracle placeholder (Module 1 / Module 2 will overwrite)
# ============================================================

def f(x):
    raise RuntimeError("global_vars.f was called before being assigned to a model.")
