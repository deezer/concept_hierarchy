'''
Open trained CAV files, sort, combine and create final weight file.

NOTE:
We could not upload the trained weight as it is >50MB.
'''

import time
from collections import defaultdict
import os
import sys
from glob import glob
import numpy as np

perf_files = glob('weights/DEEZER_CAV/*_perf.npy')  # unavailable
cav_files = glob('weights/DEEZER_CAV/*[0-9].npy')   # unavailable

playlists_path = 'data/deezer_playlists.npy'
playlists = np.load(playlists_path, allow_pickle=True).item()
THRESHOLD_PERF = 0.7  # 70% accuracy
THRESHOLD_NB_TRACKS = 39
SELECTED_LAYER = 2  # in [0 .. 3] for pool3, pool4, pool5, output

## filter

filtered_cav_1 = set()
for p_id in playlists:
    if ( (len(playlists[p_id]['tracks_array']) >= THRESHOLD_NB_TRACKS)
            and (len(playlists[p_id]['tracks_array']) < 251 )
             ):
        filtered_cav_1.add(p_id)

filtered_cav_2 = set()
left_out_cav = set()
cav_perfs = {}
for perf_file in perf_files:
    p_id = int(perf_file.split('/')[-1].split('_')[1])
    if p_id not in filtered_cav_1:
        continue
    perf = np.load(perf_file)
    cav_perfs[p_id] = perf[-4+SELECTED_LAYER]
    if cav_perfs[p_id] > THRESHOLD_PERF:
        filtered_cav_2.add(p_id)
    else:
        left_out_cav.add(p_id)

print(f'Final selection of {len(filtered_cav_2)} cavs')

## assemble

from tqdm import tqdm

weights = defaultdict(list)
biases = defaultdict(list)
first_iter = True
names = []

for fpath in tqdm(cav_files):
    name = int(fpath.split('/')[-1].split('.')[0])
    if name not in filtered_cav_2:
        continue
    names.append(name)
    f = np.load(fpath, allow_pickle=True).item()
    for cav in f:
        w, b = f[cav]
        weights[cav].append(w)
        biases[cav].append(b)

weights_np = {}
biases_np = {}
for cav in weights:
    weights_np[cav] = np.concatenate(weights[cav], axis = -1)
    biases_np[cav] = np.stack(biases[cav], axis = 1)


np.save('weights/learned_deezer_CAV', (weights_np, biases_np, names))