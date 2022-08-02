"""
Training code of the CAV regressors

NOTE:
The `deezer_mels_tensors` dataset will be provided after review,
probably with a workaround due to copyright (and size) issues: e.g., only
uploading spectrograms embedded in the backbone.
"""

import time
import os
import sys
import gc
from glob import glob
import numpy as np
import tensorflow as tf

from model_musicnn import vgg_keras
from model_cav import CAV_Regressor
from data_loader import ImbalancedDataLoader


DATA_PATH = "data/deezer_mels_tensors"      # unavailable, 30s music dataset
PLAYLIST_PATH = "data/deezer_playlists.npy"
EMBEDDER_WEIGHT_PATH = 'weights/MSD_vgg.h5'
SAVE_DIR = "CAV"
BS = 128  # training batch size


# args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="start index for the script", type=int, default=0)
parser.add_argument("--stop", help="stop index for the script", type=int, default=-1)
parser.add_argument("--step", help="step for the script", type=int, default=1)
parser.add_argument("--gpu", help="visible gpu", type=int, default=3)
args = parser.parse_args()

# Config GPUs
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


## Load embedder

vgg, bottleneck = vgg_keras(128, return_feature_model = True)
vgg.load_weights(EMBEDDER_WEIGHT_PATH)
vgg.trainable = False
bottleneck.trainable = False


## Open dataset

print('opening tags')
tag_2_songs = np.load(PLAYLIST_PATH, allow_pickle = True).item()
playlist_ids = list(tag_2_songs.keys())

print('loading audio paths')
path_set = glob(os.path.join(DATA_PATH, '**/*.tensor'))
print("\nFound {} paths\n".format(len(path_set)))


data_loader = ImbalancedDataLoader({
                'batch_size': BS,
                'song_set': path_set,
                'repeat': 4,
                })

song_2_path = {}
for path in path_set:
    song_id = int(path.split('/')[-1].split('.')[0])
    song_2_path[song_id] = path

constant_gt = tf.concat(( tf.ones(BS//2, tf.float32), tf.zeros(BS//2, tf.float32) ),
                        axis = -1)
constant_gt = tf.expand_dims(constant_gt, -1)


## Saver

class SubSaver(tf.keras.callbacks.Callback):
    def __init__(self, n_cav, dirpath, monitor='val_loss', target_tag='', verbose=0):
        super().__init__()

        self.n_cav = n_cav
        self.dirpath = dirpath
        self.monitor = monitor
        self.verbose = verbose
        self.target_tag = target_tag
        if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch + 1, self.monitor,
                                            self.best, current, self.dirpath))
            self.best = current

            weights = {}
            for i in range(self.n_cav):
                w = self.model.get_layer('CAV'+str(i)).get_weights()
                weights[i] = w
            np.save(os.path.join(self.dirpath, self.target_tag), weights)


## Train loop

existing_saves = glob(os.path.join(SAVE_DIR, '*_perf.npy'))
start_time = time.time()


# -- Loop CAV
stop_index = len(playlist_ids)
if args.stop != -1:
    stop_index = args.stop
target_list = list(playlist_ids)[args.start:stop_index:args.step]


for k, tag in enumerate(target_list):
    tag_formatted = str(tag)
    if os.path.join(SAVE_DIR, 'cav_' + str(tag_formatted) + '_perf.npy') in existing_saves:
        continue
    print("\n\n\n -- Running script for pid ", tag)

    pos_path_list = []
    for song in tag_2_songs[tag]['tracks_array']:
        if song in song_2_path:
            pos_path_list.append(song_2_path[song])

    print('Found', len(pos_path_list), 'pos songs', '(total:', len(path_set), ')')

    tf.keras.backend.clear_session()
    gc.collect()

    songs_split = data_loader.create_training_splits(pos_path_list)
    data_it = data_loader.create_tf_iterator(songs_split['train'], 'train')
    data_it = data_it.map(lambda x: (x, constant_gt) )
    val_data_it = data_loader.create_tf_iterator(songs_split['val'], 'val')
    val_data_it = val_data_it.map(lambda x: (x, constant_gt) )

    print("Training...")
    lm = CAV_Regressor('cav_' + str(tag_formatted), bottleneck, temporal_pooling = True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)
    reducer = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss',
                    patience = 5,
                    factor=0.1, mode='auto', min_lr=1e-6)
    saver = SubSaver(len(bottleneck.output), SAVE_DIR, monitor = 'val_loss',
                    target_tag = tag_formatted, verbose = 1)
    lm.model.fit(data_it, epochs = 100, steps_per_epoch = 100, verbose = 2,
                    validation_data = val_data_it, validation_steps = 10,
                    callbacks = [ early_stop, reducer, saver ] )

    print('Now evaluating...')
    del data_it
    del val_data_it
    test_data_it = data_loader.create_tf_iterator(songs_split['test'], 'test')
    test_data_it = test_data_it.map(lambda x: (x, constant_gt) )

    perfs = lm.model.evaluate(test_data_it, steps = 200, verbose = 0)
    np.save(os.path.join(SAVE_DIR, lm.name + '_perf'), perfs)

    end_time = time.time()
    print('----> took {:.2f}m {}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    start_time = end_time
