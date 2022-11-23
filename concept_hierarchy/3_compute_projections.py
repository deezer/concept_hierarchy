"""
Compute projections of learnt CAVs for similarity estimations.

NOTE:
We do not upload the resulting file for the Deezer dataset, it's 1.5Go, we will
after the reviews. We only provide the ones for APM.
"""

import argparse
import os
from collections import defaultdict
from glob import glob

import numpy as np
import tensorflow as tf
from data_loader import ImbalancedDataLoader
from model_cav import CAVPredictor
from model_musicnn import vgg_keras

DATA_PATH = "data/deezer_mels_tensors"
PLAYLIST_PATH = "data/deezer_playlist.npy"
EMBEDDER_WEIGHT_PATH = "../weights/MSD_vgg.h5"
CAV_WEIGHT_PATH = "weights/deezer_cav.npy"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="visible gpu", type=int, default=3)
    args = parser.parse_args()

    # Config GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load embedder

    vgg, bottleneck = vgg_keras(128, return_feature_model=True)
    vgg.load_weights(EMBEDDER_WEIGHT_PATH)
    vgg.trainable = False
    bottleneck.trainable = False

    # -----------------------------------------------------------------------------------------------------------------

    predictor = CAVPredictor(
        bottleneck, CAV_WEIGHT_PATH, temporal_pooling=True, activation=False   # outputs without applying sigmoid
    )

    # Open dataset

    print("opening playlists")
    playlists = np.load(PLAYLIST_PATH, allow_pickle=True).item()

    print("loading audio paths")
    path_set = glob(os.path.join(DATA_PATH, "**/*.tensor"))
    print("\nFound {} paths\n".format(len(path_set)))

    data_loader = ImbalancedDataLoader({"song_set": []})

    song_2_path = {}
    for path in path_set:
        song_id = int(path.split("/")[-1].split(".")[0])
        song_2_path[song_id] = path

    # Mean, Var estimation loop

    stats = defaultdict(
        lambda: {
            "ex": np.zeros(len(predictor.labels)),  # expectation: \mathbb{E}[X]
            "ex2": np.zeros(len(predictor.labels)),  # \mathbb{E}[X^2]
            "esigx": np.zeros(len(predictor.labels)),  # \mathbb{E}[\sigma(X)]
            "esigx2": np.zeros(len(predictor.labels)),  # \mathbb{E}[\sigma(X)^2]
            "n": 0,
        }
    )

    for tag in predictor.labels:
        pos_path_list = []
        for song in playlists[tag]["tracks_array"]:
            if song in song_2_path:
                pos_path_list.append(song_2_path[song])

        songs_split = data_loader.create_training_splits(pos_path_list)
        test_data_it = data_loader.create_reader_from_set_tf_iterator(
            songs_split["test"]["pos"]
        )
        print("Found", len(test_data_it), "pos test songs for tag", tag)

        for X in test_data_it:
            V = predictor.model.predict(X)  # outputs 0, 1, 2, 3
            sigV = tf.nn.sigmoid(V).numpy()
            for pool_layer in range(4):
                subtag = "{}_{}".format(tag, pool_layer)
                stats[subtag]["ex"] = stats[subtag]["ex"] + np.mean(V[pool_layer], 0)
                stats[subtag]["ex2"] = stats[subtag]["ex2"] + np.mean(
                    np.square(V[pool_layer]), 0
                )
                stats[subtag]["esigx"] = stats[subtag]["esigx"] + np.mean(
                    sigV[pool_layer], 0
                )
                stats[subtag]["esigx2"] = stats[subtag]["esigx2"] + np.mean(
                    np.square(sigV[pool_layer]), 0
                )
                stats[subtag]["n"] += 1

    np.save("results/stats_deezer", dict(stats))
