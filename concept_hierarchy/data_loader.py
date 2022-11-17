import numpy as np
import tensorflow as tf
from model_cav import INPUT_SIZE


class ImbalancedDataLoader:
    """Sample for given playlist versus rest."""

    def __init__(self, params):
        """
        pos_examples_list: list of audio with the positive label we are interested
        in learning, the rest of negatives is sampled uniformly.
        """
        self.params = {
            "batch_size": 64,
            "repeat": 2,
            "seed": 123,
        }
        self.params.update(params)
        self.params["batch_shape"] = (self.params["batch_size"],) + INPUT_SIZE

    @staticmethod
    @tf.function
    def pad_small(x, bonus_pad=20):
        lacking_pad = INPUT_SIZE[0] - tf.shape(x)[-2]
        lacking_pad = tf.maximum(0, lacking_pad)
        return tf.pad(
            x,
            ((bonus_pad, lacking_pad + bonus_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    @staticmethod
    @tf.function
    def random_audio_slice(x):
        """Takes a random slice of the spectrogram for data augmentation"""
        max_offset = tf.shape(x)[-2] - INPUT_SIZE[0]
        offset = tf.random.uniform((), maxval=max_offset + 1, dtype=tf.int32)
        return tf.slice(x, (offset, 0), INPUT_SIZE)

    @staticmethod
    @tf.function
    def slice_along_audio(x):
        """Takes a random slice of the spectrogram for data augmentation"""
        cut_index = tf.range(0, tf.shape(x)[0] - INPUT_SIZE[0], INPUT_SIZE[0] // 2)
        split = tf.map_fn(
            lambda i: x[i : i + INPUT_SIZE[0]],
            cut_index,
            fn_output_signature=tf.float32,
        )
        return tf.stack(split)

    @staticmethod
    @tf.function
    def multi_hot(x, depth):
        return tf.reduce_max(tf.one_hot(x, depth), axis=0)

    @tf.function
    def open_audio(self, f_name):
        audio = tf.io.read_file(f_name)
        audio = tf.io.parse_tensor(audio, tf.float32)
        audio = self.pad_small(audio)  # let's include that in the .map
        return audio

    def create_training_splits(self, pos_list, val_split=0.1, test_split=0.2):
        np.random.seed(self.params["seed"])  # same shuffle for train / test
        loc_pos_list = np.copy(pos_list)
        neg_list = list(set(self.params["song_set"]) - set(pos_list))

        np.random.shuffle(loc_pos_list)
        np.random.shuffle(neg_list)
        pos_val_split_index = round(len(loc_pos_list) * (1 - val_split - test_split))
        pos_test_split_index = round(len(loc_pos_list) * (1 - test_split))
        neg_val_split_index = round(len(neg_list) * (1 - val_split - test_split))
        neg_test_split_index = round(len(neg_list) * (1 - test_split))

        lists = {
            "train": {
                "pos": loc_pos_list[:pos_val_split_index],
                "neg": neg_list[:neg_val_split_index],
            },
            "val": {
                "pos": loc_pos_list[pos_val_split_index:pos_test_split_index],
                "neg": neg_list[neg_val_split_index:neg_test_split_index],
            },
            "test": {
                "pos": loc_pos_list[pos_test_split_index:],
                "neg": neg_list[neg_test_split_index:],
            },
        }
        return lists

    def create_tf_iterator(self, sample_lists):
        """
        use list of positives, convert to iterator
        use list of negatives, convert to iterator
        batch, concat, return
        link to labels

        train -> shuffle and stuff
        other -> read through it
        """

        pos_ds = tf.data.Dataset.from_tensor_slices(sample_lists["pos"])
        pos_ds = pos_ds.map(self.open_audio)
        pos_ds = pos_ds.repeat()

        neg_ds = tf.data.Dataset.from_tensor_slices(sample_lists["neg"])
        neg_ds = neg_ds.map(self.open_audio)
        neg_ds = neg_ds.repeat()

        pos_ds = self.open_batch(pos_ds)
        neg_ds = self.open_batch(neg_ds)

        ds = tf.data.Dataset.zip((pos_ds, neg_ds))
        ds = ds.map(lambda x, y: tf.concat((x, y), axis=0))
        return ds

    def open_batch(self, ds_):
        # reuse the same song several times -> kinda a cache of recent items
        ds_ = ds_.flat_map(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(self.params["repeat"])
        )

        ds_ = ds_.map(self.random_audio_slice)
        ds_ = ds_.shuffle(self.params["repeat"] * self.params["batch_size"] * 5)
        ds_ = ds_.batch(self.params["batch_size"] // 2)
        return ds_

    def create_reader_tf_iterator(self, mode="train", val_split=0.1, test_split=0.2):
        """No label, just sample music audio."""
        np.random.seed(self.params["seed"])
        song_list = list(self.params["song_set"])
        np.random.shuffle(song_list)

        val_split_index = round(len(song_list) * (1 - val_split - test_split))
        test_split_index = round(len(song_list) * (1 - test_split))
        if mode == "train":
            target_song = song_list[:val_split_index]
        elif mode == "val":
            target_song = song_list[val_split_index:test_split_index]
        elif mode == "test":
            target_song = song_list[test_split_index:]
        else:
            raise ValueError("Something is broken with the arg `mode` :", mode)

        ds = tf.data.Dataset.from_tensor_slices(target_song)
        ds2 = ds.map(self.open_audio)

        return tf.data.Dataset.zip((ds, ds2))

    def create_reader_slicer_tf_iterator(self, mode):
        """return slices of same size instead of full music"""
        ds = self.create_reader_tf_iterator(mode)
        ds = ds.map(lambda x, y: (x, self.slice_along_audio(y)))
        return ds

    def create_reader_from_set_tf_iterator(self, song_set):
        """given predefined split song set, do stuff"""
        ds = tf.data.Dataset.from_tensor_slices(song_set)
        ds = ds.map(self.open_audio)
        ds = ds.map(self.slice_along_audio)
        return ds
