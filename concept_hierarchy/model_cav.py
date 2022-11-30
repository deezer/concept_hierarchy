"""
The concept learners
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

INPUT_SIZE = (1, 128) # (187, 96)


class CAVRegressor:
    """
    This is solely used to train a single regressor and learn a CAV.
    """

    def __init__(self, name, temporal_pooling=False):
        self.model = None
        self.lm_layers = None
        self.name = name
        self.build(temporal_pooling)

    @staticmethod
    def build_single_pipe(f, name=""):
        """f: embedder layer"""
        g = Flatten()(f)
        g = Dropout(0.25)(g)
        layer = Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.L1L2(),
            activation="sigmoid",
            name="CAV" + name,
        )
        y = layer(g)
        return y, layer

    def build(self, temporal_pooling):
        """embedder is all features"""
        x = Input(INPUT_SIZE)
        f = x  # used to be were the embedder was
        if not isinstance(f, list):
            f = [f]  # manage the case of single embedding layer

        lms = []
        output = []
        for i, emb in enumerate(f):
            t = emb
            if temporal_pooling and len(emb.shape) == 4:
                t = Lambda(
                    lambda z: tf.reduce_mean(z, axis=1),
                    name="temporal_pooling_" + str(i),
                )(t)
            y, layer = self.build_single_pipe(t, name=str(i))
            lms.append(layer)
            output.append(y)
        self.lm_layers = lms
        self.model = Model(inputs=x, outputs=output)
        self.model.compile(
            loss=["binary_crossentropy"] * len(output),
            optimizer=Adam(1e-3),
            metrics=["acc"],
        )

    def get_cav_weights(self):
        weights = {}
        for i, l in enumerate(self.lm_layers):
            weights[i] = l.get_weights()
        return weights


class CAVPredictor:
    """
    This is the aggregated predictor, with stacked weights, not the individual
    CAV model used for training.
    """

    def __init__(
        self, embedder, wpath, temporal_pooling=False, activation=True, name="CAV_Predictor"
    ):
        self.model = None
        self.lm_layers = None
        self.name = name
        self.weights, self.biases, self.labels = np.load(wpath, allow_pickle=True)
        self.cav_keys = list(self.biases.keys())
        self.n_cav = self.biases[self.cav_keys[0]].shape[1]  # output size
        self.build(embedder, activation, temporal_pooling)

    def build(self, embedder, activation, temporal_pooling):
        x = Input(INPUT_SIZE)
        f = embedder(x, training=False)  # -> list of tensors
        lms = []
        output = []
        layer_ac = "linear"
        if activation:
            layer_ac = "sigmoid"

        for i, k in enumerate(self.cav_keys):
            emb = f[i]
            t = emb
            if temporal_pooling and len(emb.shape) == 4:
                t = Lambda(
                    lambda z: tf.reduce_mean(z, axis=1),
                    name="temporal_pooling_" + str(i),
                )(t)
            t = Flatten(name="flatten_" + str(i))(t)
            layer = Dense(self.n_cav, activation=layer_ac, name="CAV_" + str(i))
            y = layer(t)
            layer.set_weights([self.weights[k], self.biases[k][0, :]])
            layer.trainable = False

            lms.append(layer)
            output.append(y)
        self.lm_layers = lms
        self.model = Model(inputs=x, outputs=output, name="CAV_Embedder")
        self.model.compile(metrics=["acc"])
