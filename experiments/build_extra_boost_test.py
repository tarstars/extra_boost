import sys
import unittest

import numpy as np

from experiments import forest
from experiments.extra_utils import subpool, time_features, test_extra_predictions
from experiments.forest import build_tree, EMatrix
from experiments.split import SplitMaker

import tensorflow.compat.v1 as tf


class TestBuildExtraBoost(unittest.TestCase):
    def test_build_extra_boost_00(self):
        pool = np.load("../prepare_dataset/pool_cross_00.npz")
        features_bool, f_time, label_orig = (
            pool[name] for name in ["features", "f_time", "label"]
        )
        label = label_orig

        np.random.seed(42)
        features_orig = np.random.normal(loc=features_bool * 1.0, scale=0.01)
        # features = features[:150000,:]; label = label_orig[:150000, :]

        f_test, l_test = features_orig[150000:, :], label_orig[150000:, :]
        features = features_orig[:150000, :]
        label = label_orig[:150000, :]

        extra_f = time_features(f_time, feat_num=2)

        features_orig_cut = features_orig  # [:,0:8]
        f_train, extra_f_train, label_train = subpool((features_orig_cut, extra_f, label_orig), f_time,
                                                      (0, 0.5))  # 0.33
        f_train, extra_f_train, label_train = f_train[:40000, :], extra_f_train[:40000, :], label_train[:40000, :]
        time_column = f_time
        full_features = features_orig_cut
        full_extra_features = extra_f
        full_label = label_orig

        boost_params = {
            "max_depth": 4,
            "learning_rate": 0.3,
            "splitgax": True,
            "transposed_feature": 0,
            "unbalanced_penalty": 0.001,
        }

        rounds = 10

        booster = forest.train(
            boost_params,
            EMatrix(
                features=f_train.T if boost_params["transposed_feature"] else f_train,
                extra_features=extra_f_train.T
                if boost_params["transposed_feature"]
                else extra_f_train,
                label=label_train,
            ),
            num_boost_round=rounds,
        )

        whole_prediction = booster.predict(full_features, extra_features=full_extra_features, tree_limit=3000)
        test_extra_predictions(50, time_column, full_label, whole_prediction)


if __name__ == "__main__":
    unittest.main()
