import os
import sys
import unittest

import numpy as np
import tensorflow as tf

from experiments_tf2.extra_utils import subpool, test_extra_predictions, time_features
from experiments_tf2.forest import EBooster, EMatrix

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disables GPU run


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
        f_train, extra_f_train, label_train = subpool(
            (features_orig_cut, extra_f, label_orig), f_time, (0, 0.5)
        )  # 0.33
        sz = 40000
        f_train, extra_f_train, label_train = (
            f_train[:sz, :],
            extra_f_train[:sz, :],
            label_train[:sz, :],
        )
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

        booster = EBooster.train(
            boost_params,
            EMatrix.from_features_params(
                features=f_train,
                extra_features=extra_f_train,
                params=boost_params,
                label=label_train,
            ),
            num_boost_round=rounds,
        )

        whole_prediction = booster.predict(
            full_features, extra_features=full_extra_features, tree_limit=3000
        )
        test_extra_predictions(50, time_column, full_label, whole_prediction)


class TestSplitOperation(unittest.TestCase):
    def test_split_operation(self):
        a = tf.Variable([[1, 1], [1, 2], [2, 3], [3, 4], [3, 5], [3, 6]])
        b = tf.experimental.numpy.split(ary=a, indices_or_sections=[0])
        print(b)


if __name__ == "__main__":
    unittest.main()
