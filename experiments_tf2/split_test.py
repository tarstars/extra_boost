import unittest

import numpy as np

from experiments.forest import split
from experiments.split import make_gax


class TestSplit(unittest.TestCase):
    def test_small_split_00(self):
        features = np.array(
            [[1, 2, 1, 1, 2, 2], [1, 3, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
            dtype=np.float32,
        ).T
        extra_features = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32,).T
        ax = make_gax(features)
        label = np.array([0, 0, 1, 1, 1, 1], dtype=np.float32).reshape((-1, 1))
        bias = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32).reshape((-1, 1))

        params = {}
        reduce_axis = 0

        tensor_values = split(
            bias=bias,
            features=features,
            label=label,
            extra_features=extra_features,
            ax=ax,
            params=params,
            reduce_axis=reduce_axis,
            unbalanced_penalty=0,
            use_extra=False,
        )
        self.assertEqual(tensor_values["best_feature_index"], 2)
        self.assertAlmostEqual(tensor_values["thr"], 2.5)
        self.assertAlmostEqual(tensor_values["avg_current_loss"], np.log(2))
        best_avg_loss = tensor_values["best_avg_loss"]
        # best_delta_up = tensor_values['best_delta_up']
        # best_delta_down = tensor_values['best_delta_down']
        self.assertLess(best_avg_loss, np.log(2))
        # print('best_avg_loss =', best_avg_loss,
        #       '\nbest_delta_up =', best_delta_up,
        #       '\nbest_delta_down =', best_delta_down)

    def test_small_split_01(self):
        perm = np.array([5, 2, 1, 0, 3, 4])
        features = np.array(
            [[1, 2, 1, 1, 2, 2], [1, 3, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
            dtype=np.float32,
        ).T[perm, :]
        label = np.array([0, 0, 1, 1, 1, 1], dtype=np.float32).reshape((-1, 1))[perm, :]
        bias = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32).reshape((-1, 1))[perm, :]
        ax = make_gax(features=features)
        params = {}
        reduce_axis = 0

        tensor_values = split(bias=bias, features=features, label=label,
                              extra_features=None, ax=ax, params=params, reduce_axis=reduce_axis,
                              unbalanced_penalty=0, use_extra=False)
        self.assertEqual(tensor_values["best_feature_index"], 2)
        self.assertAlmostEqual(tensor_values["thr"], 2.5)
        self.assertAlmostEqual(tensor_values["avg_current_loss"], np.log(2))


if __name__ == "__main__":
    unittest.main()
