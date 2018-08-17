# import sys
# import os
# import json
import numpy as np
import tensorflow as tf
import unittest

# TODO: eliminate case when several features values are equal.
#       it is necessary to place threshold out of "claster of equity"

# TODO: manage case of constant features: one constant feature, whole constant feature array, etc.

def common_part(y, b, sorted_thresholds, features, label, bias):
    current_loss = tf.reduce_sum(-(label * tf.log_sigmoid(bias) + (1 - label) * tf.log_sigmoid(-bias)), axis=0)
    l_curr = -(y * tf.log_sigmoid(b) + (1 - y) * tf.log_sigmoid(-b))
    l_der1 = -y * tf.sigmoid(-b) + (1 - y) * tf.sigmoid(b)
    l_der2 = tf.sigmoid(-b) * tf.sigmoid(b)
    cum_l = tf.cumsum(l_curr)[:-1, :]  # -1
    cum_l_der1 = tf.cumsum(l_der1)[:-1, :]  # -1
    cum_l_der2 = tf.cumsum(l_der2)[:-1, :]  # -1
    rev_cum_l = tf.cumsum(l_curr, reverse=True)[1:, :]  # -1
    rev_cum_l_der1 = tf.cumsum(l_der1, reverse=True)[1:, :]  # -1
    rev_cum_l_der2 = tf.cumsum(l_der2, reverse=True)[1:, :]  # -1
    delta_up = -cum_l_der1 / (cum_l_der2 + 1.0) # -1
    delta_down = -rev_cum_l_der1 / (rev_cum_l_der2 + 1.0)  # -1
    loss_up = cum_l + 0.5 * delta_up * cum_l_der1
    loss_down = rev_cum_l + 0.5 * delta_down * rev_cum_l_der1
    whole_loss = (loss_up + loss_down)
    features_amount = tf.to_double(tf.shape(features)[0])
    #######
    # -----#
    # -----#   -->   iiiIiii (best_loss_argmin0), mmmMmmm (min_loss_axis0)
    # -----#
    #######
    best_loss_argmin0 = tf.argmin(whole_loss, axis=0)
    min_loss_axis0 = tf.reduce_min(whole_loss, axis=0)
    # mmmMmmm  ->    I (best_loss_argmin1), M (best_loss)
    best_loss_argmin1 = tf.argmin(min_loss_axis0, axis=0)
    best_loss = tf.reduce_min(min_loss_axis0, axis=0)
    best_avg_loss = best_loss / features_amount
    avg_current_loss = current_loss / features_amount
    # iiiIiii  ->  I
    best_index = best_loss_argmin0[best_loss_argmin1]
    thr = (sorted_thresholds[best_index, best_loss_argmin1] +
           sorted_thresholds[best_index + 1, best_loss_argmin1]) / 2
    best_delta_up = delta_up[best_index, best_loss_argmin1]
    best_delta_down = delta_down[best_index, best_loss_argmin1]
    return {'features': features,
            'bias': bias,
            'label': label,
            'l_curr': l_curr,
            'l_der1': l_der1,
            'l_der2': l_der2,
            'cum_l_der1': cum_l_der1,
            'cum_l_der2': cum_l_der2,
            'rev_cum_l_der1': rev_cum_l_der1,
            'rev_cum_l_der2': rev_cum_l_der2,
            'delta_up': delta_up,
            'delta_down': delta_down,
            'best_delta_up': best_delta_up,
            'best_delta_down': best_delta_down,
            'cum_l': cum_l,
            'rev_cum_l': rev_cum_l,
            'whole_loss': whole_loss,
            'best_loss_argmin0': best_loss_argmin0,
            'min_loss_axis0': min_loss_axis0,
            'best_loss_argmin1': best_loss_argmin1,
            'best_feature_index': best_loss_argmin1,
            'best_loss': best_loss,
            'best_avg_loss': best_avg_loss,
            'best_index': best_index,
            'T': sorted_thresholds,
            'thr': thr,
            'y': y,
            'current_loss': current_loss,
            'avg_current_loss': avg_current_loss
            }



def create_split_interface():
    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(dtype=tf.float64)
        label = tf.placeholder(dtype=tf.float64)
        bias = tf.placeholder(dtype=tf.float64)
        ax = tf.transpose(tf.nn.top_k(-tf.transpose(features), k=tf.shape(features)[-2]).indices)
        y = tf.gather(label, ax)[:, :, 0]
        b = tf.gather(bias, ax)[:, :, 0]
        # sorted_thresholds = tf.gather(features, ax, axis=0)
        sorted_thresholds = tf.contrib.framework.sort(features, axis=0)
        common_tensors = common_part(y, b, sorted_thresholds, features, label, bias)
        
    result = {'graph': graph,
              'gax': ax,
             }
    result.update(common_tensors)
    return result

# along axis 0 only
def take_along_axis(arr, indices):
    # arr (n x m)
    m = tf.shape(arr)[1]
    r = tf.range(m)
    taken_arr = tf.gather(tf.reshape(arr, (-1,)), indices * m + r, axis=0)
    return taken_arr

def create_split_quick():
    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(dtype=tf.float64)
        label = tf.placeholder(dtype=tf.float64)
        bias = tf.placeholder(dtype=tf.float64)
        ax = tf.placeholder(dtype=tf.int32)
        gax = tf.transpose(tf.nn.top_k(-tf.transpose(features), k=tf.shape(features)[-2]).indices)
        y = tf.gather(label, ax)[:, :, 0]
        b = tf.gather(bias, ax)[:, :, 0]
        # F (N x M),  ax (N, M): ax_{ij} - pos in F_{*j}, 
        # ST_{i,j} = F_{ax_{i,j}, j}
        # sorted_thresholds = tf.gather(features, ax, axis=0)
        sorted_thresholds = take_along_axis(features, ax)
        common_tensors = common_part(y, b, sorted_thresholds, features, label, bias)
        
    result = {'graph': graph,
              'gax': gax,
              'ax': ax,
             }
    result.update(common_tensors)
    return result


split_interface = create_split_interface()
split_graph = split_interface['graph']

split_quick = create_split_quick()
split_quick_graph = split_quick['graph']

def make_split(bias, features, label):
    graph = split_interface['graph']
    input_tensors = {split_interface[t]: val for t, val in [('features', features), ('bias', bias), ('label', label)]}
    tensors = {t: split_interface[t]
               for t in ['thr', 'best_loss', 'best_index', 'best_delta_up', 'best_delta_down', 'current_loss',
                         'best_feature_index',
                         'avg_current_loss', 'best_avg_loss']}
    with tf.Session(graph=graph) as s:
        tensors_values = s.run(tensors, input_tensors)
    return tensors_values


def make_split_quick(bias, features, label, ax):
    input_tensors = {split_quick[t]: val for t, val in [('features', features), ('bias', bias), ('label', label), ('ax', ax)]}
    tensors = {t: split_quick[t]
               for t in ['thr', 'best_loss', 'best_index', 'best_delta_up', 'best_delta_down', 'current_loss',
                         'best_feature_index',
                         'avg_current_loss', 'best_avg_loss']}
    with tf.Session(graph=split_quick_graph) as s:
        tensors_values = s.run(tensors, input_tensors)
    return tensors_values

def make_gax(features):
    with tf.Session(graph=split_quick_graph) as s:
        gax_val = s.run(split_quick['gax'], {split_quick['features']: features})
    return gax_val
        
class TestSplit(unittest.TestCase):
    def test_small_split_00(self):
        features = np.array([[1, 2, 1, 1, 2, 2],
                             [1, 3, 2, 4, 5, 6],
                             [1, 2, 3, 4, 5, 6]
                             ], dtype=np.float32).T
        label = np.array([0, 0, 1, 1, 1, 1], dtype=np.float32).reshape((-1, 1))
        bias = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32).reshape((-1, 1))

        tensor_values = make_split(bias, features, label)
        self.assertEqual(tensor_values['best_feature_index'], 2)
        self.assertAlmostEqual(tensor_values['thr'], 2.5)
        self.assertAlmostEqual(tensor_values['avg_current_loss'], np.log(2))
        best_avg_loss = tensor_values['best_avg_loss']
        # best_delta_up = tensor_values['best_delta_up']
        # best_delta_down = tensor_values['best_delta_down']
        self.assertLess(best_avg_loss, np.log(2))
        # print('best_avg_loss =', best_avg_loss,
        #       '\nbest_delta_up =', best_delta_up,
        #       '\nbest_delta_down =', best_delta_down)

    def test_small_split_01(self):
        perm = np.array([5, 2, 1, 0, 3, 4])
        features = np.array([[1, 2, 1, 1, 2, 2],
                             [1, 3, 2, 4, 5, 6],
                             [1, 2, 3, 4, 5, 6]
                             ], dtype=np.float32).T[perm, :]
        label = np.array([0, 0, 1, 1, 1, 1], dtype=np.float32).reshape((-1, 1))[perm, :]
        bias = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32).reshape((-1, 1))[perm, :]

        tensor_values = make_split(bias, features, label)
        self.assertEqual(tensor_values['best_feature_index'], 2)
        self.assertAlmostEqual(tensor_values['thr'], 2.5)
        self.assertAlmostEqual(tensor_values['avg_current_loss'], np.log(2))

if __name__ == '__main__':
    unittest.main()
