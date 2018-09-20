import sys
import unittest
# import os
# import json

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from cumsum_zoo import mycumsum1
from tf_debug import tf_print
from tf_utils import tfravel, take_along_axis


# TODO: eliminate case when several features values are equal.
#       it is necessary to place threshold out of "claster of equity"

# TODO: manage case of constant features: one constant feature, whole constant feature array, etc.

def cumsum_two_derivatives(l_der1, l_der2, reduce_axis, make_transpose, use_my_cumsum):
    alt_axis = 1 - reduce_axis

    # Choice of 3 cases
    #cum_l_der1_full = tf.cumsum(l_der1, axis=reduce_axis)  # -1
    #cum_l_der2_full = tf.cumsum(l_der2, axis=reduce_axis)  # -1
    der_joined = tf.concat([l_der1, l_der2], axis=alt_axis, name='der_joined')
    
    # Transpose if needed
    der_joined_input = tf.transpose(der_joined) if make_transpose else tf.identity(der_joined)
    # Axis to calc cumsum
    cum_axis = alt_axis if make_transpose else reduce_axis
    # Procedure to calc cumsum
    cum_proc = mycumsum1 if use_my_cumsum else tf.cumsum
    # Calc cumsum
    cumsum_der_joined_output = cum_proc(der_joined_input, axis=cum_axis, name='cumsum_der_joined')
    # Reverse transpose 
    cumsum_der_joined = tf.transpose(cumsum_der_joined_output) if make_transpose else tf.identity(cumsum_der_joined_output)
    
    #+++++++++++++++++
    if reduce_axis==0:
        cum_l_der1_full = cumsum_der_joined[:, :tf.shape(l_der1)[1]]
        cum_l_der2_full = cumsum_der_joined[:, tf.shape(l_der1)[1]:]
    else:
        cum_l_der1_full = cumsum_der_joined[:tf.shape(l_der1)[0], :]
        cum_l_der2_full = cumsum_der_joined[tf.shape(l_der1)[0]:, :]
    
    if reduce_axis==0:
        cum_l_der1, tot_der1 = cum_l_der1_full[:-1, :], cum_l_der1_full[-1, :]  # -1
        cum_l_der2, tot_der2 = cum_l_der2_full[:-1, :], cum_l_der2_full[-1, :]  # -1
    else:
        cum_l_der1, tot_der1 = cum_l_der1_full[:, :-1], cum_l_der1_full[:, -1:]  # -1
        cum_l_der2, tot_der2 = cum_l_der2_full[:, :-1], cum_l_der2_full[:, -1:]  # -1     

    #rev_cum_l_der1 = tf.cumsum(l_der1, reverse=True)[1:, :]  # -1
    #rev_cum_l_der2 = tf.cumsum(l_der2, reverse=True)[1:, :]  # -1
    rev_cum_l_der1 = tot_der1 - cum_l_der1
    rev_cum_l_der2 = tot_der2 - cum_l_der2
    return cum_l_der1, cum_l_der2, rev_cum_l_der1, rev_cum_l_der2


def default_get_loss(l_der1, l_der2, reduce_axis, make_transpose, use_my_cumsum):
    cum_l_der1, cum_l_der2, rev_cum_l_der1, rev_cum_l_der2 = cumsum_two_derivatives(l_der1, l_der2, reduce_axis=reduce_axis, make_transpose=make_transpose, use_my_cumsum=use_my_cumsum)
    
    delta_up = -cum_l_der1 / (cum_l_der2 + 1.0) # -1
    delta_down = -rev_cum_l_der1 / (rev_cum_l_der2 + 1.0)  # -1
    
    #loss_up = cum_l + 0.5 * delta_up * cum_l_der1
    #loss_down = rev_cum_l + 0.5 * delta_down * rev_cum_l_der1
    loss_up = 0.5 * delta_up * cum_l_der1
    loss_down = 0.5 * delta_down * rev_cum_l_der1
    loss_sum = loss_up + loss_down
    
    return tf.expand_dims(delta_up, axis=2), tf.expand_dims(delta_down, axis=2), loss_sum


def extra_get_loss(l_der1, l_der2, extra, reduce_axis, make_transpose, use_my_cumsum):
    extra_shape = tf.shape(extra)
    extra_1 = tf.reshape(extra, (extra_shape[0], extra_shape[1], 1, extra_shape[2]))
    extra_2 = tf.reshape(extra, (extra_shape[0], extra_shape[1], extra_shape[2], 1))
    l_der_shape = tf.shape(l_der2)
    l_der2_multy = tf.reshape(l_der2, (l_der_shape[0], l_der_shape[1], 1, 1))
    l_der1_multy = tf.reshape(l_der1, (l_der_shape[0], l_der_shape[1], 1, 1))
    h = l_der2_multy * extra_1 * extra_2
    lder_extra = l_der1_multy * extra_2
    
    h_cumsum = tf.cumsum(h, axis=reduce_axis)
    g_cumsum = tf.cumsum(lder_extra, axis=reduce_axis)
    
    if reduce_axis==0:
        g_upper, tot_g = g_cumsum[:-1, :, :, :], g_cumsum[-1:, :, :, :]  # -1
        h_upper, tot_h = h_cumsum[:-1, :, :, :], h_cumsum[-1:, :, :, :]  # -1
    else:
        g_upper, tot_g = g_cumsum[:, :-1, :, :], g_cumsum[:, -1:, :, :]  # -1
        h_upper, tot_h = h_cumsum[:, :-1, :, :], h_cumsum[:, -1:, :, :]  # -1

    g_lower = tot_g - g_upper
    h_lower = tot_h - h_upper
    
    I = tf.eye(extra_shape[2], batch_shape=(1,1))
    ih_upper = tf.matrix_inverse(h_upper + I)  #TODO: pass lambda through parameters
    ih_lower = tf.matrix_inverse(h_lower + I)
    
    w_up = -tf.matmul(ih_upper, g_upper)
    w_dn = -tf.matmul(ih_lower, g_lower)
    
    loss_down = 0.5 * tf.matmul(w_dn, g_lower, transpose_a=True)[:,:,0,0]
    loss_up = 0.5 * tf.matmul(w_up, g_upper, transpose_a=True)[:,:,0,0]
    loss_sum = loss_up + loss_down
    
    # print(w_up.shape, w_dn.shape, loss_sum.shape)
    return w_up[:,:,:,0], w_dn[:,:,:,0], loss_sum

    
def common_part(y, b, sorted_thresholds, features, label, bias, extra=None, unbalanced_penalty=0, reduce_axis=0, make_transpose=True, use_my_cumsum=True):
    alt_axis = 1 - reduce_axis
    current_loss = tf.reduce_sum(-(label * tf.log_sigmoid(bias) + (1 - label) * tf.log_sigmoid(-bias)))
    # l_curr = -(y * tf.log_sigmoid(b) + (1 - y) * tf.log_sigmoid(-b))
    l_der1 = -y * tf.sigmoid(-b) + (1 - y) * tf.sigmoid(b)
    l_der2 = tf.sigmoid(-b) * tf.sigmoid(b)
    

    if extra is None:
        delta_up, delta_down, loss_sum = default_get_loss(l_der1, l_der2, reduce_axis=reduce_axis, make_transpose=make_transpose, use_my_cumsum=use_my_cumsum)
    else:
        delta_up, delta_down, loss_sum = extra_get_loss(l_der1, l_der2, extra=extra, reduce_axis=reduce_axis, make_transpose=make_transpose, use_my_cumsum=use_my_cumsum)

    
    features_amount_int = tf.shape(features)[reduce_axis]
    features_amount = tf.cast(features_amount_int, dtype=features.dtype)
    frange = tf.cast(tf.range(features_amount_int-1), dtype=features_amount.dtype)
    edge_penalty = tf.reshape(tf.abs(frange - features_amount / 2), 
                              (-1, 1) if reduce_axis == 0 else (1, -1)) 
    whole_loss = loss_sum + current_loss + edge_penalty * unbalanced_penalty
    
    
    #######
    # -----#
    # -----#   -->   iiiIiii (best_loss_argmin0), mmmMmmm (min_loss_axis0)
    # -----#
    #######
    best_loss_argmin0 = tf.argmin(whole_loss, axis=reduce_axis)
    min_loss_axis0 = tf.reduce_min(whole_loss, axis=reduce_axis)
    # mmmMmmm  ->    I (best_loss_argmin1), M (best_loss)
    best_loss_argmin1 = tf.argmin(min_loss_axis0, axis=0)
    best_loss = tf.reduce_min(min_loss_axis0, axis=0)
    best_avg_loss = best_loss / features_amount
    avg_current_loss = current_loss / features_amount
    # iiiIiii  ->  I
    best_index_x = best_loss_argmin0[best_loss_argmin1]
    best_index = tf.identity(best_index_x, name='best_index')
    # print(best_index_x.name, features_amount_int.name, last_cum_l_der2x.name, cum_l_der2_full.name)
    if reduce_axis == 0:
        thr = (sorted_thresholds[best_index, best_loss_argmin1] +
               sorted_thresholds[best_index + 1, best_loss_argmin1]) / 2
        best_delta_up = delta_up[best_index, best_loss_argmin1, :]
        best_delta_down = delta_down[best_index, best_loss_argmin1, :]
    else:
        thr = (sorted_thresholds[best_loss_argmin1, best_index] +
           sorted_thresholds[best_loss_argmin1, best_index + 1]) / 2
        best_delta_up = delta_up[best_loss_argmin1, best_index, :]
        best_delta_down = delta_down[best_loss_argmin1, best_index, :]

        best_delta_up = tf.squeeze(best_delta_up)
        best_delta_down = tf.squeeze(best_delta_down)
    
    return {'features': features,
            'bias': bias,
            'label': label,
            'unbalanced_penalty': unbalanced_penalty,
            #'l_curr': l_curr,
            'l_der1': l_der1,
            'l_der2': l_der2,
            'delta_up': delta_up,
            'delta_down': delta_down,
            'best_delta_up': best_delta_up,
            'best_delta_down': best_delta_down,
            #'cum_l': cum_l,
            #'rev_cum_l': rev_cum_l,
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



def tf_new_ax(ax, cond, reduce_axis=0, name=''):
    #reindex = np.cumsum(cond)-1
    #axT = ax.T
    #return reindex[axT][cond[axT]].reshape((ax.shape[1], -1)).T
    
    
    #reindex = tf.cumsum(tf.to_int32(cond), name='cumsum_'+name)-1  # Normal
    reindex = tfravel(tf.to_int32(mycumsum1(tf.reshape(tf.to_double(cond), (1,-1)), axis=1, name='cumsum_'+name)+.5)-1) # via float
    #reindex = tf.to_int32(tf.cumsum(tf.to_int64(cond), name='cumsum_'+name)-1)  # 

    axT = tf.transpose(ax) if reduce_axis==0 else ax
    newindecies = tf.gather(reindex, axT, axis=0, name='newindecies')
    huge_cond = tf.gather(cond, axT, axis=0, name='huge_cond') # like axT.shape 
    newindecies_ravel = tfravel(newindecies, name='newindecies_ravel')
    huge_cond_ravel = tfravel(huge_cond, name='huge_cond_ravel')
    #print(newindecies_ravel.shape, newindecies_ravel.dtype)
    #print(huge_cond_ravel.shape, huge_cond_ravel.dtype)
    squeezed_ax = tf.boolean_mask(newindecies_ravel, huge_cond_ravel)
    new_axT = tf.reshape(squeezed_ax, (tf.shape(axT)[0], -1) )
    
    return tf.transpose(new_axT) if reduce_axis==0 else new_axT

def create_split_quick(reduce_axis=0, make_transpose=True, use_my_cumsum=True, use_extra=False):
    graph = tf.Graph()
    with graph.as_default():
        info_name='quick_axis_{}_make_transpose_{}_use_my_cumsum_{}_extra_{}_'.format(reduce_axis, make_transpose, use_my_cumsum, use_extra)
        features = tf.placeholder(dtype=tf.float32, shape=(None, None), name='features_' + info_name)
        extra_features = tf.placeholder(dtype=tf.float32, shape=(None, None), name='extra_features_' + info_name)
        label = tf.placeholder(dtype=tf.float32, name='label_' + info_name)
        bias = tf.placeholder(dtype=tf.float32, name='bias_' + info_name)
        ax = tf.placeholder(dtype=tf.int32, shape=(None, None), name='ax_' + info_name)
        unbalanced_penalty = tf.placeholder(dtype=tf.float32, shape=(), name='unbalanced_penalty_' + info_name)
        #gax = tf.transpose(tf.nn.top_k(-tf.transpose(features), k=tf.shape(features)[-2]).indices)
        y = tf.gather(label, ax)[:, :, 0]
        b = tf.gather(bias, ax)[:, :, 0]

        if use_extra:
            if reduce_axis == 0:
                extra = tf.gather(extra_features, ax, name='extra_' + info_name)
            else:
                extra = tf.gather(tf.transpose(extra_features), tf.transpose(ax), name='extra_' + info_name)
                extra = tf.transpose(extra, perm=[1, 0, 2], name='textra_' + info_name)
        else:
            extra = None
        
        # F (N x M),  ax (N, M): ax_{ij} - pos in F_{*j}, 
        # ST_{i,j} = F_{ax_{i,j}, j}
        # sorted_thresholds = tf.gather(features, ax, axis=0)
        sorted_thresholds = take_along_axis(features, ax, reduce_axis=reduce_axis)
        common_tensors = common_part(y, b, sorted_thresholds, features, label, bias, extra=extra, unbalanced_penalty=unbalanced_penalty,
                                     reduce_axis=reduce_axis, make_transpose=make_transpose, use_my_cumsum=use_my_cumsum)
        
        if reduce_axis == 0:
            best_feature = features[:, common_tensors['best_feature_index']]
        else:
            best_feature = features[common_tensors['best_feature_index'], :]
        left_cond = best_feature < common_tensors['thr']
        #left_cond = ax[:common_tensors['best_index']+1, common_tensors['best_feature_index']]
        
        right_cond = tf.logical_not(left_cond)
        ax_left = tf_new_ax(ax, left_cond, name='ax_left', reduce_axis=reduce_axis)
        ax_right = tf_new_ax(ax, right_cond, name='ax_right', reduce_axis=reduce_axis)

    result = {'graph': graph,
              #'gax': gax,
              'extra_features': extra_features,
              'ax': ax,
              'ax_left': ax_left,
              'ax_right': ax_right,
              'left_cond': left_cond,
              'right_cond': right_cond,
             }
    result.update(common_tensors)
    return result


tensors_list = ['thr', 'best_loss', 'best_index', 'best_delta_up', 'best_delta_down', 'current_loss',
                     'best_feature_index',
                     'avg_current_loss', 'best_avg_loss',
                     'ax_left', 'ax_right', 'left_cond', 'right_cond']

####
class SplitMaker:
    def __init__(self, interface, tensors_list):
        self.interface = interface
        self.graph = self.interface['graph']
        self.tensors_list = tensors_list
        self.tensors = {t: self.interface[t] for t in tensors_list}
        
    def make_gax(self, features, axis=0):
        with tf.Session(graph=self.graph) as s:
            gax_val = s.run(self.interface['gax'], {self.interface['features']: (features if axis==0 else features.T)})
        return gax_val if axis==0 else gax_val.T


    def split_old(self, bias, features, label):
        input_tensors = {self.interface[t]: val for t, val in [('features', features), ('bias', bias), ('label', label)]}
        with tf.Session(graph=self.graph) as s:
            tensors_values = s.run(self.tensors, input_tensors)
        return tensors_values
        
    @classmethod
    def make_split_old(cls):
        tensors_list = ['thr', 'best_loss', 'best_index', 'best_delta_up', 'best_delta_down', 'current_loss',
                        'best_feature_index', 'avg_current_loss', 'best_avg_loss']
        split_maker = cls(create_split_interface(), tensors_list)
        split_maker.split = split_maker.split_old
        return split_maker

    @classmethod
    def make_split_new(cls, reduce_axis=0, make_transpose=True, use_my_cumsum=True, use_extra=False):
        tensors_list = ['thr', 'best_loss', 'best_index', 'best_delta_up', 'best_delta_down', 'current_loss',
                        'best_feature_index', 'avg_current_loss', 'best_avg_loss',
                        'ax_left', 'ax_right', 'left_cond', 'right_cond']
        split_maker = cls(create_split_quick(reduce_axis, make_transpose, use_my_cumsum, use_extra), tensors_list)
        split_maker.reduce_axis = reduce_axis
        split_maker.split = split_maker.split_quick
        return split_maker
    
    def quick_run_session(self, s, input_tensors, profile_file = None):
        if profile_file is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            #sess.run(res, options=run_options, run_metadata=run_metadata)
            tensors_values = s.run(self.tensors, input_tensors, options=run_options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(profile_file, 'w') as f:
                f.write(ctf)
        else:
            tensors_values = s.run(self.tensors, input_tensors) # No profile
        return tensors_values

    def split_quick(self, *, bias, features, extra_features, label, ax, params, profile_file = None, sess = None):
        final_params = {"unbalanced_penalty": 0, "lambda": 1}
        final_params.update(params)
        input_tensors = {self.interface[t]: val for t, val in [('features', features), ('extra_features', extra_features), ('bias', bias), ('label', label), ('ax', ax),
                                                                   ('unbalanced_penalty', final_params['unbalanced_penalty'])]}
        if extra_features is None:
            del input_tensors[self.interface['extra_features']]
        #print('make_split_quick graph id:', id(self.interface['features'].graph))
        if sess is None:
            with tf.Session(graph=self.graph) as s:
                tensors_values = self.quick_run_session(s, input_tensors, profile_file=profile_file)
        else:
            tensors_values = self.quick_run_session(sess, input_tensors, profile_file=profile_file)
        return tensors_values

    def split_new(self):
        pass

        
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
