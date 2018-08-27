import sys
# import os
# import json
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import unittest

# TODO: eliminate case when several features values are equal.
#       it is necessary to place threshold out of "claster of equity"

# TODO: manage case of constant features: one constant feature, whole constant feature array, etc.

def tf_print(op, tensors, message='', name=None, cond=None):
    cond = tf.constant(1, dtype=tf.bool) if cond is None else cond
    def print_header(cond):
        if cond:
            sys.stderr.write(message)
        return cond
    def print_bottom(cond):
        if cond:
            sys.stderr.write('\n')
        return cond
    def print_message(i, x, cond):
        if cond:
            sys.stderr.write(message + "     {}:{}\n".format(i, x))
        return x

    prints = (#[tf.py_func(print_header, [cond], tf.bool)] +
              [tf.py_func(print_message, [i, tensor, cond], tensor.dtype) for i, tensor in enumerate(tensors)]
              #+[tf.py_func(print_bottom, [cond], tf.bool)]
             )
    with tf.control_dependencies(prints):
        op = tf.identity(op, name=name)
    return op

def mycumsum0(inputarr):
    s = tf.shape(inputarr)
    z0 = tf.zeros((s[0]//2, s[1]), dtype=inputarr.dtype)
    z1 = tf.concat((z0,inputarr), axis=0)
    s1 = tf.shape(z1)
    reshaped = tf.reshape(z1, (1, s1[0], s1[1], 1))
    filter = tf.ones((s[0], 1, 1, 1), dtype=reshaped.dtype)
    cum = tf.nn.conv2d(reshaped, filter, strides=(1,1,1,1), padding="SAME")
    cum = tf.reshape(cum, (s1[0], s1[1]))[:s[0],:]
    return cum

def hreblock(arr, isplus=False, digit=0):
    s = tf.shape(arr)
    half1 = s[0]//2  # floor
    half2 = (s[0]+1)//2 # ceil
    arr1 = arr[:half2, :s[1]-digit]
    arr2 = arr[half1:,:]
    arr2x = arr2
    if isplus:
        arr2 = tf.cond(tf.equal(tf.shape(arr1)[1], 0), lambda: arr2, lambda: arr2 + arr1[:, -1:])
    #arr1 = tf_print(arr1, [half1, half2, tf.shape(arr1), tf.shape(arr2), tf.shape(arr2x), tf.to_int32(digit)], cond=tf.shape(arr)[1]<2, message='hre')
    return tf.concat([arr1, arr2], axis=1), s[0]%2

def vreblock(arr, isplus=False, digit=0):
    s = tf.shape(arr)
    half1 = s[1]//2  # floor
    half2 = (s[1]+1)//2 # ceil
    arr1 = arr[:s[0]-digit, :half2]
    arr2 = arr[:, half1:]
    if isplus:
        arr2 = tf.cond(tf.equal(tf.shape(arr1)[0], 0), lambda: arr2, lambda: arr2 + arr1[-1:,:])
    return tf.concat([arr1, arr2], axis=0), s[1]%2

def mycumsum1(arr, axis, name):
    forward_reblock, backward_reblock = (hreblock, vreblock) if axis==0 else (vreblock, hreblock) 
    block1, dig = forward_reblock(arr)
    block2, dig2 = forward_reblock(block1)
    cumres = tf.cumsum(block2, axis=axis, name=name)
    res2, outdig = backward_reblock(cumres, isplus=True, digit=dig2)
    res, outdig = backward_reblock(res2, isplus=True, digit=dig)
    #res = tf_print(res, [tf.shape(arr), tf.shape(block1), tf.shape(block2), dig, dig2, tf.shape(cumres),  tf.shape(res2), tf.shape(res)], 
    #               cond = tf.shape(arr)[1] < 3,
    #               message='CumSum1', name='cumres')
    return res

def mycumsum2(arr, axis, name):
    m =  100000.0
    return tf.cast(tf.cumsum(tf.to_int64(arr * m), axis=axis, name=name), dtype=arr.dtype)/tf.cast(m, dtype=arr.dtype)
    

def common_part(y, b, sorted_thresholds, features, label, bias, reduce_axis=0, make_transpose=True, use_my_cumsum=True):
    alt_axis = 1 - reduce_axis
    current_loss = tf.reduce_sum(-(label * tf.log_sigmoid(bias) + (1 - label) * tf.log_sigmoid(-bias)))
    # l_curr = -(y * tf.log_sigmoid(b) + (1 - y) * tf.log_sigmoid(-b))
    l_der1 = -y * tf.sigmoid(-b) + (1 - y) * tf.sigmoid(b)
    l_der2 = tf.sigmoid(-b) * tf.sigmoid(b)
    
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
    
    delta_up = -cum_l_der1 / (cum_l_der2 + 1.0) # -1
    delta_down = -rev_cum_l_der1 / (rev_cum_l_der2 + 1.0)  # -1
    
    #loss_up = cum_l + 0.5 * delta_up * cum_l_der1
    #loss_down = rev_cum_l + 0.5 * delta_down * rev_cum_l_der1
    #whole_loss = (loss_up + loss_down)
    loss_up = 0.5 * delta_up * cum_l_der1
    loss_down = 0.5 * delta_down * rev_cum_l_der1
    whole_loss = (loss_up + loss_down) + current_loss
    
    features_amount_int = tf.shape(features)[reduce_axis]
    features_amount = tf.cast(features_amount_int, dtype=features.dtype)
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
        best_delta_up = delta_up[best_index, best_loss_argmin1]
        best_delta_down = delta_down[best_index, best_loss_argmin1]
    else:
        thr = (sorted_thresholds[best_loss_argmin1, best_index] +
           sorted_thresholds[best_loss_argmin1, best_index + 1]) / 2
        best_delta_up = delta_up[best_loss_argmin1, best_index]
        best_delta_down = delta_down[best_loss_argmin1, best_index]
    
    
    return {'features': features,
            'bias': bias,
            'label': label,
            #'l_curr': l_curr,
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

# along axis 0 only
def take_along_axis(arr, indices, reduce_axis):
    if reduce_axis==0:
        # arr (n x m), indecies (n x m)
        m = tf.shape(arr)[1]
        r = tf.range(m)
        taken_arr = tf.gather(tf.reshape(arr, (-1,)), indices * m + r, axis=0)
    else:
        # arr (m x n), indecies (m x n)
        m = tf.shape(arr)[0]
        n = tf.shape(arr)[1]
        r = tf.reshape(tf.range(m), (-1,1))
        taken_arr = tf.gather(tf.reshape(arr, (-1,)), indices + n * r, axis=0)
        
    return taken_arr

def tfravel(x, name=None):
    return tf.reshape(x, (-1, ), name=name)

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

def create_split_quick(reduce_axis=0, make_transpose=True, use_my_cumsum=True):
    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(dtype=tf.float32, shape=(None, None))
        label = tf.placeholder(dtype=tf.float32)
        bias = tf.placeholder(dtype=tf.float32)
        ax = tf.placeholder(dtype=tf.int32, shape=(None, None))
        #gax = tf.transpose(tf.nn.top_k(-tf.transpose(features), k=tf.shape(features)[-2]).indices)
        y = tf.gather(label, ax)[:, :, 0]
        b = tf.gather(bias, ax)[:, :, 0]
        # F (N x M),  ax (N, M): ax_{ij} - pos in F_{*j}, 
        # ST_{i,j} = F_{ax_{i,j}, j}
        # sorted_thresholds = tf.gather(features, ax, axis=0)
        sorted_thresholds = take_along_axis(features, ax, reduce_axis=reduce_axis)
        common_tensors = common_part(y, b, sorted_thresholds, features, label, bias, reduce_axis=reduce_axis, make_transpose=make_transpose, use_my_cumsum=use_my_cumsum)
        
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
              'ax': ax,
              'ax_left': ax_left,
              'ax_right': ax_right,
              'left_cond': left_cond,
              'right_cond': right_cond,
             }
    result.update(common_tensors)
    return result


split_interface = create_split_interface()
split_graph = split_interface['graph']

split_quick = create_split_quick()
split_quick_graph = split_quick['graph']

split_quick_transpose = create_split_quick(reduce_axis=1, make_transpose=False, use_my_cumsum=True)
split_quick_graph_transpose = split_quick_transpose['graph']

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

tensors_list = ['thr', 'best_loss', 'best_index', 'best_delta_up', 'best_delta_down', 'current_loss',
                     'best_feature_index',
                     'avg_current_loss', 'best_avg_loss',
                     'ax_left', 'ax_right', 'left_cond', 'right_cond']

quick_tensors = {t: split_quick[t] for t in tensors_list}
quick_tensors_transpose = {t: split_quick_transpose[t] for t in tensors_list}

def quick_run_session(s, quick_input_tensors, transposed_feature=False, profile_file = None):
    current_tensors = quick_tensors_transpose if transposed_feature else quick_tensors
    if profile_file is not None:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        #sess.run(res, options=run_options, run_metadata=run_metadata)
        tensors_values = s.run(current_tensors, quick_input_tensors, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(profile_file, 'w') as f:
            f.write(ctf)
    else:
        tensors_values = s.run(current_tensors, quick_input_tensors) # No profile
    return tensors_values

def make_split_quick(bias, features, label, ax, transposed_feature=False, profile_file = None, sess = None):
    current_split = split_quick_transpose if transposed_feature else split_quick
    current_graph = split_quick_graph_transpose if transposed_feature else split_quick_graph
    quick_input_tensors = {current_split[t]: val for t, val in [('features', features), ('bias', bias), ('label', label), ('ax', ax)]}
    if sess is None:
        with tf.Session(graph=current_graph) as s:
            tensors_values = quick_run_session(s, quick_input_tensors, transposed_feature=transposed_feature, profile_file=profile_file)
    else:
        tensors_values = quick_run_session(sess, quick_input_tensors, transposed_feature=transposed_feature, profile_file=profile_file)
    return tensors_values

def make_gax(features, axis=0):
    with tf.Session(graph=split_graph) as s:
        gax_val = s.run(split_interface['gax'], {split_interface['features']: (features if axis==0 else features.T)})
    return gax_val if axis==0 else gax_val.T
        
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
