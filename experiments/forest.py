import sys, os, io, json, numpy as np, random, time
import graphviz

import tensorflow.compat.v1 as tf
from split import SplitMaker

# import cProfile

#right now unused
def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper


class EMatrix:
    def __init__(self, features, label, *, extra_features=None, bias=None, gax=None, splitgax = False):
        self.bias = bias
        self.features = features
        self.extra_features = extra_features
        self.label = label
        self.gax = gax
        self.splitgax = splitgax

        
class LeafData:
    def __init__(self, info):
        self.val = info['prediction'] * info['learning_rate']
        self.train_size = info['ematrix'].label.shape[0]
        self.avg_target = np.mean(info['ematrix'].label, axis=0)[0]
        
    def to_text(self, floatformat = '.6f'):
        valfloatformat = ':' + floatformat if isinstance(self.val, float) else ''
        return ('{'+ valfloatformat + '} ({})\n({:'+floatformat+'})').format(self.val, self.train_size, self.avg_target)
    
    def shape(self):
        return 'box'

class SplitData:
    def __init__(self, val):
        self.val = val
        
    def to_text(self, floatformat = '.4f'):
        return ('f_{{{ind}}} < {thr:'+floatformat+'}').format(ind=self.val['best_feature_index'], thr=self.val['thr'])
    
        
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.depth = 0
        self.val = None
        self.id = None
        
    def to_text(self, floatformat = '.6f'):
        return self.val.to_text(floatformat)
    
    def shape(self):
        return 'circle'
    
def init_id_helper(node, current_id):
    node.id = current_id[0]
    current_id[0] += 1
    if not isinstance(node, TreeNode):
        return
    init_id_helper(node.left, current_id)
    init_id_helper(node.right, current_id)

def init_id(root):
    current_id = [0]
    init_id_helper(root, current_id)
    return current_id[0] 


def init_arrays_helper(node, arrays):
    if not isinstance(node, TreeNode):
        arrays['is_leaf'][node.id] = 1
        arrays['leaf_data'][node.id, :] = node.val  # Leaf
        return
    init_arrays_helper(node.left, arrays)
    init_arrays_helper(node.right, arrays)
    arrays['yes_node'][node.id] = node.left.id
    arrays['no_node'][node.id] = node.right.id
    arrays['thresholds'][node.id] = node.val.val['thr']
    arrays['features'][node.id] = node.val.val['best_feature_index']
    arrays['is_leaf'][node.id] = 0
    arrays['depths'][node.id] = node.depth


def init_arrays(root, n, weights_num=1):
    def empty_array():
        return np.zeros(n, dtype=np.int32)
    arrays = dict(features=empty_array(),
                  thresholds=np.zeros(n, dtype=np.float32),
                  yes_node=empty_array(),
                  no_node=empty_array(),
                  is_leaf=empty_array(),
                  depths=empty_array(),
                  leaf_data=np.zeros((n, weights_num), dtype=np.float32)
                 )
    init_arrays_helper(root, arrays)
    arrays['treedepth'] = np.max(arrays['depths'])
    return arrays

###################################################
#   Build forest
###################################################

def prior_finish(params, info, parent):
    if parent is None:
        return False
    return params['max_depth'] <= parent.depth

def post_finish(params, info, left_info, right_info, split_info, parent):
    if left_info['ematrix'].label.shape[0] < 2:
        return True
    if right_info['ematrix'].label.shape[0] < 2:
        return True
    return False

def new_ax(ax, cond):
    reindex = np.cumsum(cond)-1
    axT = ax.T
    return reindex[axT][cond[axT]].reshape((ax.shape[1], -1)).T

time1 = 0
time2 = 0
time3 = []

def getslice(arr, slice, axis):
    if arr is None:
        return None
    return arr[:, slice] if axis==1 else arr[slice, :]

def split_ematrix(ematrix, depth, params, split_maker, sess=None):
    global time1, time2, time3
    if ematrix.gax is not None:
        start = time.process_time()
        #print(ematrix.features.shape, file=sys.stderr)
        #if (ematrix.features.shape[0]<=2):
        #    print(ematrix.gax, file=sys.stderr)
        split_info = split_maker.split(bias=ematrix.bias, features=ematrix.features, 
                                       extra_features=ematrix.extra_features,
                                       label=ematrix.label, ax=ematrix.gax, params=params,
                                      profile_file = ('profile_cumsum3.json' if depth==1 else None), sess=sess)
        dif = time.process_time() - start
        time1 += dif
        time3 += [dif]
        
        cond_left = split_info['left_cond']
        cond_right = split_info['right_cond']
        ax_left = split_info['ax_left']
        ax_right = split_info['ax_right']
    else:
        split_info = make_split(ematrix.bias, ematrix.features, ematrix.label)
        cond_left = best_feature < split_info['thr']
        cond_right = np.logical_not(cond_left)
        ax_left = ax_right = None

        
    start = time.process_time()
    features = ematrix.features
    extra_features = ematrix.extra_features
    bias = ematrix.bias
    label = ematrix.label
        
    if False:
        thr = split_info['thr']        

        best_feature = features[:, split_info['best_feature_index']] 

        cond_left = best_feature < thr
        if not np.all((cond_left == split_info['left_cond']).ravel()):
                print("AAAA", file=sys.stderr)
        cond_right = np.logical_not(cond_left)
        if ematrix.gax is not None:
            ax_left = new_ax(ematrix.gax, cond_left)
            ax_right = new_ax(ematrix.gax, cond_right)
            if not np.all((ax_left == split_info['ax_left']).ravel()):
                print("AAAA", file=sys.stderr)
            if not np.all((ax_right == split_info['ax_right']).ravel()):
                print("BBBB", file=sys.stderr)
        else:
            ax_left = ax_right = None
    
    axis1 = split_maker.reduce_axis == 1
    left_ematrix = EMatrix(features=getslice(features, cond_left, split_maker.reduce_axis), 
                           extra_features=getslice(extra_features, cond_left, split_maker.reduce_axis),
                           label=label[cond_left], bias=bias[cond_left], gax=ax_left)
    right_ematrix = EMatrix(features=getslice(features, cond_right, split_maker.reduce_axis), 
                           extra_features=getslice(extra_features, cond_right, split_maker.reduce_axis),
                            label=label[cond_right], bias=bias[cond_right], gax=ax_right)
    left_info = {'prediction': split_info['best_delta_up'], 'ematrix': left_ematrix, 'sess': sess}
    right_info = {'prediction': split_info['best_delta_down'], 'ematrix': right_ematrix, 'sess': sess}
    time2 += time.process_time() - start
    return left_info, right_info, split_info

def build_tree_helper(params, info, parent, split_maker):
    info['learning_rate'] = params['learning_rate']
    if False and parent and parent.depth < 6:
        print("{d}".format(d=parent.depth) 
              if parent else '---',
              "".format(shape=info['ematrix'].label.shape[0]),
              end=' ', file=sys.stderr)
    if prior_finish(params, info, parent):
        return LeafData(info)
    node = TreeNode()
    node.depth = parent.depth + 1 if parent else 1
    
    left_info, right_info, split_info = split_ematrix(info['ematrix'], node.depth, params, split_maker=split_maker, sess=info['sess'])
    if post_finish(params, info, left_info, right_info, split_info, parent):
        #print(split_info['right_info']['ematrix'].label.shape[0])
        #print(split_info)
        return LeafData(info)

    node.val = SplitData(split_info)
    node.left = build_tree_helper(params, left_info, parent=node, split_maker=split_maker)
    node.right = build_tree_helper(params, right_info, parent=node, split_maker=split_maker)  
    return node


def build_tree(params, ematrix, split_maker, sess=None):
    if split_maker is None:
        # not neccesary
        split_maker = SplitMaker.make_split_new()
    info = {'ematrix': ematrix, 'sess': sess}
    if ematrix.splitgax and ematrix.gax is None:
        ematrix.gax = split_maker.make_gax(ematrix.features, axis=split_maker.reduce_axis)
    return build_tree_helper(params, info=info, parent=None, split_maker=split_maker)

def tree_apply(tree_arrays, features, extra_features=None, reduce_axis=0):
    qi = np.zeros(features.shape[reduce_axis], dtype=np.int32)
    for current_depth in range(tree_arrays['treedepth']):        
        fi = tree_arrays['features'][qi]
        f = np.choose(fi, features.T if reduce_axis == 0 else features)
        t = tree_arrays['thresholds'][qi]
        #print(qi, fi, f, t)
        #if current_depth == 0: 
        #    print(fi, f.shape, features.shape, f)
        answer = (f < t)*1
        new_qi = answer*tree_arrays['yes_node'][qi] + (1-answer)*tree_arrays['no_node'][qi]
        qi = new_qi
    if extra_features is None:
        assert tree_arrays['leaf_data'].shape[1]==1, 'extra_features needed'
        leaf_data = tree_arrays['leaf_data'][qi, 0]
    else:
        leaf_data = (tree_arrays['leaf_data'][qi, :]*(extra_features.T if reduce_axis == 1 else extra_features)).sum(axis=1)
    return leaf_data

######################################################################
#            Forest
######################################################################

def tree2gv(tree):
    result = graphviz.Graph('ni')
    #result.attr(size='12,0')
    tree2gv_helper(tree, result, '')
    return result

def tree2gv_helper(node, result, id):
    idn = id
    result.node(idn, node.to_text(), shape='box') # node.shape())
    if isinstance(node, LeafData):
        return
    if node.left is not None:
        idl = id + '0'
        tree2gv_helper(node.left, result, idl)
        result.edge(idn, idl)
    if node.right is not None:
        idr = id + '1'
        tree2gv_helper(node.right, result, idr)
        result.edge(idn, idr)

######################################################################
#            Forest
######################################################################

class EBooster:
    def __init__(self, forest):
        self.forest = forest
    
    def predict(self, features, tree_limit = None, extra_features=None, reduce_axis=0):
        pred = np.zeros(features.shape[0], dtype=np.float32)
        for tree, tree_arrays in (self.forest if tree_limit is None else self.forest[:tree_limit]):
            pred = pred + tree_apply(tree_arrays, features, extra_features=extra_features, reduce_axis=reduce_axis)
        return pred 


def train(params, ematrix, num_boost_round = 10):
    start_params = {'max_depth': 5, 'learning_rate': 0.3, 'splitgax': False, 'transposed_feature': False, 
                   'progress_callback': None} 
    start_params.update(params)
    
    reduce_axis=1 if start_params['transposed_feature'] else 0
    # TODO Singleton
    split_maker_old = SplitMaker.make_split_old()
    split_maker = SplitMaker.make_split_new(reduce_axis=reduce_axis, make_transpose=(reduce_axis==0), use_extra = ematrix.extra_features is not None)
        
    if start_params['splitgax'] and ematrix.gax is None:
        ematrix.gax = split_maker_old.make_gax(ematrix.features, axis=reduce_axis)
    
    forest = []
    bias = np.zeros(ematrix.label.shape)
    features = ematrix.features
    with tf.Session(graph=split_maker.graph) as s:
        for r in range(num_boost_round):
            # print("\n{} round".format(r), file=sys.stderr)
            tree = build_tree(start_params, EMatrix(ematrix.features, ematrix.label, bias=bias, 
                                                    extra_features=ematrix.extra_features, gax=ematrix.gax,
                                                    splitgax=start_params['splitgax']), split_maker=split_maker, sess=s)
            #print("tree ok, bias shape = {}".format(bias.shape), file=sys.stderr)
            tree_arrays = init_arrays(tree, init_id(tree), weights_num = ematrix.extra_features.shape[1-reduce_axis] if ematrix.extra_features is not None else 1)
            bias_delta = tree_apply(tree_arrays, features=features, extra_features=ematrix.extra_features, reduce_axis=reduce_axis)
            #print("apply ok, bias delta shape = {}".format(bias_delta.shape), file=sys.stderr)
            bias = bias + np.reshape(bias_delta, newshape=bias.shape)
            forest.append((tree, tree_arrays))
            #print("forest appended", file=sys.stderr)
            if start_params['progress_callback'] is not None:
                start_params['progress_callback'](r, num_boost_round)
        
    return EBooster(forest)