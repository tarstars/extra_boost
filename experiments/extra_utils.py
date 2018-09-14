#
from sklearn.metrics import roc_auc_score
import numpy as np

def get_time_cond(f_time, interval = (0,1)):
    return np.logical_and(interval[0] <= f_time[:,0], f_time[:,0] < interval[1])

def cal_lift(features, f_time, label, feat_index = 0, interval = (0,1)):
    feat_cond = features[:,feat_index]==1
    #print("f", feat_cond.shape, file=sys.stderr)
    time_cond = np.logical_and(interval[0] <= f_time[:,0], f_time[:,0] < interval[1])
    #print("t", time_cond.shape, file=sys.stderr)
    full_cond = np.logical_and(feat_cond, time_cond)
    #print("full", full_cond.shape, file=sys.stderr)
    cond_avg_target = np.mean(label[full_cond])
    avg_target = np.mean(label)
    return cond_avg_target/avg_target

def subpool(pools, f_time, interval = (0,1)):
    time_cond = np.logical_and(interval[0] <= f_time[:,0], f_time[:,0] < interval[1])
    return (pool[time_cond] for pool in pools)

def test_extra_predictions(k_folds, f_time, label, predictions):
    for i in range(k_folds):
        time_interval = (i*1.0/k_folds, (i+1)*1.0/k_folds)
        time_cond = get_time_cond(f_time, time_interval)
        preds = predictions[time_cond]
        y = label[time_cond]
        auc = roc_auc_score(y, preds)
        print(time_interval, 'score =',auc)
        
def time_features(f_time, feat_num=2):
    f_time_shaped = np.array(f_time)
    f_time_shaped.reshape((-1, 1))
    return np.concatenate(list(f_time**d for d in range(feat_num)), axis=1)
