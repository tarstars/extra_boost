import tensorflow as tf


def tfravel(x, name=None):
    return tf.reshape(x, (-1, ), name=name)


# along axis 0 or 1
# arr is 2D-Matrix
def take_along_axis(arr, indices, reduce_axis):
    if reduce_axis == 0:
        # arr (n x m), indicies (n x m)
        m = tf.shape(arr)[1]
        r = tf.range(m)
        taken_arr = tf.gather(tf.reshape(arr, (-1,)), indices * m + r, axis=0)
    else:
        # arr (m x n), indicies (m x n)
        m = tf.shape(arr)[0]
        n = tf.shape(arr)[1]
        r = tf.reshape(tf.range(m), (-1, 1))
        taken_arr = tf.gather(tf.reshape(arr, (-1,)), indices + n * r, axis=0)
        
    return taken_arr
