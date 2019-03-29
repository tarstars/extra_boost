import tensorflow as tf


def mycumsum0(inputarr):
    s = tf.shape(inputarr)
    z0 = tf.zeros((s[0]//2, s[1]), dtype=inputarr.dtype)
    z1 = tf.concat((z0, inputarr), axis=0)
    s1 = tf.shape(z1)
    reshaped = tf.reshape(z1, (1, s1[0], s1[1], 1))
    filter = tf.ones((s[0], 1, 1, 1), dtype=reshaped.dtype)
    cum = tf.nn.conv2d(reshaped, filter, strides=(1, 1, 1, 1), padding="SAME")
    cum = tf.reshape(cum, (s1[0], s1[1]))[:s[0], :]
    return cum


def hreblock(arr, isplus=False, digit=0):
    s = tf.shape(arr)
    half1 = s[0]//2  # floor
    half2 = (s[0]+1)//2  # ceil
    arr1 = arr[:half2, :s[1]-digit]
    arr2 = arr[half1:, :]
    # arr2x = arr2
    if isplus:
        arr2 = tf.cond(tf.equal(tf.shape(arr1)[1], 0), lambda: arr2, lambda: arr2 + arr1[:, -1:])
    #  arr1 = tf_print(arr1, [half1, half2, tf.shape(arr1), tf.shape(arr2), tf.shape(arr2x), tf.to_int32(digit)],
    #  cond=tf.shape(arr)[1]<2, message='hre')
    return tf.concat([arr1, arr2], axis=1), s[0] % 2


def vreblock(arr, isplus=False, digit=0):
    s = tf.shape(arr)
    half1 = s[1]//2  # floor
    half2 = (s[1]+1)//2  # ceil
    arr1 = arr[:s[0]-digit, :half2]
    arr2 = arr[:, half1:]
    if isplus:
        arr2 = tf.cond(tf.equal(tf.shape(arr1)[0], 0), lambda: arr2, lambda: arr2 + arr1[-1:, :])
    return tf.concat([arr1, arr2], axis=0), s[1] % 2


def mycumsum1(arr, axis, name):
    forward_reblock, backward_reblock = (hreblock, vreblock) if axis == 0 else (vreblock, hreblock)
    block1, dig = forward_reblock(arr)
    block2, dig2 = forward_reblock(block1)
    cumres = tf.cumsum(block2, axis=axis, name=name)
    res2, outdig = backward_reblock(cumres, isplus=True, digit=dig2)
    res, outdig = backward_reblock(res2, isplus=True, digit=dig)
    # res = tf_print(res, [tf.shape(arr), tf.shape(block1), tf.shape(block2), dig, dig2, tf.shape(cumres),
    #  tf.shape(res2), tf.shape(res)],
    #               cond = tf.shape(arr)[1] < 3,
    #               message='CumSum1', name='cumres')
    return res


def mycumsum2(arr, axis, name):
    m = 100000.0
    return tf.cast(tf.cumsum(tf.to_int64(arr * m), axis=axis, name=name), dtype=arr.dtype)/tf.cast(m, dtype=arr.dtype)
