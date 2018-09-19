import tensorflow as tf

def tf_print(op, tensors, message='', name=None, cond=None):
    gcond = tf.constant(True, dtype=tf.bool) if cond is None else cond
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
              [tf.py_func(print_message, [i, tensor, gcond], tensor.dtype) for i, tensor in enumerate(tensors)]
              #+[tf.py_func(print_bottom, [cond], tf.bool)]
             )
    with tf.control_dependencies(prints):
        op = tf.identity(op, name=name)
    return op
