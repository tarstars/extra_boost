from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from cumsum_zoo import mycumsum1

# from tf_debug import tf_print
from tf_utils import tfravel, take_along_axis


# TODO: eliminate case when several features values are equal.
#       it is necessary to place threshold out of "claster of equity"

# TODO: manage case of constant features: one constant feature, whole constant feature array, etc.


@dataclass
class CumsumDerivatives:
    """Forward and backward cumulative sums for the first and the second derivatives."""

    straight_first: tf.Tensor
    straight_second: tf.Tensor
    reverse_first: tf.Tensor
    reverse_second: tf.Tensor


def cumsum_two_derivatives(
    l_der1: tf.Tensor,
    l_der2: tf.Tensor,
    reduce_axis: int,
    make_transpose: bool,
    use_my_cumsum: bool,
) -> CumsumDerivatives:
    """Calculate cumulative sum of the first and second derivative matrices

    :param l_der1: a matrix with first order derivatives
    :param l_der2: a matrix with second order derivatives
    :param reduce_axis: 0 or 1 - direction of objects in the feature-objects matrix
    :param make_transpose: transpose input matrix
    :param use_my_cumsum: use custom cumsum function
    :return: cumulative and return the cumulative matrixes of the first and second derivatives
    """
    alt_axis = 1 - reduce_axis

    # Choice of 3 cases
    # cum_l_der1_full = tf.cumsum(l_der1, axis=reduce_axis)  # -1
    # cum_l_der2_full = tf.cumsum(l_der2, axis=reduce_axis)  # -1
    der_joined = tf.concat([l_der1, l_der2], axis=alt_axis, name="der_joined")

    # Transpose if needed
    der_joined_input = (
        tf.transpose(a=der_joined) if make_transpose else tf.identity(der_joined)
    )
    # Axis to calc cumsum
    cum_axis = alt_axis if make_transpose else reduce_axis
    # Procedure to calc cumsum
    cum_proc = mycumsum1 if use_my_cumsum else tf.cumsum
    # Calc cumsum
    cumsum_der_joined_output = cum_proc(
        der_joined_input, axis=cum_axis, name="cumsum_der_joined"
    )
    # Reverse transpose
    cumsum_der_joined = (
        tf.transpose(a=cumsum_der_joined_output)
        if make_transpose
        else tf.identity(cumsum_der_joined_output)
    )

    # +++++++++++++++++
    if reduce_axis == 0:
        cum_l_der1_full = cumsum_der_joined[:, : tf.shape(input=l_der1)[1]]
        cum_l_der2_full = cumsum_der_joined[:, tf.shape(input=l_der1)[1] :]
    else:
        cum_l_der1_full = cumsum_der_joined[: tf.shape(input=l_der1)[0], :]
        cum_l_der2_full = cumsum_der_joined[tf.shape(input=l_der1)[0] :, :]

    if reduce_axis == 0:
        cum_l_der1, tot_der1 = cum_l_der1_full[:-1, :], cum_l_der1_full[-1, :]  # -1
        cum_l_der2, tot_der2 = cum_l_der2_full[:-1, :], cum_l_der2_full[-1, :]  # -1
    else:
        cum_l_der1, tot_der1 = cum_l_der1_full[:, :-1], cum_l_der1_full[:, -1:]  # -1
        cum_l_der2, tot_der2 = cum_l_der2_full[:, :-1], cum_l_der2_full[:, -1:]  # -1

    # rev_cum_l_der1 = tf.cumsum(l_der1, reverse=True)[1:, :]  # -1
    # rev_cum_l_der2 = tf.cumsum(l_der2, reverse=True)[1:, :]  # -1
    rev_cum_l_der1 = tot_der1 - cum_l_der1
    rev_cum_l_der2 = tot_der2 - cum_l_der2
    return CumsumDerivatives(
        straight_first=cum_l_der1,
        straight_second=cum_l_der2,
        reverse_first=rev_cum_l_der1,
        reverse_second=rev_cum_l_der2,
    )


@dataclass
class LossResult:
    """Contains the upper half of the loss, the lower half and the total loss."""

    upper: tf.Tensor
    lower: tf.Tensor
    total: tf.Tensor


def default_get_loss(
    l_der1: tf.Tensor,
    l_der2: tf.Tensor,
    reduce_axis: int,
    make_transpose: bool,
    use_my_cumsum: bool,
) -> LossResult:
    """Calculates the loss function for vanilla gbdt.

    :param l_der1: derivative of the first order
    :param l_der2: second order derivative
    :param reduce_axis: 0 or 1 - axis of reduce
    :param make_transpose: whether we want to transpose input matrix
    :param use_my_cumsum: whether we want to use custom function for cumulative summation
    :return:upper half of the loss, down half and the full loss
    """
    cs = cumsum_two_derivatives(
        l_der1=l_der1,
        l_der2=l_der2,
        reduce_axis=reduce_axis,
        make_transpose=make_transpose,
        use_my_cumsum=use_my_cumsum,
    )

    delta_up = -cs.straight_first / (cs.straight_second + 1.0)  # -1 TODO: pass lambda
    delta_down = -cs.reverse_first / (cs.reverse_second + 1.0)  # -1

    # loss_up = cum_l + 0.5 * delta_up * cum_l_der1
    # loss_down = rev_cum_l + 0.5 * delta_down * rev_cum_l_der1
    loss_up = 0.5 * delta_up * cs.straight_first
    loss_down = 0.5 * delta_down * cs.reverse_first
    loss_sum = loss_up + loss_down

    return LossResult(
        upper=tf.expand_dims(delta_up, axis=2),
        lower=tf.expand_dims(delta_down, axis=2),
        total=loss_sum,
    )


def extra_get_loss(l_der1: tf.Tensor, l_der2: tf.Tensor, extra, reduce_axis):
    """Calculates the loss for extra_boost.

    :param l_der1: first derivative
    :param l_der2: second derivative
    :param extra: extra features
    :param reduce_axis: 0 or 1 - direction of reduction
    :return: upper_loss, down_loss and full loss for extra_boost
    """
    extra_shape = tf.shape(input=extra)
    extra_1 = tf.reshape(extra, (extra_shape[0], extra_shape[1], 1, extra_shape[2]))
    extra_2 = tf.reshape(extra, (extra_shape[0], extra_shape[1], extra_shape[2], 1))
    l_der_shape = tf.shape(input=l_der2)
    l_der2_multy = tf.reshape(l_der2, (l_der_shape[0], l_der_shape[1], 1, 1))
    l_der1_multy = tf.reshape(l_der1, (l_der_shape[0], l_der_shape[1], 1, 1))
    h = l_der2_multy * extra_1 * extra_2
    lder_extra = l_der1_multy * extra_2

    h_cumsum = tf.cumsum(h, axis=reduce_axis)
    g_cumsum = tf.cumsum(lder_extra, axis=reduce_axis)

    if reduce_axis == 0:
        g_upper, tot_g = g_cumsum[:-1, :, :, :], g_cumsum[-1:, :, :, :]  # -1
        h_upper, tot_h = h_cumsum[:-1, :, :, :], h_cumsum[-1:, :, :, :]  # -1
    else:
        g_upper, tot_g = g_cumsum[:, :-1, :, :], g_cumsum[:, -1:, :, :]  # -1
        h_upper, tot_h = h_cumsum[:, :-1, :, :], h_cumsum[:, -1:, :, :]  # -1

    g_lower = tot_g - g_upper
    h_lower = tot_h - h_upper

    mat_i = tf.eye(extra_shape[2], batch_shape=(1, 1), dtype=tf.float32)
    ih_upper = tf.linalg.inv(h_upper + mat_i)  # TODO: pass lambda through parameters
    ih_lower = tf.linalg.inv(h_lower + mat_i)

    w_up = -tf.matmul(ih_upper, g_upper)
    w_dn = -tf.matmul(ih_lower, g_lower)

    loss_down = 0.5 * tf.matmul(w_dn, g_lower, transpose_a=True)[:, :, 0, 0]
    loss_up = 0.5 * tf.matmul(w_up, g_upper, transpose_a=True)[:, :, 0, 0]
    loss_sum = loss_up + loss_down

    # print(w_up.shape, w_dn.shape, loss_sum.shape)
    return LossResult(upper=w_up[:, :, :, 0], lower=w_dn[:, :, :, 0], total=loss_sum)


@dataclass
class CommonResult:
    features: tf.Tensor
    bias: np.ndarray
    label: tf.Tensor
    unbalanced_penalty: float
    l_der1: tf.Tensor
    l_der2: tf.Tensor
    delta_up: tf.Tensor
    delta_down: tf.Tensor
    best_delta_up: tf.Tensor
    best_delta_down: tf.Tensor
    whole_loss: tf.Tensor
    best_loss_argmin0: tf.Tensor
    min_loss_axis0: tf.Tensor
    best_loss_argmin1: tf.Tensor
    best_feature_index: tf.Tensor
    best_loss: tf.Tensor
    best_avg_loss: tf.Tensor
    best_index: tf.Tensor
    T: tf.Tensor
    thr: tf.Tensor
    y: tf.Tensor
    current_loss: tf.Tensor
    avg_current_loss: tf.Tensor

    extra_features: tf.Tensor
    ax: tf.Tensor
    ax_left: tf.Tensor
    ax_right: tf.Tensor
    left_cond: tf.Tensor
    right_cond: tf.Tensor



def common_part(
    y,
    b,
    sorted_thresholds,
    features,
    label,
    bias,
    extra=None,
    unbalanced_penalty=0,
    reduce_axis=0,
    make_transpose=True,
    use_my_cumsum=True,
) -> CommonResult:
    """ Creates the common part of vanilla and extra GBDT algorithm.

    :param y: target
    :param b: b
    :param sorted_thresholds:  sorted thresholds
    :param features: split features
    :param label: target
    :param bias: suggested bias
    :param extra: extrapolating features
    :param unbalanced_penalty: penalty for different heights
    :param reduce_axis: the direction of reduce operation
    :param make_transpose: use transposed matrix
    :param use_my_cumsum: use the custom cumulative sum procedure
    :return: dict of named tensorflow graph nodes
    """
    current_loss = tf.reduce_sum(
        input_tensor=-(
            label * tf.math.log_sigmoid(bias) + (1 - label) * tf.math.log_sigmoid(-bias)
        )
    )
    l_der1 = -y * tf.sigmoid(-b) + (1 - y) * tf.sigmoid(b)
    l_der2 = tf.sigmoid(-b) * tf.sigmoid(b)

    if extra is None:
        # delta_up, delta_down, loss_sum \
        loss_info = default_get_loss(
            l_der1,
            l_der2,
            reduce_axis=reduce_axis,
            make_transpose=make_transpose,
            use_my_cumsum=use_my_cumsum,
        )
    else:
        # delta_up, delta_down, loss_sum
        loss_info = extra_get_loss(
            l_der1, l_der2, extra=extra, reduce_axis=reduce_axis,
        )

    features_amount_int = tf.shape(input=features)[reduce_axis]
    features_amount = tf.cast(features_amount_int, dtype=features.dtype)
    frange = tf.cast(tf.range(features_amount_int - 1), dtype=features_amount.dtype)
    edge_penalty = tf.reshape(
        tf.abs(frange - features_amount / 2), (-1, 1) if reduce_axis == 0 else (1, -1)
    )
    whole_loss = loss_info.total + current_loss + edge_penalty * unbalanced_penalty

    #######
    # -----#
    # -----#   -->   iiiIiii (best_loss_argmin0), mmmMmmm (min_loss_axis0)
    # -----#
    #######
    best_loss_argmin0 = tf.argmin(input=whole_loss, axis=reduce_axis)
    min_loss_axis0 = tf.reduce_min(input_tensor=whole_loss, axis=reduce_axis)
    # mmmMmmm  ->    I (best_loss_argmin1), M (best_loss)
    best_loss_argmin1 = tf.argmin(input=min_loss_axis0, axis=0)
    best_loss = tf.reduce_min(input_tensor=min_loss_axis0, axis=0)
    best_avg_loss = best_loss / features_amount
    avg_current_loss = current_loss / features_amount
    # iiiIiii  ->  I
    best_index_x = best_loss_argmin0[best_loss_argmin1]
    best_index = tf.identity(best_index_x, name="best_index")
    # print(best_index_x.name, features_amount_int.name, last_cum_l_der2x.name, cum_l_der2_full.name)
    if reduce_axis == 0:
        thr = (
            sorted_thresholds[best_index, best_loss_argmin1]
            + sorted_thresholds[best_index + 1, best_loss_argmin1]
        ) / 2
        best_delta_up = loss_info.upper[best_index, best_loss_argmin1, :]
        best_delta_down = loss_info.lower[best_index, best_loss_argmin1, :]
    else:
        thr = (
            sorted_thresholds[best_loss_argmin1, best_index]
            + sorted_thresholds[best_loss_argmin1, best_index + 1]
        ) / 2
        best_delta_up = loss_info.upper[best_loss_argmin1, best_index, :]
        best_delta_down = loss_info.lower[best_loss_argmin1, best_index, :]

        best_delta_up = tf.squeeze(best_delta_up)
        best_delta_down = tf.squeeze(best_delta_down)

    return CommonResult(
        features=features,
        bias=bias,
        label=label,
        unbalanced_penalty=unbalanced_penalty,
        l_der1=l_der1,
        l_der2=l_der2,
        delta_up=loss_info.upper,
        delta_down=loss_info.lower,
        best_delta_up=best_delta_up,
        best_delta_down=best_delta_down,
        whole_loss=whole_loss,
        best_loss_argmin0=best_loss_argmin0,
        min_loss_axis0=min_loss_axis0,
        best_loss_argmin1=best_loss_argmin1,
        best_feature_index=best_loss_argmin1,
        best_loss=best_loss,
        best_avg_loss=best_avg_loss,
        best_index=best_index,
        T=sorted_thresholds,
        thr=thr,
        y=y,
        current_loss=current_loss,
        avg_current_loss=avg_current_loss,
        extra_features=None,
        ax=None,
        ax_left=None,
        ax_right=None,
        left_cond=None,
        right_cond=None,
    )


def tf_new_ax(ax, cond, reduce_axis=0, name=""):
    # reindex = np.cumsum(cond)-1
    # axT = ax.T
    # return reindex[axT][cond[axT]].reshape((ax.shape[1], -1)).T
    # reindex = tf.cumsum(tf.to_int32(cond), name='cumsum_'+name)-1  # Normal
    reindex = tfravel(
        tf.cast(
            mycumsum1(
                tf.reshape(tf.cast(cond, dtype=tf.float32), (1, -1)),
                axis=1,
                name="cumsum_" + name,
            )
            + 0.5,
            dtype=tf.int32,
        )
        - 1
    )  # via float
    # reindex = tf.to_int32(tf.cumsum(tf.to_int64(cond), name='cumsum_'+name)-1)  #

    ax_t = tf.transpose(a=ax) if reduce_axis == 0 else ax
    newindecies = tf.gather(reindex, ax_t, axis=0, name="newindecies")
    huge_cond = tf.gather(cond, ax_t, axis=0, name="huge_cond")  # like axT.shape
    newindecies_ravel = tfravel(newindecies, name="newindecies_ravel")
    huge_cond_ravel = tfravel(huge_cond, name="huge_cond_ravel")
    # print(newindecies_ravel.shape, newindecies_ravel.dtype)
    # print(huge_cond_ravel.shape, huge_cond_ravel.dtype)
    squeezed_ax = tf.boolean_mask(tensor=newindecies_ravel, mask=huge_cond_ravel)
    new_ax_t = tf.reshape(squeezed_ax, (tf.shape(input=ax_t)[0], -1))

    return tf.transpose(a=new_ax_t) if reduce_axis == 0 else new_ax_t


def make_gax(features: tf.Tensor, axis: int = 0) -> tf.Tensor:
    """Creates a matrix with argsort indices by columns

    :param features: split features
    :param axis: axis by wich we enumerate objects of our dataset
    :return: substitution matrix to sort features
    """
    proper_features = features if axis == 0 else features.T
    ax = tf.transpose(
        a=tf.nn.top_k(
            -tf.transpose(a=proper_features), k=tf.shape(input=proper_features)[-2]
        ).indices
    )
    return ax if axis == 0 else ax.T
