import sys, os, io, json, numpy as np, random, time
from typing import Dict, Any, Optional, Union, List

import graphviz

import tensorflow as tf

from experiments.tf_utils import take_along_axis
from split import make_gax, common_part, tf_new_ax, CommonResult


# import cProfile

# right now unused
def profile(func):
    """Decorator for run function profile"""

    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + ".prof"
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result

    return wrapper


class EMatrix:
    """Container for an extra dataset."""

    def __init__(
        self,
        features: np.ndarray,
        label: np.ndarray,
        *,
        extra_features: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        gax: Optional[np.ndarray] = None,
        splitgax: bool = False,
    ):
        """Creates an extra dataset object.

        :param features: interpolaring features
        :param label: target
        :param extra_features: extrapolating features
        :param bias: expected bias
        :param gax: indices of argsort operation
        :param splitgax: whether to split gax
        """
        self.bias = bias if bias is None else tf.cast(bias, tf.float32)
        self.features = tf.cast(features, tf.float32)
        self.extra_features = tf.cast(extra_features, tf.float32)
        self.label = tf.cast(label, tf.float32)
        self.gax = gax
        self.splitgax = splitgax

    @classmethod
    def from_features_params(
        cls,
        features: np.array,
        extra_features: np.array,
        params: Dict[str, Any],
        label: np.array,
    ) -> "EMatrix":
        """Creates a EMatrix object with the right alignment of data.

        :param features: interpolation features, for splits
        :param extra_features: blending functions for the extrapolation
        :param params: a dictionary with the parameters of calculations
        :param label: a target to predict
        :return: a new EMatrix object with properly aligned data
        """
        perform_transposition = params["transposed_feature"]
        return cls(
            features=features.T if perform_transposition else features,
            extra_features=extra_features.T
            if perform_transposition
            else extra_features,
            label=label,
        )


class LeafData:
    """Describes a leaf of a tree."""
    def __init__(self, info):
        """Creates a new leaf object.

        :param info: leaf information
        """
        self.val = info["prediction"] * info["learning_rate"]
        self.train_size = info["ematrix"].label.shape[0]
        self.avg_target = np.mean(info["ematrix"].label, axis=0)[0]

    def to_text(self, floatformat: str = ".6f") -> str:
        """Renders leaf information for the graph representation.

        :param floatformat: float precision
        :return: the text representation of a leaf
        """
        valfloatformat = ":" + floatformat if isinstance(self.val, float) else ""
        return ("{" + valfloatformat + "} ({})\n({:" + floatformat + "})").format(
            self.val, self.train_size, self.avg_target
        )

    def shape(self) -> str:
        """Defines the shape of a leaf in the graph representation of a tree.

        :return: the shape name
        """
        return "box"


class SplitData:
    """Describes a split."""
    def __init__(self, val: Dict[str, Any]):
        """Creates a split description.

        :param val: a bunch of named data
        """
        self.val = val

    def to_text(self, floatformat=".4f"):
        """Renders information for the graph representation.

        :param floatformat: format to use for floats.
        :return: the string representation of a tree
        """
        return ("f_{{{ind}}} < {thr:" + floatformat + "}").format(
            ind=self.val["best_feature_index"], thr=self.val["thr"]
        )


class TreeNode:
    """Represents a node of a tree."""

    def __init__(self):
        """Creates an empty tree node."""
        self.left = None
        self.right = None
        self.depth = 0
        self.val = None
        self.id = None

    def to_text(self, floatformat: str = ".6f") -> str:
        """Renders a representation of the current node.

        :param floatformat: format to use
        :return: string representation of the current node
        """
        return self.val.to_text(floatformat)

    def shape(self) -> str:
        """Shape of a node in the graph representation.

        :return: the shape of a nodes
        """
        return "circle"


def init_id_helper(node: TreeNode, current_id: List[int]) -> None:
    """Traverses a tree and assigns id to each node.

    :param node: the current node of a tree
    :param current_id: the id of the current node
    """
    node.id = current_id[0]
    current_id[0] += 1
    if not isinstance(node, TreeNode):
        return
    init_id_helper(node.left, current_id)
    init_id_helper(node.right, current_id)


def init_id(root: TreeNode):
    """Enumerates nodes of the current tree.

    :param root: the root of the current tree.
    :return: the number of nodes in the tree
    """
    current_id = [0]
    init_id_helper(root, current_id)
    return current_id[0]


def init_arrays_helper(node: TreeNode, arrays: Dict[str, np.ndarray]) -> None:
    """Fills arrays with the information from the current node.

    :param node: the current node
    :param arrays: the collection of arrays associated with the tree
    """
    if not isinstance(node, TreeNode):
        arrays["is_leaf"][node.id] = 1
        arrays["leaf_data"][node.id, :] = node.val  # Leaf
        return
    init_arrays_helper(node.left, arrays)
    init_arrays_helper(node.right, arrays)
    arrays["yes_node"][node.id] = node.left.id
    arrays["no_node"][node.id] = node.right.id
    arrays["thresholds"][node.id] = node.val.val.thr
    arrays["features"][node.id] = node.val.val.best_feature_index
    arrays["is_leaf"][node.id] = 0
    arrays["depths"][node.id] = node.depth


def init_arrays(root: TreeNode, n: int, weights_num: int = 1) -> Dict[str, np.ndarray]:
    """Initializes arrays associated with a tree.

    :param root: the root of a tree
    :param n: the size of a tree
    :param weights_num: the number of weights
    :return: a collection of arrays
    """
    def empty_array():
        return np.zeros(n, dtype=np.int32)

    arrays = dict(
        features=empty_array(),
        thresholds=np.zeros(n, dtype=np.float32),
        yes_node=empty_array(),
        no_node=empty_array(),
        is_leaf=empty_array(),
        depths=empty_array(),
        leaf_data=np.zeros((n, weights_num), dtype=np.float32),
    )
    init_arrays_helper(root, arrays)
    arrays["treedepth"] = np.max(arrays["depths"])
    return arrays


def prior_finish(params: Dict[str, Any], info: Dict[str, Any], parent: TreeNode) -> bool:
    """Determines whether to stop the process before making a split.

    :param params: parameters of the process
    :param info: step-related information
    :param parent: parent tree node
    :return: True to stop the process
    """
    if parent is None:
        return False
    return params["max_depth"] <= parent.depth


def post_finish(params: Dict[str, Any], info: Dict[str, Any],
                left_info: Dict[str, Any], right_info: Dict[str, Any],
                split_info: Dict[str, Any], parent: TreeNode) -> bool:
    """Determines whether to process the process after making a split.

    :param params: parameters of the process
    :param info: step-related information
    :param left_info: information related with the left side of the split
    :param right_info: information related with the right side of the split
    :param split_info: information related with the split itself
    :param parent: the parent node of the current split
    :return: True if we are to stop the process.
    """
    if left_info["ematrix"].label.shape[0] < 2:
        return True
    if right_info["ematrix"].label.shape[0] < 2:
        return True
    return False


def getslice(arr: tf.Tensor, slice: tf.Tensor, axis: int) -> tf.Tensor:
    """Calculates a slice of an array.

    :param arr: an input array
    :param slice: boolean mask
    :param axis: axis to apply mask
    :return: result of the mask application along the axis
    """
    if arr is None:
        return None
    return tf.boolean_mask(arr, slice, axis=axis)


def split(
    *,
    bias,
    features,
    extra_features,
    label,
    ax,
    params,
    reduce_axis,
    unbalanced_penalty,
    use_extra,
    make_transpose=True,
) -> CommonResult:
    final_params = {"unbalanced_penalty": 0, "lambda": 1}
    final_params.update(params)
    y = tf.gather(label, ax)[:, :, 0]
    b = tf.gather(bias, ax)[:, :, 0]

    if use_extra:
        if reduce_axis == 0:
            extra = tf.gather(extra_features, ax, name="extra_features")
        else:
            extra = tf.gather(
                tf.transpose(a=extra_features),
                tf.transpose(a=ax),
                name="extra_features",
            )
            extra = tf.transpose(a=extra, perm=[1, 0, 2], name="textra_features")
    else:
        extra = None

    # F (N x M),  ax (N, M): ax_{ij} - pos in F_{*j},
    # ST_{i,j} = F_{ax_{i,j}, j}
    # sorted_thresholds = tf.gather(features, ax, axis=0)
    sorted_thresholds = take_along_axis(features, ax, reduce_axis=reduce_axis)
    common_result = common_part(
        y=y,
        b=b,
        sorted_thresholds=sorted_thresholds,
        features=features,
        label=label,
        bias=bias,
        extra=extra,
        unbalanced_penalty=unbalanced_penalty,
        reduce_axis=reduce_axis,
        make_transpose=make_transpose,
    )

    if reduce_axis == 0:
        best_feature = features[:, common_result.best_feature_index]
    else:
        best_feature = features[common_result.best_feature_index, :]

    common_result.left_cond = best_feature < common_result.thr
    common_result.right_cond = tf.logical_not(common_result.left_cond)
    common_result.ax_left = tf_new_ax(
        ax, common_result.left_cond, name="ax_left", reduce_axis=reduce_axis
    )
    common_result.ax_right = tf_new_ax(
        ax, common_result.right_cond, name="ax_right", reduce_axis=reduce_axis
    )

    common_result.extra_features = extra_features

    return common_result


def split_ematrix(
    ematrix: EMatrix,
    params: Dict[str, Any],
    reduce_axis: int,
    unbalanced_penalty: float,
    use_extra: bool,
):
    if ematrix.gax is not None:
        split_info = split(
            bias=ematrix.bias,
            features=ematrix.features,
            extra_features=ematrix.extra_features,
            label=ematrix.label,
            ax=ematrix.gax,
            params=params,
            reduce_axis=reduce_axis,
            unbalanced_penalty=unbalanced_penalty,
            use_extra=use_extra,
        )
        cond_left = split_info.left_cond
        cond_right = split_info.right_cond
        ax_left = split_info.ax_left
        ax_right = split_info.ax_right
    else:
        split_info = make_split(ematrix.bias, ematrix.features, ematrix.label)
        cond_left = best_feature < split_info["thr"]
        cond_right = np.logical_not(cond_left)
        ax_left = None
        ax_right = None

    features = ematrix.features
    extra_features = ematrix.extra_features
    bias = ematrix.bias
    label = ematrix.label

    left_ematrix = EMatrix(
        features=getslice(features, cond_left, reduce_axis),
        extra_features=getslice(extra_features, cond_left, reduce_axis),
        label=label[cond_left],
        bias=bias[cond_left],
        gax=ax_left,
    )
    right_ematrix = EMatrix(
        features=getslice(features, cond_right, reduce_axis),
        extra_features=getslice(extra_features, cond_right, reduce_axis),
        label=label[cond_right],
        bias=bias[cond_right],
        gax=ax_right,
    )
    left_info = {
        "prediction": split_info.best_delta_up,
        "ematrix": left_ematrix,
    }
    right_info = {
        "prediction": split_info.best_delta_down,
        "ematrix": right_ematrix,
    }
    return left_info, right_info, split_info


def build_tree_helper(
    params: Dict[str, Any],
    info: Dict[str, Any],
    parent: Optional[TreeNode],
    transposed_feature: bool,
    unbalanced_penalty: float,
    reduce_axis: int,
    use_extra: bool,
) -> Union[TreeNode, LeafData]:
    """Builds a tree.

    :param params: parameters of a build process
    :param info: basically EMatrix
    :param parent: the parent node of current step
    :param transposed_feature: whether our features are transposed or not
    :param unbalanced_penalty: penalty for different heights of subtrees
    :param reduce_axis: axis that enumerates objects in our dataset
    :param use_extra: whether we use extra features in our calculations
    :return: ready to use tree
    """
    info["learning_rate"] = params["learning_rate"]
    if False and parent and parent.depth < 6:
        print(
            "{d}".format(d=parent.depth) if parent else "---",
            "".format(shape=info["ematrix"].label.shape[0]),
            end=" ",
            file=sys.stderr,
        )
    if prior_finish(params, info, parent):
        return LeafData(info)
    node = TreeNode()
    node.depth = parent.depth + 1 if parent else 1

    left_info, right_info, split_info = split_ematrix(
        ematrix=info["ematrix"],
        params=params,
        reduce_axis=reduce_axis,
        unbalanced_penalty=unbalanced_penalty,
        use_extra=use_extra,
    )
    if post_finish(params, info, left_info, right_info, split_info, parent):
        # print(split_info['right_info']['ematrix'].label.shape[0])
        # print(split_info)
        return LeafData(info)

    node.val = SplitData(split_info)
    node.left = build_tree_helper(
        params,
        left_info,
        parent=node,
        transposed_feature=transposed_feature,
        unbalanced_penalty=unbalanced_penalty,
        reduce_axis=reduce_axis,
        use_extra=use_extra,
    )
    node.right = build_tree_helper(
        params,
        right_info,
        parent=node,
        transposed_feature=transposed_feature,
        unbalanced_penalty=unbalanced_penalty,
        reduce_axis=reduce_axis,
        use_extra=use_extra,
    )
    return node


def build_tree(
    params: Dict[str, Any],
    ematrix: EMatrix,
    transposed_feature: bool,
    unbalanced_penalty: float,
    reduce_axis: int,
    use_extra: bool,
) -> TreeNode:
    """Creates a classification tree.

    :param params: metaparameters of the training process
    :param ematrix: a dataset
    :param transposed_feature: whether our features are transposed
    :param unbalanced_penalty: penalty for different heights of our trees
    :param reduce_axis: the axis that enumerates objests in the dataset
    :param use_extra: whether there are extra features presented
    :return: ready to use classification-regression tree
    """
    info = {"ematrix": ematrix}
    if ematrix.splitgax and ematrix.gax is None:
        ematrix.gax = make_gax(ematrix.features, axis=reduce_axis)
    return build_tree_helper(
        params,
        transposed_feature=transposed_feature,
        unbalanced_penalty=unbalanced_penalty,
        info=info,
        parent=None,
        reduce_axis=reduce_axis,
        use_extra=use_extra,
    )


def assure_numpy(a: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
    """Assures that the output is np.ndarray

    :param a: either tf.Tensor or np.ndarray
    :return: np.ndarray
    """
    if isinstance(a, np.ndarray):
        return a
    return a.numpy()


def tree_apply(tree_arrays, features, extra_features=None, reduce_axis=0) -> np.ndarray:
    """Calculates the prediction of a tree.

    :param tree_arrays: arrays with decisions and data
    :param features: interpolating features
    :param extra_features: extrapolating features
    :param reduce_axis: the axis that enumerates objects of our dataset
    :return: the prediction of the classifier
    """
    qi = np.zeros(features.shape[reduce_axis], dtype=np.int32)
    for current_depth in range(tree_arrays["treedepth"]):
        fi = tree_arrays["features"][qi]
        f = np.choose(
            fi, assure_numpy(features).T if reduce_axis == 0 else assure_numpy(features)
        )  # TODO: try to do it in more effective tf-oriented way
        t = tree_arrays["thresholds"][qi]
        # print(qi, fi, f, t)
        # if current_depth == 0:
        #    print(fi, f.shape, features.shape, f)
        answer = (f < t) * 1
        new_qi = (
            answer * tree_arrays["yes_node"][qi]
            + (1 - answer) * tree_arrays["no_node"][qi]
        )
        qi = new_qi
    if extra_features is None:
        assert tree_arrays["leaf_data"].shape[1] == 1, "extra_features needed"
        leaf_data = tree_arrays["leaf_data"][qi, 0]
    else:
        leaf_data = assure_numpy(
            tree_arrays["leaf_data"][qi, :]
            * (extra_features.T if reduce_axis == 1 else extra_features)
        ).sum(
            axis=1
        )  # TODO: try to do it in more effective tf-oriented way
    return leaf_data


def tree2gv(tree: TreeNode) -> graphviz.Graph:
    """Represents a tree as a ready-to-render graph.

    :param tree: classification-regression tree
    :return: a ready-to render graph
    """
    result = graphviz.Graph("ni")
    # result.attr(size='12,0')
    tree2gv_helper(tree, result, "")
    return result


def tree2gv_helper(node: TreeNode, result: graphviz.Graph, id: int) -> None:
    """Performs a recursive step to render a graph.

    :param node: the current node to draw
    :param result: a graphviz graph
    :param id: the number of the current node
    """
    idn = id
    result.node(idn, node.to_text(), shape="box")  # node.shape())
    if isinstance(node, LeafData):
        return
    if node.left is not None:
        idl = id + "0"
        tree2gv_helper(node.left, result, idl)
        result.edge(idn, idl)
    if node.right is not None:
        idr = id + "1"
        tree2gv_helper(node.right, result, idr)
        result.edge(idn, idr)


class EBooster:
    """The main classifier class."""

    def __init__(self, forest):
        """Creates a classifier object."""
        self.forest = forest

    @classmethod
    def train(
        cls, params: Dict[str, Any], ematrix: EMatrix, num_boost_round: int = 10
    ) -> "EBooster":
        """Trains an ExtraBoost model.

        :param params: a dict of named train parameters
        :param ematrix: extra matrix with data
        :param num_boost_round: a number of rounds to train
        :return: a trained classifier
        """
        start_params = {
            "max_depth": 5,
            "learning_rate": 0.3,
            "splitgax": False,
            "transposed_feature": False,
            "progress_callback": None,
        }
        start_params.update(params)

        reduce_axis = 1 if start_params["transposed_feature"] else 0
        use_extra = ematrix.extra_features is not None

        if start_params["splitgax"] and ematrix.gax is None:
            ematrix.gax = make_gax(ematrix.features, axis=reduce_axis)

        forest = []
        bias = np.zeros(ematrix.label.shape)
        features = ematrix.features
        for r in range(num_boost_round):
            print(f"\n{r} round", file=sys.stderr)
            tree = build_tree(
                start_params,
                EMatrix(
                    features=ematrix.features,
                    label=ematrix.label,
                    bias=bias,
                    extra_features=ematrix.extra_features,
                    gax=ematrix.gax,
                    splitgax=start_params["splitgax"],
                ),
                # split_maker=split_maker,
                transposed_feature=start_params["transposed_feature"],
                unbalanced_penalty=start_params["unbalanced_penalty"],
                reduce_axis=reduce_axis,
                use_extra=use_extra,
            )
            # print("tree ok, bias shape = {}".format(bias.shape), file=sys.stderr)
            tree_arrays = init_arrays(
                root=tree,
                n=init_id(tree),
                weights_num=ematrix.extra_features.shape[1 - reduce_axis]
                if ematrix.extra_features is not None
                else 1,
            )
            bias_delta = tree_apply(
                tree_arrays=tree_arrays,
                features=features,
                extra_features=ematrix.extra_features,
                reduce_axis=reduce_axis,
            )
            # print("apply ok, bias delta shape = {}".format(bias_delta.shape), file=sys.stderr)
            bias = bias + np.reshape(bias_delta, newshape=bias.shape)
            forest.append((tree, tree_arrays))
            # print("forest appended", file=sys.stderr)
            if start_params["progress_callback"] is not None:
                start_params["progress_callback"](r, num_boost_round)

        return cls(forest)

    def predict(self, features, tree_limit=None, extra_features=None, reduce_axis=0):
        pred = np.zeros(features.shape[0], dtype=np.float32)
        for tree, tree_arrays in (
            self.forest if tree_limit is None else self.forest[:tree_limit]
        ):
            pred = pred + tree_apply(
                tree_arrays,
                features,
                extra_features=extra_features,
                reduce_axis=reduce_axis,
            )
        return pred
